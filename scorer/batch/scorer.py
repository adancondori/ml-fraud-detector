"""BatchScorer — orchestrates the full batch scoring pipeline.

Flow:
  1. Fetch payments from ClickHouse in a closed window [cursor_start, cursor_end]
  2. Build batch context via BatchContextProvider (6 queries total)
  3. Score each payment using SingleTransactionScorer internals
  4. INSERT scored rows to anomaly_scores in 10K chunks with dedup tokens
  5. Return summary dict with processed/scored counts, critical alerts, and next_cursor
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

from fraud_detector.features.engineering import FEATURE_NAMES
from fraud_detector.scoring.context import UserContext
from fraud_detector.scoring.scorer import SingleTransactionScorer
from scorer.batch.context_provider import BatchContextProvider

# Column order must match anomaly_scores DDL
_INSERT_COLUMNS = [
    "payment_id",
    "facility_id",
    "user_id",
    "scored_at",
    "payment_created_at",
    "amount_usd",
    "raw_score",
    "percentile",
    "risk_level",
    "is_anomaly",
    "model_version",
    "top_factors",
    "features_json",
]

# Fetch query — payments in a closed window [cursor_start, cursor_end].
# Using a closed window makes the payment set deterministic: retries see
# exactly the same rows and produce identical dedup tokens.
_FETCH_SQL = """
SELECT
    payment_id,
    user_id,
    facility_id,
    reservation_paid_out,
    created_at,
    discount,
    tip,
    payment_method,
    category,
    club_credit_flag,
    paid_by_manager,
    currency,
    status
FROM pbp_productionDB_optimized.payments FINAL
WHERE created_at >= {cursor_start:DateTime}
  AND created_at <= {cursor_end:DateTime}
  AND _peerdb_is_deleted = 0
  AND payment_method NOT IN ('reversal', 'free')
ORDER BY created_at ASC
"""

# Query to resolve cursor_end as the max created_at available since cursor_start.
# Running this first pins the upper bound before we fetch the full result set.
_CURSOR_END_SQL = """
SELECT max(created_at)
FROM pbp_productionDB_optimized.payments FINAL
WHERE created_at >= {cursor_start:DateTime}
  AND _peerdb_is_deleted = 0
  AND payment_method NOT IN ('reversal', 'free')
"""


class BatchScorer:
    """Orchestrates batch scoring: fetch -> context -> score -> INSERT.

    Args:
        scorer: Loaded SingleTransactionScorer instance (model + scaler + classifier).
        ch_client: Raw clickhouse-connect client (thread-safe).
        context_chunk_size: Max pairs per VALUES JOIN chunk (default 2000).
        insert_chunk_size: Rows per INSERT batch (default 10000).
    """

    def __init__(
        self,
        scorer: SingleTransactionScorer,
        ch_client,
        context_chunk_size: int = 2000,
        insert_chunk_size: int = 10_000,
    ):
        self._scorer = scorer
        self._ch = ch_client
        self._context_chunk_size = context_chunk_size
        self._insert_chunk_size = insert_chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_batch(self, cursor: datetime) -> Dict:
        """Score all payments in a closed window [cursor, cursor_end] and insert.

        The upper bound (cursor_end) is pinned before fetching so retries see
        the same payment set, making dedup tokens fully deterministic.

        Args:
            cursor: Lower bound — fetch payments where created_at >= cursor.

        Returns:
            dict with keys:
              - processed (int): total payments fetched
              - scored (int): total payments scored and inserted
              - critical_alerts (list[dict]): payments with risk_level == "critical"
              - next_cursor (datetime | None): cursor_end + 1 second, or None if no
                payments were found; the Rails client should use this as the next
                cursor to avoid re-processing the same window.
        """
        t_start = time.monotonic()
        logger.info(f"BatchScorer: starting batch from cursor={cursor.isoformat()}")

        # Step 1: Pin cursor_end then fetch payments in the closed window
        cursor_end = self._resolve_cursor_end(cursor)
        if cursor_end is None:
            logger.info("BatchScorer: no payments found since cursor — nothing to do")
            return {"processed": 0, "scored": 0, "critical_alerts": [], "next_cursor": None}

        payments = self._fetch_payments(cursor, cursor_end)
        total_fetched = len(payments)
        logger.info(
            f"BatchScorer: fetched {total_fetched} payments "
            f"[{cursor.isoformat()} … {cursor_end.isoformat()}]"
        )

        if not payments:
            return {"processed": 0, "scored": 0, "critical_alerts": [], "next_cursor": None}

        # Step 2: Build batch context (6 queries total)
        t_ctx = time.monotonic()
        ctx_provider = BatchContextProvider(self._ch, chunk_size=self._context_chunk_size)
        ctx_map = ctx_provider.get_batch_context(payments)
        logger.info(
            f"BatchScorer: context built in {time.monotonic() - t_ctx:.2f}s "
            f"for {len(ctx_map)} unique pairs"
        )

        # Step 3: Score each payment
        model_version = getattr(self._scorer._model, "_version", "if-31-v1")
        scored_rows, critical_alerts = self._score_all(
            payments, ctx_map, model_version
        )

        # Step 4: INSERT in 10K chunks with dedup tokens
        total_scored = len(scored_rows)
        self._insert_chunks(scored_rows, cursor, cursor_end, model_version)

        elapsed = time.monotonic() - t_start
        rate = total_scored / elapsed if elapsed > 0 else 0
        logger.info(
            f"BatchScorer: complete — {total_scored}/{total_fetched} scored "
            f"in {elapsed:.1f}s ({rate:.0f} txn/s), "
            f"{len(critical_alerts)} critical alerts"
        )
        next_cursor = cursor_end + timedelta(seconds=1)
        return {
            "processed": total_fetched,
            "scored": total_scored,
            "critical_alerts": critical_alerts,
            "next_cursor": next_cursor,
        }

    # ------------------------------------------------------------------
    # Step 1: Resolve cursor_end, then fetch payments in closed window
    # ------------------------------------------------------------------

    def _resolve_cursor_end(self, cursor_start: datetime) -> Optional[datetime]:
        """Return the max created_at available since cursor_start, or None."""
        result = self._ch.query(
            _CURSOR_END_SQL, parameters={"cursor_start": cursor_start}
        )
        rows = result.result_rows
        if not rows or rows[0][0] is None:
            return None
        return rows[0][0]

    def _fetch_payments(self, cursor_start: datetime, cursor_end: datetime) -> List[dict]:
        """Fetch all eligible payments in the closed window [cursor_start, cursor_end]."""
        result = self._ch.query(
            _FETCH_SQL,
            parameters={"cursor_start": cursor_start, "cursor_end": cursor_end},
        )
        column_names = [
            "payment_id", "user_id", "facility_id", "reservation_paid_out",
            "created_at", "discount", "tip", "payment_method", "category",
            "club_credit_flag", "paid_by_manager", "currency", "status",
        ]
        payments = []
        for row in result.result_rows:
            payments.append(dict(zip(column_names, row)))
        return payments

    # ------------------------------------------------------------------
    # Step 3: Score all payments
    # ------------------------------------------------------------------

    def _score_all(
        self,
        payments: List[dict],
        ctx_map: Dict,
        model_version: str,
    ):
        """Score each payment and collect rows for INSERT.

        Args:
            payments: Payment dicts from _fetch_payments.
            ctx_map: Context map from BatchContextProvider.
            model_version: Version string already resolved by score_batch.

        Returns:
            (scored_rows, critical_alerts)
        """
        scorer = self._scorer
        scored_rows = []
        critical_alerts = []

        t_score = time.monotonic()
        for payment in tqdm(payments, desc="Scoring", unit="txn", leave=False):
            uid = int(payment["user_id"])
            fid = int(payment["facility_id"])
            context: UserContext = ctx_map.get((uid, fid), UserContext())

            # Compute 31 features
            features: np.ndarray = scorer._feature_calc.calculate(payment, context)

            # Scale
            X_scaled = scorer._scaler.scaler.transform(
                features.reshape(1, -1)
            ).astype(np.float32)

            # Score (higher = more anomalous)
            raw_score = float(-scorer._model.score_samples(X_scaled)[0])

            # Classify
            is_anomaly, risk_level, percentile = scorer._classifier.classify(raw_score)

            # Explain (top 5 factors by absolute z-score)
            factors = SingleTransactionScorer._explain_top_factors(
                features, X_scaled[0], top_n=5
            )

            # amount_usd: reservation_paid_out is already in USD after CurrencyNormalizer
            amount_usd = float(payment.get("reservation_paid_out") or 0)

            # Build row in _INSERT_COLUMNS order
            row = [
                int(payment["payment_id"]),        # payment_id UInt64
                fid,                                # facility_id UInt32
                uid,                                # user_id UInt64
                datetime.now(tz=timezone.utc).replace(tzinfo=None),  # scored_at DateTime
                payment["created_at"],              # payment_created_at DateTime
                amount_usd,                         # amount_usd Float32
                raw_score,                          # raw_score Float32
                float(percentile),                  # percentile Float32
                risk_level,                         # risk_level LowCardinality(String)
                int(is_anomaly),                    # is_anomaly UInt8 — NOT bool
                model_version,                      # model_version LowCardinality(String)
                json.dumps(factors),                # top_factors String
                json.dumps(                         # features_json String — ALL 31 features
                    dict(zip(FEATURE_NAMES, features.tolist()))
                ),
            ]
            scored_rows.append(row)

            # Collect critical alerts for response
            if risk_level == "critical":
                critical_alerts.append({
                    "payment_id": int(payment["payment_id"]),
                    "user_id": uid,
                    "facility_id": fid,
                    "raw_score": raw_score,
                    "risk_level": risk_level,
                    "amount_usd": amount_usd,
                })

        scoring_elapsed = time.monotonic() - t_score
        logger.info(
            f"BatchScorer: scored {len(scored_rows)} payments in {scoring_elapsed:.2f}s"
        )
        return scored_rows, critical_alerts

    # ------------------------------------------------------------------
    # Step 4: INSERT in chunks
    # ------------------------------------------------------------------

    def _insert_chunks(
        self,
        scored_rows: List[list],
        cursor: datetime,
        cursor_end: datetime,
        model_version: str,
    ) -> None:
        """INSERT scored rows to anomaly_scores in insert_chunk_size batches.

        Each chunk uses a fully deterministic dedup token:
            f"batch-{cursor_start}-{cursor_end}-{model_version}-chunk-{chunk_index}"

        Including cursor_end and model_version ensures tokens are stable across
        retries even if a different model is loaded between calls, and eliminates
        the non-determinism that arose when new payments arrived in an open window.
        """
        chunk_size = self._insert_chunk_size
        total = len(scored_rows)
        num_chunks = (total + chunk_size - 1) // chunk_size

        logger.info(
            f"BatchScorer: inserting {total} rows in {num_chunks} chunks "
            f"of {chunk_size}"
        )

        for chunk_index, chunk_start in enumerate(range(0, total, chunk_size)):
            chunk = scored_rows[chunk_start: chunk_start + chunk_size]
            token = (
                f"batch-{cursor.isoformat()}-{cursor_end.isoformat()}"
                f"-{model_version}-chunk-{chunk_index}"
            )

            t_ins = time.monotonic()
            self._ch.insert(
                "pbp_productionDB_optimized.anomaly_scores",
                chunk,
                column_names=_INSERT_COLUMNS,
                settings={"insert_deduplication_token": token},
            )
            logger.debug(
                f"BatchScorer: inserted chunk {chunk_index}/{num_chunks - 1} "
                f"({len(chunk)} rows, token={token!r}) "
                f"in {time.monotonic() - t_ins:.3f}s"
            )
