"""BatchScorer — orchestrates the full batch scoring pipeline.

Flow:
  1. Fetch payments from ClickHouse (READ client) in a closed window
     [cursor_start, cursor_end]
  2. Build batch context via BatchContextProvider (6 queries total, READ client)
  3. Score each payment using SingleTransactionScorer internals
  4. INSERT scored rows to anomaly_scores (WRITE client) in 10K chunks with
     dedup tokens — guarded so writes never land on the READ (prod) target
  5. Return summary dict with processed/scored counts, critical alerts, and next_cursor

READ vs WRITE separation: cursor resolution, payment fetch and context building
run on a read-only client (production). The anomaly_scores INSERT runs on a
separate local write client. A guardrail (`assert_write_target_is_safe`) aborts
before inserting if the WRITE target matches the READ fingerprint or points to a
non-local host without an explicit bypass.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

from fraud_detector.features.engineering import FEATURE_NAMES
from fraud_detector.scoring.context import UserContext
from fraud_detector.scoring.scorer import SingleTransactionScorer
from fraud_detector.utils.currency import normalize_amount_value
from scorer.batch.context_provider import BatchContextProvider

# Default destination table. Local docker reuses the production DB *name*
# (pbp_productionDB_optimized) but lives on a separate local ClickHouse server,
# so the DB name alone never identifies production — the host fingerprint does.
DEFAULT_ANOMALY_SCORES_TABLE = "pbp_productionDB_optimized.anomaly_scores"

# Hostnames considered local for the WRITE guardrail.
_LOCAL_WRITE_HOSTS = {"clickhouse", "localhost", "127.0.0.1", "::1", "0.0.0.0"}

# Fingerprint identifying a ClickHouse connection target.
CHFingerprint = Tuple[str, int, str, bool, str]  # (host, port, database, secure, user)


def ch_fingerprint(host: str, port, database: str, secure, user: str) -> CHFingerprint:
    """Build a normalized fingerprint for a ClickHouse connection target."""
    return (
        str(host or "").strip().lower(),
        int(port or 0),
        str(database or "").strip().lower(),
        bool(secure),
        str(user or "").strip().lower(),
    )


def assert_write_target_is_safe(
    read_fingerprint: Optional[CHFingerprint],
    write_fingerprint: Optional[CHFingerprint],
    write_host: Optional[str],
    allow_nonlocal_write: bool,
) -> None:
    """Abort the batch INSERT unless the WRITE target is safe.

    Two independent checks:
      1. The WRITE fingerprint must differ from the READ fingerprint — otherwise
         a write would land on the production (read-only) server.
      2. The WRITE host must be local (docker/localhost) unless the explicit
         bypass flag ALLOW_NONLOCAL_ANOMALY_SCORE_WRITES is set.

    Raises:
        ValueError: if either check fails.
    """
    if read_fingerprint is not None and write_fingerprint is not None:
        if read_fingerprint == write_fingerprint:
            raise ValueError(
                "Refusing to insert anomaly_scores: WRITE target points at the "
                "same ClickHouse connection as READ (production). Configure "
                "ANOMALY_SCORES_CH_* to a local ClickHouse."
            )

    host_norm = str(write_host or "").strip().lower()
    if not allow_nonlocal_write and host_norm not in _LOCAL_WRITE_HOSTS:
        raise ValueError(
            f"Refusing to insert anomaly_scores: WRITE host {write_host!r} is "
            "non-local. For local runs use 'clickhouse', 'localhost' or "
            "'127.0.0.1'. To override deliberately set "
            "ALLOW_NONLOCAL_ANOMALY_SCORE_WRITES=true."
        )

# Column order must match anomaly_scores DDL
_INSERT_COLUMNS = [
    "payment_id",
    "facility_id",
    "facility_name",
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
    "scoring_mode",
    "feature_version",
    "threshold_version",
    "latency_ms",
    "error",
    "gateway",
    "payment_method",
    "currency",
    "source_enum",
]

# Fetch query — payments in a closed window [cursor_start, cursor_end].
# Using a closed window makes the payment set deterministic: retries see
# exactly the same rows and produce identical dedup tokens.
_FETCH_SQL = """
SELECT
    id AS payment_id,
    user_id,
    facility_id,
    facility_name,
    reservation_paid_out,
    created_at,
    discount,
    tip,
    payment_method,
    category,
    club_credit_flag,
    paid_by_manager,
    currency,
    status,
    effective_user_id,
    captured_at,
    gateway,
    source_enum
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

    READ operations (cursor resolution, payment fetch, batch context) use
    ``read_ch_client`` (production read-only). The anomaly_scores INSERT uses
    ``write_ch_client`` (local), guarded by ``assert_write_target_is_safe``.

    Args:
        scorer: Loaded SingleTransactionScorer instance (model + scaler + classifier).
        read_ch_client: clickhouse-connect client for READ (prod read-only).
        write_ch_client: clickhouse-connect client for WRITE (local). Defaults to
            ``read_ch_client`` only as a convenience; the guardrail then aborts
            unless fingerprints differ and the host is local.
        anomaly_scores_table: Fully-qualified destination table for the INSERT.
        read_fingerprint: Connection fingerprint of the READ client (for guardrail).
        write_fingerprint: Connection fingerprint of the WRITE client (for guardrail).
        write_host: Host of the WRITE client (for the local-host guardrail).
        allow_nonlocal_write: Explicit bypass for the non-local host check.
        context_chunk_size: Max pairs per VALUES JOIN chunk (default 2000).
        insert_chunk_size: Rows per INSERT batch (default 10000).
    """

    def __init__(
        self,
        scorer: SingleTransactionScorer,
        read_ch_client,
        write_ch_client=None,
        *,
        anomaly_scores_table: str = DEFAULT_ANOMALY_SCORES_TABLE,
        read_fingerprint: Optional[CHFingerprint] = None,
        write_fingerprint: Optional[CHFingerprint] = None,
        write_host: Optional[str] = None,
        allow_nonlocal_write: bool = False,
        context_chunk_size: int = 2000,
        insert_chunk_size: int = 10_000,
    ):
        self._scorer = scorer
        self._read_ch = read_ch_client
        self._write_ch = write_ch_client if write_ch_client is not None else read_ch_client
        self._anomaly_scores_table = anomaly_scores_table
        self._read_fingerprint = read_fingerprint
        self._write_fingerprint = write_fingerprint
        self._write_host = write_host
        self._allow_nonlocal_write = allow_nonlocal_write
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

        # Fail fast: validate the WRITE target before touching production data.
        assert_write_target_is_safe(
            read_fingerprint=self._read_fingerprint,
            write_fingerprint=self._write_fingerprint,
            write_host=self._write_host,
            allow_nonlocal_write=self._allow_nonlocal_write,
        )

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
        ctx_provider = BatchContextProvider(self._read_ch, chunk_size=self._context_chunk_size)
        ctx_map = ctx_provider.get_batch_context(payments)
        logger.info(
            f"BatchScorer: context built in {time.monotonic() - t_ctx:.2f}s "
            f"for {len(ctx_map)} unique pairs"
        )

        # Step 3: Score each payment
        model_version = getattr(self._scorer, "_model_version", "IF-31-v1")
        scored_rows, critical_alerts = self._score_all(payments, ctx_map, model_version)

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
        result = self._read_ch.query(_CURSOR_END_SQL, parameters={"cursor_start": cursor_start})
        rows = result.result_rows
        if not rows or rows[0][0] is None:
            return None
        return rows[0][0]

    def _fetch_payments(self, cursor_start: datetime, cursor_end: datetime) -> List[dict]:
        """Fetch all eligible payments in the closed window [cursor_start, cursor_end]."""
        result = self._read_ch.query(
            _FETCH_SQL,
            parameters={"cursor_start": cursor_start, "cursor_end": cursor_end},
        )
        column_names = [
            "payment_id",
            "user_id",
            "facility_id",
            "facility_name",
            "reservation_paid_out",
            "created_at",
            "discount",
            "tip",
            "payment_method",
            "category",
            "club_credit_flag",
            "paid_by_manager",
            "currency",
            "status",
            "effective_user_id",
            "captured_at",
            "gateway",
            "source_enum",
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
            feature_names = getattr(scorer, "_feature_names", FEATURE_NAMES)
            if len(feature_names) == 40:
                context = scorer._context_provider.get_context(
                    user_id=uid,
                    facility_id=fid,
                    timestamp=payment["created_at"],
                    payment=payment,
                )

            row_started = time.monotonic()
            error = ""
            try:
                features: np.ndarray = scorer._feature_calc.calculate(payment, context)
                raw_score, X_scaled = scorer.score_features(features)
                is_anomaly, risk_level, percentile = scorer._classifier.classify(raw_score)
                factors = SingleTransactionScorer._explain_top_factors(
                    features,
                    X_scaled[0],
                    top_n=5,
                    feature_names=feature_names,
                )
            except Exception as exc:
                logger.warning(
                    f"BatchScorer: failed to score payment_id={payment.get('payment_id')}: {exc}"
                )
                error = str(exc)
                features = np.zeros(len(feature_names), dtype=np.float32)
                raw_score = 0.0
                percentile = 0.0
                risk_level = "minimal"
                is_anomaly = False
                factors = []

            latency_ms = (time.monotonic() - row_started) * 1000.0
            amount_usd = normalize_amount_value(
                payment.get("reservation_paid_out"),
                payment.get("currency"),
            )

            # Build row in _INSERT_COLUMNS order
            row = [
                int(payment["payment_id"]),  # payment_id UInt64
                fid,  # facility_id UInt32
                str(payment.get("facility_name") or ""),  # facility_name String
                uid,  # user_id UInt64
                datetime.now(tz=timezone.utc).replace(tzinfo=None),  # scored_at DateTime
                payment["created_at"],  # payment_created_at DateTime
                amount_usd,  # amount_usd Float32
                raw_score,  # raw_score Float32
                float(percentile),  # percentile Float32
                risk_level,  # risk_level LowCardinality(String)
                int(is_anomaly),  # is_anomaly UInt8 — NOT bool
                model_version,  # model_version LowCardinality(String)
                json.dumps(factors),  # top_factors String
                json.dumps(dict(zip(feature_names, features.tolist()))),
                os.environ.get("SCORING_MODE", "active"),
                getattr(scorer, "_feature_version", "base-31"),
                getattr(scorer, "_threshold_version", "v1"),
                float(latency_ms),
                error,
                str(payment.get("gateway") or ""),  # gateway
                str(payment.get("payment_method") or ""),  # payment_method
                str(payment.get("currency") or ""),  # currency
                str(payment.get("source_enum") or ""),  # source_enum (channel)
            ]
            scored_rows.append(row)

            # Collect critical alerts for response
            if risk_level == "critical":
                critical_alerts.append(
                    {
                        "payment_id": int(payment["payment_id"]),
                        "user_id": uid,
                        "facility_id": fid,
                        "raw_score": raw_score,
                        "risk_level": risk_level,
                        "amount_usd": amount_usd,
                        "model_version": model_version,
                        "feature_version": getattr(scorer, "_feature_version", "base-31"),
                        "threshold_version": getattr(scorer, "_threshold_version", "v1"),
                    }
                )

        scoring_elapsed = time.monotonic() - t_score
        logger.info(f"BatchScorer: scored {len(scored_rows)} payments in {scoring_elapsed:.2f}s")
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
        # Hard safety gate: never let an INSERT reach the production (READ) target.
        assert_write_target_is_safe(
            read_fingerprint=self._read_fingerprint,
            write_fingerprint=self._write_fingerprint,
            write_host=self._write_host,
            allow_nonlocal_write=self._allow_nonlocal_write,
        )

        chunk_size = self._insert_chunk_size
        total = len(scored_rows)
        num_chunks = (total + chunk_size - 1) // chunk_size

        logger.info(
            f"BatchScorer: inserting {total} rows into {self._anomaly_scores_table} "
            f"in {num_chunks} chunks of {chunk_size}"
        )

        for chunk_index, chunk_start in enumerate(range(0, total, chunk_size)):
            chunk = scored_rows[chunk_start : chunk_start + chunk_size]
            token = (
                f"batch-{cursor.isoformat()}-{cursor_end.isoformat()}"
                f"-{model_version}-chunk-{chunk_index}"
            )

            t_ins = time.monotonic()
            self._write_ch.insert(
                self._anomaly_scores_table,
                chunk,
                column_names=_INSERT_COLUMNS,
                settings={"insert_deduplication_token": token},
            )
            logger.debug(
                f"BatchScorer: inserted chunk {chunk_index}/{num_chunks - 1} "
                f"({len(chunk)} rows, token={token!r}) "
                f"in {time.monotonic() - t_ins:.3f}s"
            )
