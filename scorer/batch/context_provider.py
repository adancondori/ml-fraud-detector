"""BatchContextProvider — fetches rolling aggregates for a batch of payments.

Uses a VALUES JOIN strategy to execute exactly 6 ClickHouse queries regardless
of batch size (not 6xN). Suitable for batch scoring pipelines with thousands
of payments per cursor run.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Tuple

from loguru import logger

from fraud_detector.scoring.context import UserContext
from fraud_detector.utils.currency import clickhouse_rate_case

# Type alias for the context key
ContextKey = Tuple[int, int]  # (user_id, facility_id)
P_AMOUNT_USD_SQL = f"(p.reservation_paid_out * {clickhouse_rate_case('p.currency')})"
P_DISCOUNT_USD_SQL = f"(p.discount * {clickhouse_rate_case('p.currency')})"


class BatchContextProvider:
    """Fetches user context aggregates for a batch of payments using VALUES JOIN.

    Executes exactly 6 queries per chunk — not 6 per payment.
    Total query count = 6 * ceil(unique_pairs / chunk_size).

    Args:
        client: Raw clickhouse-connect client instance.
        chunk_size: Max number of (user_id, facility_id) pairs per VALUES clause.
                    Default 2000 keeps VALUES clause well under ClickHouse max_query_size.
    """

    def __init__(self, client, chunk_size: int = 2000):
        self._client = client
        self._chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_batch_context(self, payments: List[dict]) -> Dict[ContextKey, UserContext]:
        """Fetch rolling aggregates for all unique (user_id, facility_id) pairs.

        Args:
            payments: List of payment dicts, each containing at minimum:
                      user_id (int), facility_id (int), created_at (datetime).

        Returns:
            Dict keyed by (user_id, facility_id) -> UserContext with populated
            aggregate fields. Missing pairs return a zero-valued UserContext.
        """
        if not payments:
            return {}

        # Deduplicate pairs and take max timestamp per pair
        pair_timestamps: Dict[ContextKey, datetime] = {}
        for p in payments:
            uid = int(p["user_id"])
            fid = int(p["facility_id"])
            ts = p["created_at"]
            if not isinstance(ts, datetime):
                import pandas as pd

                ts = pd.Timestamp(ts).to_pydatetime()
            key = (uid, fid)
            if key not in pair_timestamps or ts > pair_timestamps[key]:
                pair_timestamps[key] = ts

        # Initialize result dict with default UserContext for all pairs
        result: Dict[ContextKey, UserContext] = {key: UserContext() for key in pair_timestamps}

        # Process in chunks to stay under ClickHouse max_query_size
        pairs_list = list(pair_timestamps.items())  # [((uid, fid), ts), ...]
        total_pairs = len(pairs_list)
        logger.info(
            f"BatchContextProvider: {total_pairs} unique (user_id, facility_id) pairs, "
            f"chunk_size={self._chunk_size}"
        )

        for chunk_start in range(0, total_pairs, self._chunk_size):
            chunk = pairs_list[chunk_start : chunk_start + self._chunk_size]
            chunk_result = self._query_chunk(chunk)
            result.update(chunk_result)

        logger.info(
            f"BatchContextProvider: context fetched for {len(result)} pairs "
            f"across {6 * ((total_pairs - 1) // self._chunk_size + 1)} total queries"
        )
        return result

    # ------------------------------------------------------------------
    # Internal: VALUES clause builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_values_str(chunk: List[Tuple[ContextKey, datetime]]) -> str:
        """Build VALUES clause from typed Python values — safe, no raw string input."""
        return ", ".join(
            f"({int(uid)}, {int(fid)}, '{ts.strftime('%Y-%m-%d %H:%M:%S')}')"
            for (uid, fid), ts in chunk
        )

    # ------------------------------------------------------------------
    # Internal: Per-chunk 6-query execution
    # ------------------------------------------------------------------

    def _query_chunk(
        self, chunk: List[Tuple[ContextKey, datetime]]
    ) -> Dict[ContextKey, UserContext]:
        """Run all 6 queries for one chunk of pairs, return populated context dict."""
        # Initialize results for this chunk
        chunk_result: Dict[ContextKey, UserContext] = {key: UserContext() for key, _ in chunk}
        values_str = self._build_values_str(chunk)

        self._query_velocity(values_str, chunk_result)
        self._query_behavior(values_str, chunk_result)
        self._query_credit(values_str, chunk_result)
        self._query_diversity(values_str, chunk_result)
        self._query_user_info(values_str, chunk_result)
        self._query_role(values_str, chunk_result)

        return chunk_result

    # ------------------------------------------------------------------
    # Query 1: Velocity (Group C)
    # ------------------------------------------------------------------

    def _query_velocity(self, values_str: str, ctx_map: Dict[ContextKey, UserContext]) -> None:
        """txn_count_1h, txn_count_24h, amount_24h, last_txn_at."""
        sql = f"""
        SELECT
            v.user_id,
            v.facility_id,
            countIf(
                p.created_at >= v.ts - INTERVAL 1 HOUR
                AND p.created_at < v.ts
            ) AS txn_count_1h,
            countIf(
                p.created_at >= v.ts - INTERVAL 24 HOUR
                AND p.created_at < v.ts
            ) AS txn_count_24h,
            sumIf(
                {P_AMOUNT_USD_SQL},
                p.created_at >= v.ts - INTERVAL 24 HOUR
                AND p.created_at < v.ts
            ) AS amount_24h,
            max(CASE WHEN p.created_at < v.ts THEN p.created_at ELSE NULL END) AS last_txn_at
        FROM pbp_productionDB_optimized.payments FINAL AS p
        INNER JOIN (
            SELECT * FROM VALUES(
                'user_id UInt64, facility_id UInt32, ts DateTime',
                {values_str}
            )
        ) AS v ON p.user_id = v.user_id
        WHERE p._peerdb_is_deleted = 0
          AND p.payment_method NOT IN ('reversal', 'free')
        GROUP BY v.user_id, v.facility_id
        """
        t0 = time.monotonic()
        try:
            result = self._client.query(sql)
            for row in result.result_rows:
                key = (int(row[0]), int(row[1]))
                if key in ctx_map:
                    ctx_map[key].txn_count_1h = int(row[2] or 0)
                    ctx_map[key].txn_count_24h = int(row[3] or 0)
                    ctx_map[key].amount_24h = float(row[4] or 0)
                    ctx_map[key].last_txn_at = row[5]
        except Exception as exc:
            logger.warning(f"BatchContextProvider velocity query failed: {exc}")
        logger.debug(f"velocity query: {time.monotonic() - t0:.3f}s, {len(ctx_map)} pairs")

    # ------------------------------------------------------------------
    # Query 2: Behavior (Group D)
    # ------------------------------------------------------------------

    def _query_behavior(self, values_str: str, ctx_map: Dict[ContextKey, UserContext]) -> None:
        """distinct_facilities_30d, distinct_methods, reversal_ratio_30d,
        discount_ratio_30d, txn_count_30d."""
        sql = f"""
        SELECT
            v.user_id,
            v.facility_id,
            count(DISTINCT p.facility_id) AS distinct_facilities_30d,
            count(DISTINCT p.payment_method) AS distinct_methods,
            countIf(p.status IN ('totally_refunded', 'refunded_to_credit'))
                * 1.0 / greatest(count(), 1) AS reversal_ratio_30d,
            sumIf({P_DISCOUNT_USD_SQL}, 1=1) / greatest(sumIf({P_AMOUNT_USD_SQL}, 1=1), 0.01)
                AS discount_ratio_30d,
            count() AS txn_count_30d
        FROM pbp_productionDB_optimized.payments FINAL AS p
        INNER JOIN (
            SELECT * FROM VALUES(
                'user_id UInt64, facility_id UInt32, ts DateTime',
                {values_str}
            )
        ) AS v ON p.user_id = v.user_id
        WHERE p.created_at >= v.ts - INTERVAL 30 DAY
          AND p.created_at < v.ts
          AND p._peerdb_is_deleted = 0
          AND p.payment_method NOT IN ('reversal', 'free')
        GROUP BY v.user_id, v.facility_id
        """
        t0 = time.monotonic()
        try:
            result = self._client.query(sql)
            for row in result.result_rows:
                key = (int(row[0]), int(row[1]))
                if key in ctx_map:
                    ctx_map[key].distinct_facilities_30d = int(row[2] or 0)
                    ctx_map[key].distinct_methods = int(row[3] or 0)
                    ctx_map[key].reversal_ratio_30d = float(row[4] or 0)
                    ctx_map[key].discount_ratio_30d = float(row[5] or 0)
                    ctx_map[key].txn_count_30d = int(row[6] or 0)
        except Exception as exc:
            logger.warning(f"BatchContextProvider behavior query failed: {exc}")
        logger.debug(f"behavior query: {time.monotonic() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Query 3: Credit/Flow (Group F)
    # ------------------------------------------------------------------

    def _query_credit(self, values_str: str, ctx_map: Dict[ContextKey, UserContext]) -> None:
        """debit_count_30d, debit_amount_30d, prepaid_spend_30d."""
        sql = f"""
        SELECT
            v.user_id,
            v.facility_id,
            countIf(p.category = 'debit') AS debit_count_30d,
            sumIf({P_AMOUNT_USD_SQL}, p.category = 'debit') AS debit_amount_30d,
            sumIf({P_AMOUNT_USD_SQL}, p.payment_method = 'prepaid') AS prepaid_spend_30d
        FROM pbp_productionDB_optimized.payments FINAL AS p
        INNER JOIN (
            SELECT * FROM VALUES(
                'user_id UInt64, facility_id UInt32, ts DateTime',
                {values_str}
            )
        ) AS v ON p.user_id = v.user_id
        WHERE p.created_at >= v.ts - INTERVAL 30 DAY
          AND p.created_at < v.ts
          AND p._peerdb_is_deleted = 0
        GROUP BY v.user_id, v.facility_id
        """
        t0 = time.monotonic()
        try:
            result = self._client.query(sql)
            for row in result.result_rows:
                key = (int(row[0]), int(row[1]))
                if key in ctx_map:
                    ctx_map[key].debit_count_30d = int(row[2] or 0)
                    ctx_map[key].debit_amount_30d = float(row[3] or 0)
                    ctx_map[key].prepaid_spend_30d = float(row[4] or 0)
        except Exception as exc:
            logger.warning(f"BatchContextProvider credit query failed: {exc}")
        logger.debug(f"credit query: {time.monotonic() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Query 4: Diversity (Group H)
    # ------------------------------------------------------------------

    def _query_diversity(self, values_str: str, ctx_map: Dict[ContextKey, UserContext]) -> None:
        """categories_30d (groupArray), reversal_count_30d, merchandise_ratio_30d."""
        sql = f"""
        SELECT
            v.user_id,
            v.facility_id,
            groupArray(p.category) AS categories_30d,
            countIf(p.status IN ('totally_refunded', 'refunded_to_credit'))
                AS reversal_count_30d,
            countIf(p.category = 'merchandise') * 1.0
                / greatest(count(), 1) AS merchandise_ratio_30d
        FROM pbp_productionDB_optimized.payments FINAL AS p
        INNER JOIN (
            SELECT * FROM VALUES(
                'user_id UInt64, facility_id UInt32, ts DateTime',
                {values_str}
            )
        ) AS v ON p.user_id = v.user_id
        WHERE p.created_at >= v.ts - INTERVAL 30 DAY
          AND p.created_at < v.ts
          AND p._peerdb_is_deleted = 0
          AND p.payment_method NOT IN ('reversal', 'free')
        GROUP BY v.user_id, v.facility_id
        """
        t0 = time.monotonic()
        try:
            result = self._client.query(sql)
            for row in result.result_rows:
                key = (int(row[0]), int(row[1]))
                if key in ctx_map:
                    ctx_map[key].categories_30d = list(row[2] or [])
                    ctx_map[key].reversal_count_30d = int(row[3] or 0)
                    ctx_map[key].merchandise_ratio_30d = float(row[4] or 0)
        except Exception as exc:
            logger.warning(f"BatchContextProvider diversity query failed: {exc}")
        logger.debug(f"diversity query: {time.monotonic() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Query 5: User info
    # ------------------------------------------------------------------

    def _query_user_info(self, values_str: str, ctx_map: Dict[ContextKey, UserContext]) -> None:
        """user_created_at from users table (join on user_id only)."""
        # Deduplicate user_ids for this query — no facility_id needed
        # Build a VALUES clause with only user_ids to avoid duplicates across facilities
        unique_users_str = ", ".join(f"({int(uid)})" for uid in {key[0] for key in ctx_map})
        sql = f"""
        SELECT
            u.id AS user_id,
            u.created_at AS user_created_at
        FROM pbp_productionDB_optimized.users FINAL AS u
        INNER JOIN (
            SELECT * FROM VALUES(
                'user_id UInt64',
                {unique_users_str}
            )
        ) AS v ON u.id = v.user_id
        WHERE u._peerdb_is_deleted = 0
        """
        t0 = time.monotonic()
        try:
            result = self._client.query(sql)
            user_created: dict = {}
            for row in result.result_rows:
                user_created[int(row[0])] = row[1]
            # Apply to all (user_id, facility_id) pairs for this user
            for uid, fid in ctx_map:
                if uid in user_created:
                    ctx_map[(uid, fid)].user_created_at = user_created[uid]
        except Exception as exc:
            logger.warning(f"BatchContextProvider user_info query failed: {exc}")
        logger.debug(f"user_info query: {time.monotonic() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Query 6: Role
    # ------------------------------------------------------------------

    def _query_role(self, values_str: str, ctx_map: Dict[ContextKey, UserContext]) -> None:
        """role from facilities_users table, keyed by (user_id, facility_id)."""
        sql = f"""
        SELECT
            fu.user_id,
            fu.facility_id,
            fu.role
        FROM pbp_productionDB_optimized.facilities_users FINAL AS fu
        INNER JOIN (
            SELECT * FROM VALUES(
                'user_id UInt64, facility_id UInt32, ts DateTime',
                {values_str}
            )
        ) AS v ON fu.user_id = v.user_id AND fu.facility_id = v.facility_id
        WHERE fu._peerdb_is_deleted = 0
        """
        t0 = time.monotonic()
        try:
            result = self._client.query(sql)
            for row in result.result_rows:
                key = (int(row[0]), int(row[1]))
                if key in ctx_map and row[2]:
                    ctx_map[key].user_role = str(row[2])
        except Exception as exc:
            logger.warning(f"BatchContextProvider role query failed: {exc}")
        logger.debug(f"role query: {time.monotonic() - t0:.3f}s")
