"""User context provider — fetches rolling aggregates from ClickHouse for a single transaction."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from fraud_detector.utils.logger import logger


@dataclass
class UserContext:
    """Rolling aggregates for a single user at a point in time."""

    # Velocity (Group C)
    txn_count_1h: int = 0
    txn_count_24h: int = 0
    amount_24h: float = 0.0
    last_txn_at: Optional[datetime] = None

    # Behavior (Group D)
    distinct_facilities_30d: int = 0
    distinct_methods: int = 0
    reversal_ratio_30d: float = 0.0
    discount_ratio_30d: float = 0.0
    txn_count_30d: int = 0

    # Credit/Flow (Group F)
    debit_count_30d: int = 0
    debit_amount_30d: float = 0.0
    prepaid_spend_30d: float = 0.0

    # Diversity (Group H)
    categories_30d: List[str] = field(default_factory=list)
    reversal_count_30d: int = 0
    merchandise_ratio_30d: float = 0.0

    # User info
    user_created_at: Optional[datetime] = None
    user_role: str = "player"


class UserContextProvider:
    """Fetches user context from ClickHouse for single-transaction scoring.

    Each query is lightweight (filtered by user_id with index).
    Total latency: ~50-200ms for 5 queries.
    """

    VELOCITY_SQL = """
        SELECT
            countIf(created_at >= {ts:DateTime} - INTERVAL 1 HOUR
                    AND created_at < {ts:DateTime}) AS txn_count_1h,
            countIf(created_at >= {ts:DateTime} - INTERVAL 24 HOUR
                    AND created_at < {ts:DateTime}) AS txn_count_24h,
            sumIf(reservation_paid_out,
                  created_at >= {ts:DateTime} - INTERVAL 24 HOUR
                  AND created_at < {ts:DateTime}) AS amount_24h,
            max(CASE WHEN created_at < {ts:DateTime}
                     THEN created_at ELSE NULL END) AS last_txn_at
        FROM pbp_productionDB_optimized.payments FINAL
        WHERE user_id = {uid:Int64}
          AND _peerdb_is_deleted = 0
          AND payment_method NOT IN ('reversal', 'free')
    """

    BEHAVIOR_SQL = """
        SELECT
            count(DISTINCT facility_id) AS distinct_facilities_30d,
            count(DISTINCT payment_method) AS distinct_methods,
            countIf(status IN ('totally_refunded', 'refunded_to_credit'))
                * 1.0 / greatest(count(), 1) AS reversal_ratio_30d,
            sumIf(discount, 1=1) / greatest(sumIf(reservation_paid_out, 1=1), 0.01)
                AS discount_ratio_30d,
            count() AS txn_count_30d
        FROM pbp_productionDB_optimized.payments FINAL
        WHERE user_id = {uid:Int64}
          AND created_at >= {ts:DateTime} - INTERVAL 30 DAY
          AND created_at < {ts:DateTime}
          AND _peerdb_is_deleted = 0
          AND payment_method NOT IN ('reversal', 'free')
    """

    CREDIT_SQL = """
        SELECT
            countIf(category = 'debit') AS debit_count_30d,
            sumIf(reservation_paid_out, category = 'debit') AS debit_amount_30d,
            sumIf(reservation_paid_out, payment_method = 'prepaid') AS prepaid_spend_30d
        FROM pbp_productionDB_optimized.payments FINAL
        WHERE user_id = {uid:Int64}
          AND created_at >= {ts:DateTime} - INTERVAL 30 DAY
          AND created_at < {ts:DateTime}
          AND _peerdb_is_deleted = 0
    """

    DIVERSITY_SQL = """
        SELECT
            groupArray(category) AS categories_30d,
            countIf(status IN ('totally_refunded', 'refunded_to_credit'))
                AS reversal_count_30d,
            countIf(category = 'merchandise') * 1.0
                / greatest(count(), 1) AS merchandise_ratio_30d
        FROM pbp_productionDB_optimized.payments FINAL
        WHERE user_id = {uid:Int64}
          AND created_at >= {ts:DateTime} - INTERVAL 30 DAY
          AND created_at < {ts:DateTime}
          AND _peerdb_is_deleted = 0
          AND payment_method NOT IN ('reversal', 'free')
    """

    USER_SQL = """
        SELECT created_at AS user_created_at
        FROM pbp_productionDB_optimized.users FINAL
        WHERE id = {uid:Int64} AND _peerdb_is_deleted = 0
        LIMIT 1
    """

    ROLE_SQL = """
        SELECT role
        FROM pbp_productionDB_optimized.facilities_users FINAL
        WHERE user_id = {uid:Int64} AND facility_id = {fid:Int64}
          AND _peerdb_is_deleted = 0
        LIMIT 1
    """

    def __init__(self, ch_connector=None):
        self._ch = ch_connector

    def get_context(
        self, user_id: int, facility_id: int, timestamp: datetime
    ) -> UserContext:
        """Fetch rolling aggregates for a user from ClickHouse.

        Returns UserContext with zeros if ClickHouse is unavailable or user has no history.
        """
        if self._ch is None:
            logger.warning("No ClickHouse connector — returning empty context")
            return UserContext()

        ctx = UserContext()
        params = {"uid": user_id, "fid": facility_id, "ts": str(timestamp)}

        try:
            client = self._ch._client

            # Velocity
            row = client.query(self.VELOCITY_SQL, parameters=params).first_row
            if row:
                ctx.txn_count_1h = int(row[0] or 0)
                ctx.txn_count_24h = int(row[1] or 0)
                ctx.amount_24h = float(row[2] or 0)
                ctx.last_txn_at = row[3]

            # Behavior
            row = client.query(self.BEHAVIOR_SQL, parameters=params).first_row
            if row:
                ctx.distinct_facilities_30d = int(row[0] or 0)
                ctx.distinct_methods = int(row[1] or 0)
                ctx.reversal_ratio_30d = float(row[2] or 0)
                ctx.discount_ratio_30d = float(row[3] or 0)
                ctx.txn_count_30d = int(row[4] or 0)

            # Credit
            row = client.query(self.CREDIT_SQL, parameters=params).first_row
            if row:
                ctx.debit_count_30d = int(row[0] or 0)
                ctx.debit_amount_30d = float(row[1] or 0)
                ctx.prepaid_spend_30d = float(row[2] or 0)

            # Diversity
            row = client.query(self.DIVERSITY_SQL, parameters=params).first_row
            if row:
                ctx.categories_30d = list(row[0] or [])
                ctx.reversal_count_30d = int(row[1] or 0)
                ctx.merchandise_ratio_30d = float(row[2] or 0)

            # User info
            row = client.query(self.USER_SQL, parameters=params).first_row
            if row:
                ctx.user_created_at = row[0]

            # Role
            row = client.query(self.ROLE_SQL, parameters=params).first_row
            if row and row[0]:
                ctx.user_role = str(row[0])

        except Exception as e:
            logger.warning(f"ClickHouse query failed for user {user_id}: {e}")

        return ctx
