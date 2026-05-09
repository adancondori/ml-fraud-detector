"""Pydantic request/response models for the entire ML Scorer API."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    """Payload for scoring a single transaction."""

    user_id: int
    facility_id: int
    reservation_paid_out: float
    created_at: datetime
    discount: float = 0.0
    tip: float = 0.0
    payment_method: str = "card"
    category: str = "reservation"
    club_credit_flag: bool = False
    paid_by_manager: bool = False
    currency: str = "USD"


class FactorItem(BaseModel):
    """Single feature explanation entry."""

    feature: str
    value: float
    z_score: float
    direction: str


class ScoreResponse(BaseModel):
    """Result of scoring a single transaction."""

    raw_score: float
    percentile: float
    risk_level: str
    is_anomaly: bool
    factors: List[FactorItem]


class BatchScoreRequest(BaseModel):
    """Payload for triggering a batch scoring run."""

    cursor: datetime = Field(..., description="ISO8601 timestamp — process records after this point")


class CriticalAlert(BaseModel):
    """One critical-risk transaction found during batch scoring."""

    payment_id: int
    user_id: int
    facility_id: int
    raw_score: float
    risk_level: str
    amount_usd: float


class BatchScoreResponse(BaseModel):
    """Summary of a completed batch scoring run."""

    processed: int
    scored: int
    critical_alerts: List[CriticalAlert]
    next_cursor: Optional[datetime] = None


class HealthResponse(BaseModel):
    """Application health status."""

    model_loaded: bool
    clickhouse_connected: bool
    model_version: str
    last_batch_at: Optional[datetime] = None


class ModelInfoResponse(BaseModel):
    """Metadata about the loaded model."""

    model_version: str
    feature_count: int
    threshold: float
    risk_levels: Dict[str, list]


class ReloadResponse(BaseModel):
    """Result of a model reload operation."""

    status: str
    model_version: str
