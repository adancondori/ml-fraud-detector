"""FastAPI dependency functions for scorer and ClickHouse client.

_state is written by the lifespan in main.py and read here via DI.
"""
from __future__ import annotations

from fastapi import HTTPException

from fraud_detector.scoring.scorer import SingleTransactionScorer

# Shared state dict — lifespan in main.py writes to this dict;
# dependency functions read from it.  Both modules import the SAME object.
_state: dict = {}


def get_scorer() -> SingleTransactionScorer:
    """Return the loaded SingleTransactionScorer or raise 503."""
    scorer = _state.get("scorer")
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return scorer


def get_ch_client():
    """Return the ClickHouse client or raise 503."""
    client = _state.get("ch_client")
    if client is None:
        raise HTTPException(status_code=503, detail="ClickHouse not connected")
    return client
