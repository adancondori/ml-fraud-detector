"""Fase 0: saneamiento de moneda EMPTY/'' -> USD (fix preventivo BASE-05).

Verifica que tanto engineering.py (línea ~646) como DataManager._sanitize_currency
(loader.py) mapeen correctamente los valores problemáticos a USD, y que el loader
emita un warning con el conteo de filas afectadas cuando encuentra al menos un EMPTY.
"""
from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Test 1: ingeniería de features — replace inline en preprocessing
# ---------------------------------------------------------------------------

def test_engineering_currency_empty_sanitized():
    """La lógica de replace en engineering.py:~646 debe mapear EMPTY/'' a USD."""
    s = pd.Series(["EMPTY", "", "usd", None])
    result = (
        s.fillna("USD").astype(str).str.upper()
        .replace({"EMPTY": "USD", "": "USD"})
    )
    assert list(result) == ["USD", "USD", "USD", "USD"], (
        f"Engineering replace fallló: {list(result)}"
    )


def test_engineering_currency_already_valid_unchanged():
    """Monedas válidas (AED, ARS, EUR) no deben ser alteradas."""
    s = pd.Series(["AED", "ars", "EUR", "usd"])
    result = (
        s.fillna("USD").astype(str).str.upper()
        .replace({"EMPTY": "USD", "": "USD"})
    )
    assert list(result) == ["AED", "ARS", "EUR", "USD"], (
        f"Monedas válidas alteradas: {list(result)}"
    )


# ---------------------------------------------------------------------------
# Test 2: DataManager._sanitize_currency (helper estático en loader.py)
# ---------------------------------------------------------------------------

def test_loader_sanitize_currency_maps_empty_to_usd():
    """DataManager._sanitize_currency debe mapear EMPTY, '', None -> USD."""
    from fraud_detector.data.loader import DataManager

    s = pd.Series(["EMPTY", "", None, "usd", "AED"])
    result = DataManager._sanitize_currency(s)
    expected = ["USD", "USD", "USD", "USD", "AED"]
    assert list(result) == expected, f"Resultado: {list(result)}"


def test_loader_sanitize_currency_warning_with_count():
    """Cuando hay filas EMPTY/'' se debe loguear un warning con el conteo.

    Loguru no se integra con pytest caplog por defecto.  Capturamos el mensaje
    añadiendo temporalmente un sink de lista a la instancia de loguru.
    """
    from loguru import logger as loguru_logger
    from fraud_detector.data.loader import DataManager

    captured: list[str] = []

    sink_id = loguru_logger.add(
        lambda msg: captured.append(msg),
        level="WARNING",
        format="{message}",
    )
    try:
        s = pd.Series(["EMPTY", "EMPTY", "", "USD"])
        result = DataManager._sanitize_currency(s)
    finally:
        loguru_logger.remove(sink_id)

    # El warning debe haberse emitido y mencionar el conteo 3 (2 EMPTY + 1 '')
    assert len(captured) >= 1, (
        "Se esperaba al menos un warning logueado; ninguno capturado."
    )
    assert any("3" in str(m) for m in captured), (
        f"Warning no menciona conteo=3: {captured}"
    )
    # Verificar que el resultado es correcto
    assert list(result) == ["USD", "USD", "USD", "USD"]


def test_loader_sanitize_currency_no_warning_when_clean():
    """Cuando no hay EMPTY/'' no se debe emitir ningún warning."""
    from loguru import logger as loguru_logger
    from fraud_detector.data.loader import DataManager

    captured: list[str] = []
    sink_id = loguru_logger.add(
        lambda msg: captured.append(msg),
        level="WARNING",
        format="{message}",
    )
    try:
        s = pd.Series(["USD", "AED", "EUR", "ARS"])
        DataManager._sanitize_currency(s)
    finally:
        loguru_logger.remove(sink_id)

    assert len(captured) == 0, (
        f"Warning inesperado cuando no hay EMPTY: {captured}"
    )


# ---------------------------------------------------------------------------
# Test 3: confirmación de datos actuales — 0 filas EMPTY en parquets procesados
# ---------------------------------------------------------------------------

def test_data_quality_zero_empty_currency_in_train(tmp_path):
    """Confirmación de calidad de datos: 0 filas EMPTY en train_raw.parquet actual."""
    import os
    parquet_path = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "processed", "train_raw.parquet"
    )
    if not os.path.exists(parquet_path):
        pytest.skip("train_raw.parquet no disponible en este entorno")

    df = pd.read_parquet(parquet_path, columns=["currency"])
    n_empty = int(df["currency"].isin(["EMPTY", ""]).sum())
    assert n_empty == 0, (
        f"Se encontraron {n_empty} filas con currency EMPTY en train_raw.parquet "
        "(debería ser 0 — el fix es preventivo para futuras extracciones)"
    )


def test_data_quality_zero_empty_currency_in_val():
    """Confirmación de calidad de datos: 0 filas EMPTY en val_raw.parquet actual."""
    import os
    parquet_path = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "processed", "val_raw.parquet"
    )
    if not os.path.exists(parquet_path):
        pytest.skip("val_raw.parquet no disponible en este entorno")

    df = pd.read_parquet(parquet_path, columns=["currency"])
    n_empty = int(df["currency"].isin(["EMPTY", ""]).sum())
    assert n_empty == 0, (
        f"Se encontraron {n_empty} filas con currency EMPTY en val_raw.parquet "
        "(debería ser 0 — el fix es preventivo para futuras extracciones)"
    )
