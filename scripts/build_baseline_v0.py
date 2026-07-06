"""
Fase 0: Construccion del baseline congelado v0.

Produce:
  - output/golden_set_v0.parquet  : >=500 pagos estratificados por facility_id (seed=42)
  - output/baseline_v0.json       : documento de baseline post-fix con gate de sesgo formal

Gate de exito: reduccion de sesgo (top-5% monto <4x, off-hours local ~4-5%).
AUC vs pure_fraud: diagnostico circular, NO criterio de exito.

Metricas computadas con scores post-fix (scorer corregido en plan 00-01).
No re-entrena el modelo ni modifica thresholds_v2.json.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VAL_PARQUET = os.path.join(BASE_DIR, "data/processed/val_features_enriched.parquet")
VAL_SCORES = os.path.join(BASE_DIR, "output/scores/if_val_scores_final.npy")
GOLDEN_OUT = os.path.join(BASE_DIR, "output/golden_set_v0.parquet")
BASELINE_OUT = os.path.join(BASE_DIR, "output/baseline_v0.json")
THRESHOLD_PATH = os.path.join(BASE_DIR, "output/models/thresholds_v2.json")
FEATURE_LIST_PATH = os.path.join(
    BASE_DIR, "output/models/final_feature_list_operational.json"
)

# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------
N_ROWS = 500
MIN_FACILITIES = 20
SEED = 42

# Proxy Tipo A (reembolso): SOLO para diagnostico, no para entrenamiento
TIPO_A_STATUSES = {"totally_refunded", "refunded_to_credit"}


# ---------------------------------------------------------------------------
# Task 1: Golden set
# ---------------------------------------------------------------------------


def build_golden_set() -> pd.DataFrame:
    """Construye el set dorado estratificado por facility_id con seed fijo.

    Returns >=500 filas con >=20 facilities distintas.
    Reproducible byte-a-byte con SEED=42.
    """
    df = pd.read_parquet(VAL_PARQUET)

    # Estratificado: min(per_fac, filas disponibles) por facility
    per_fac = max(1, N_ROWS // MIN_FACILITIES)
    sampled = df.groupby("facility_id", group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_fac), random_state=SEED)
    )

    # Completar hasta N_ROWS si la estratificacion no llega
    if len(sampled) < N_ROWS:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(N_ROWS - len(sampled), random_state=SEED)
        sampled = pd.concat([sampled, extra])

    sampled = sampled.reset_index(drop=True)

    # Invariantes de calidad
    assert len(sampled) >= N_ROWS, (
        f"Golden set tiene {len(sampled)} filas, se requieren >={N_ROWS}"
    )
    assert sampled["facility_id"].nunique() >= MIN_FACILITIES, (
        f"Golden set cubre {sampled['facility_id'].nunique()} facilities, "
        f"se requieren >={MIN_FACILITIES}"
    )

    # Columnas requeridas para paridad y metricas de sesgo
    required_cols = [
        "facility_id",
        "status",
        "currency",
        "user_role",
        "amount",
        "created_at",
        "facility_avg_amount",
        "amount_facility_ratio",
        "staff_amount_zscore",
        "capture_delay_seconds",
        "is_off_hours",
    ]
    missing = [c for c in required_cols if c not in sampled.columns]
    assert not missing, f"Columnas requeridas ausentes: {missing}"

    sampled.to_parquet(GOLDEN_OUT, index=False)
    print(
        f"[build_golden_set] Escrito: {GOLDEN_OUT}"
        f" | rows={len(sampled)}, facilities={sampled['facility_id'].nunique()}"
    )
    return sampled


# ---------------------------------------------------------------------------
# Task 2: Baseline document
# ---------------------------------------------------------------------------


def _load_threshold_artifact() -> dict:
    with open(THRESHOLD_PATH) as f:
        return json.load(f)


def _load_feature_list() -> dict:
    with open(FEATURE_LIST_PATH) as f:
        return json.load(f)


def _compute_bias_metrics(df: pd.DataFrame, scores: np.ndarray) -> dict:
    """Computa metricas de sesgo (punto de partida, no valores buenos)."""
    # Top-5% por score mas alto (mas anomalo)
    top5_cutoff = np.percentile(scores, 95)
    top5_mask = scores >= top5_cutoff

    top5_amount_mean = float(df.loc[top5_mask, "amount"].mean())
    global_amount_mean = float(df["amount"].mean())

    top5pct_amount_ratio = top5_amount_mean / global_amount_mean if global_amount_mean > 0 else None

    # Off-hours rate en UTC (is_off_hours ya calculado en feature engineering)
    off_hours_utc_pct = float(df["is_off_hours"].mean() * 100)
    top5_off_hours_pct = float(df.loc[top5_mask, "is_off_hours"].mean() * 100)

    return {
        "top5pct_amount_ratio": round(top5pct_amount_ratio, 4) if top5pct_amount_ratio else None,
        "top5pct_amount_mean_usd": round(top5_amount_mean, 2),
        "global_amount_mean_usd": round(global_amount_mean, 2),
        "top5pct_cutoff_score": round(float(top5_cutoff), 10),
        "top5pct_count": int(top5_mask.sum()),
        "off_hours_rate_utc_pct": round(off_hours_utc_pct, 4),
        "top5pct_off_hours_rate_utc_pct": round(top5_off_hours_pct, 4),
        "off_hours_note": (
            "UTC-based; objetivo local ~4-5% requiere offset de timezone por facility "
            "(Fase 1). El ~30% UTC es el punto de partida a reducir."
        ),
        "top5pct_note": (
            "Ratio actual >4x es el punto de partida; objetivo Fase 1: reducir a <4x "
            "con calibracion por facility. No es un valor aceptable, es la linea base."
        ),
    }


def _compute_data_quality_metrics(df: pd.DataFrame) -> dict:
    """Conteos EMPTY de 00-02 mas captura de capture_delay zero pct."""
    # capture_delay_seconds: zero pct en val
    capture_delay_zero_pct = float((df["capture_delay_seconds"] == 0).mean() * 100)

    return {
        "currency_empty_train": 0,
        "currency_empty_val": 0,
        "currency_empty_test": "not_verified_preventive",
        "currency_empty_note": (
            "0 filas con EMPTY en splits actuales (confirmado en plan 00-02). "
            "Fix aplicado preventivamente en loader.py y engineering.py "
            "para futuras extracciones ClickHouse."
        ),
        "capture_delay_seconds_zero_pct": round(capture_delay_zero_pct, 4),
        "capture_delay_note": "excluded_from_FS-frame-operational-v1 (train/serve skew)",
    }


def _compute_tipo_a_metrics(df: pd.DataFrame) -> dict:
    tipo_a_mask = df["status"].isin(TIPO_A_STATUSES)
    return {
        "val_set_size": int(len(df)),
        "val_set_tipo_a_count": int(tipo_a_mask.sum()),
        "tipo_a_rate_pct": round(float(tipo_a_mask.mean() * 100), 6),
        "tipo_a_proxy_note": (
            "Proxy Tipo A = reembolso (totally_refunded | refunded_to_credit). "
            "SOLO para diagnostico; el modelo NO entrena con etiquetas. "
            "Proxy != fraude: anomalia de reembolso es un indicador indirecto, "
            "no causalidad."
        ),
    }


def build_baseline_document() -> dict:
    """Computa metricas POST-fix y escribe baseline_v0.json."""
    df = pd.read_parquet(VAL_PARQUET)
    scores = np.load(VAL_SCORES)

    assert len(scores) == len(df), (
        f"Desalineamiento: scores tiene {len(scores)} filas, df tiene {len(df)}"
    )

    # Cargar artefactos de referencia
    threshold_artifact = _load_threshold_artifact()
    feature_list_artifact = _load_feature_list()

    # Calcular metricas
    tipo_a_metrics = _compute_tipo_a_metrics(df)
    bias_metrics = _compute_bias_metrics(df, scores)
    data_quality = _compute_data_quality_metrics(df)

    # Golden set stats (si ya fue generado)
    golden_rows = None
    golden_facilities = None
    if os.path.exists(GOLDEN_OUT):
        golden_df = pd.read_parquet(GOLDEN_OUT)
        golden_rows = int(len(golden_df))
        golden_facilities = int(golden_df["facility_id"].nunique())

    baseline = {
        "baseline_version": "v0",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scorer_model": "IF-40-v1",
        "feature_set": feature_list_artifact.get("feature_set_id", "FS-frame-operational-v1"),
        "feature_list_source": "output/models/final_feature_list_operational.json",
        "feature_count": len(feature_list_artifact.get("features", [])),
        "threshold_artifact": "thresholds_v2.json",
        "threshold_source": threshold_artifact.get("threshold_source", "percentile_95_validation_set"),
        "threshold_value": threshold_artifact.get("binary_threshold", 0.024223975402714343),
        #
        # Gate de exito: reduccion de sesgo, NUNCA AUC
        #
        "gate_metric": "bias_reduction",
        "gate_criteria": {
            "top5pct_amount_ratio_target": "<4x",
            "off_hours_rate_local_target": "~4-5%",
            "notes": (
                "AUC vs pure_fraud es diagnostico circular — NO usar como gate. "
                "4 features del modelo definen el proxy pure_fraud (user_reversal_ratio_30d, "
                "user_discount_ratio_30d, user_reversal_count_30d, user_merchandise_ratio_30d). "
                "El AUC alto (~0.84) refleja parcialmente esa circularidad, no discriminacion genuina."
            ),
        },
        #
        # Metricas actuales POST-fix (scorer corregido en 00-01)
        # Estos son los valores de partida a mejorar en Fase 1, NO valores aceptables.
        #
        "current_metrics_post_fix": {
            **tipo_a_metrics,
            **bias_metrics,
            "auc_tipo_a_val_diagnostic": {
                "value": 0.4926,
                "note": (
                    "AUC vs Tipo A en val (honest, FS-operational-v1, post-fix). "
                    "Valor <0.5 refleja que las anomalias del IF no se alinean con reembolsos. "
                    "DIAGNOSTICO — no criterio de exito."
                ),
            },
            "auc_pure_fraud_val_diagnostic": {
                "value": 0.8364,
                "classification": "diagnostic_circular_not_a_gate_metric",
                "note": (
                    "AUC vs pure_fraud en val. Circularidad: 4 features del modelo "
                    "derivadas del comportamiento de reembolso definen el proxy pure_fraud. "
                    "El AUC alto es parcialmente autovalidacion. "
                    "DIAGNOSTICO — NO es criterio de exito ni de progreso."
                ),
            },
        },
        #
        # Reporte de calidad de datos (de plan 00-02)
        #
        "data_quality_report": data_quality,
        #
        # Set dorado
        #
        "golden_set": {
            "source": "val_features_enriched.parquet",
            "n_rows": golden_rows,
            "stratified_by": "facility_id",
            "per_facility_sample": max(1, N_ROWS // MIN_FACILITIES),
            "seed": SEED,
            "facilities_covered": golden_facilities,
            "path": "output/golden_set_v0.parquet",
            "note": "Reproducible byte-a-byte con seed=42. Incluye columnas para paridad y metricas de sesgo.",
        },
        #
        # Resultado del test de paridad batch<->real-time (de plan 00-01)
        #
        "parity_test_result": {
            "facility_avg_amount_max_delta": 0.0,
            "staff_amount_zscore_max_delta": 0.0,
            "n_transactions_tested": 14831,
            "n_facilities_covered": 680,
            "tolerance": 1e-6,
            "status": "PASS",
            "source": "tests/test_parity_phase0.py (plan 00-01)",
            "note": (
                "Delta exactamente 0.0 para ambas features porque los stats almacenados "
                "son float32 y el path de calculo es identico post-fix."
            ),
        },
        #
        # Bugs corregidos (planes 00-01 y 00-02)
        #
        "bugs_fixed": [
            {
                "id": "BUG-01",
                "file": "src/fraud_detector/scoring/features.py:27",
                "description": "getattr(fe._groups[4], '_facility_avg', {}) -> acceso directo _facility_avg_amount",
                "impact": "facility_avg_amount = global mean para TODAS las facilities en scorer RT (689 facilities afectadas)",
                "plan": "00-01",
                "commit": "480e5af",
            },
            {
                "id": "BUG-02",
                "file": "src/fraud_detector/scoring/features.py:28",
                "description": "getattr(fe._groups[6], '_staff_stats', {}) -> acceso directo _role_currency_stats (clave compuesta tuple)",
                "impact": "staff_amount_zscore = fallback global para TODOS los roles en scorer RT (81 combinaciones afectadas)",
                "plan": "00-01",
                "commit": "480e5af",
            },
            {
                "id": "BUG-03",
                "file": "src/fraud_detector/scoring/features.py:role_key",
                "description": "role_key forzaba a 'player' para roles no-staff; corregido a actual_role sin remapeo",
                "impact": "z-scores incorrectos para ~14% de filas (guest, rental_user, etc.)",
                "plan": "00-01",
                "commit": "3b745da",
            },
            {
                "id": "BUG-04",
                "file": "src/fraud_detector/scoring/features_enriched.py:_capture_delay_seconds",
                "description": "is pd.NaT vulnerable a strings malformados -> pd.isnull() + try/except",
                "impact": "ValueError silencioso para timestamps no parseables",
                "plan": "00-01",
                "commit": "ab8b1d6",
            },
            {
                "id": "BUG-05",
                "file": "src/fraud_detector/data/loader.py + features/engineering.py",
                "description": "currency EMPTY/'' no sanitizada -> reemplazada por USD con warning",
                "impact": "Preventivo: 0 filas afectadas en splits actuales; protege futuras extracciones ClickHouse",
                "plan": "00-02",
                "commit": "70d59e7",
            },
        ],
        #
        # Features excluidas del conjunto operativo
        #
        "excluded_features": feature_list_artifact.get("excluded", ["capture_delay_seconds"]),
        "excluded_features_note": {
            "capture_delay_seconds": (
                "Train/serve skew: valor real en batch (min=-86400s, max=86400s, mean=-52847s, zero_pct=6.2%), "
                "~0 en real-time. AUC flag 0.511 (no informativa para discriminacion en RT). "
                "Excluida de FS-frame-operational-v1."
            ),
        },
        #
        # Clasificacion de AUC pure_fraud (requerida por metodologia)
        #
        "auc_pure_fraud_classification": "diagnostic_circular_not_a_gate_metric",
        "methodology_notes": {
            "approach": "no_supervisado",
            "language_constraint": (
                "Sin lenguaje causal. Las anomalias muestran asociacion con patrones de reembolso, "
                "no causalidad. Usar 'asociacion', 'capacidad discriminativa', nunca 'predice'."
            ),
            "proxy_note": (
                "El proxy Tipo A (reembolso) es SOLO para diagnostico/evaluacion. "
                "El modelo entrena sin etiquetas. Anomalia != fraude."
            ),
        },
        #
        # Trazabilidad de artefactos
        #
        "artifact_links": {
            "feature_list": "output/models/final_feature_list_operational.json",
            "threshold": "output/models/thresholds_v2.json",
            "golden_set": "output/golden_set_v0.parquet",
            "parity_tests": "tests/test_parity_phase0.py",
            "val_scores": "output/scores/if_val_scores_final.npy",
        },
    }

    with open(BASELINE_OUT, "w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)

    print(f"[build_baseline_document] Escrito: {BASELINE_OUT}")
    return baseline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)

    print("=== Task 1: Construir set dorado ===")
    golden = build_golden_set()
    print(
        f"  rows={len(golden)}, facilities={golden['facility_id'].nunique()}, "
        f"seed={SEED}"
    )

    print()
    print("=== Task 2: Materializar baseline_v0.json ===")
    baseline = build_baseline_document()
    print(f"  gate_metric: {baseline['gate_metric']}")
    print(f"  feature_set: {baseline['feature_set']}")
    print(f"  threshold:   {baseline['threshold_value']}")
    top5_ratio = baseline["current_metrics_post_fix"].get("top5pct_amount_ratio")
    off_hours = baseline["current_metrics_post_fix"].get("off_hours_rate_utc_pct")
    print(f"  top5pct_amount_ratio (baseline): {top5_ratio}x  (objetivo Fase 1: <4x)")
    print(f"  off_hours_rate_utc (baseline): {off_hours}%  (objetivo Fase 1: ~4-5% local)")
    print(f"  auc_pure_fraud_classification: {baseline['auc_pure_fraud_classification']}")
    print()
    print("OK: baseline v0 congelado.")


if __name__ == "__main__":
    main()
