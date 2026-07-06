"""Offline retraining: IsolationForest sobre FS-frame-v1 (30 features).

Receta congelada:
  IsolationForest(n_estimators=200, max_samples=512, max_features=0.6,
                  contamination="auto", random_state=42, n_jobs=-1)
  RobustScaler(quantile_range=(5.0, 95.0))
  post-transform clip [-10, 10]

Artefactos producidos (NUNCA toca el modelo de producción):
  output/models/isolation_forest_frame_v1.joblib
  output/models/scaler_frame_v1.joblib
  output/models/model_metadata_frame_v1.json
  output/frame_v1_bias_report.json

Gate de sesgo (val set completo, ~1.13M filas):
  Gate 1 — top-5% amount ratio:  < 4.0x  (baseline: 11.79x)
  Gate 2 — off-hours local pct:  ~4-5%   (baseline UTC: 29.78%)

Paridad garantizada: add_frame_features_from_artifact (vectorizado) replica
_compute_frame_features de features_frame_v1.py línea por línea (diff <1e-8).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "output" / "models"

# Añadir src al path para importar desde el proyecto
sys.path.insert(0, str(ROOT / "src"))

from fraud_detector.scoring.features_frame_v1 import (  # noqa: E402
    FRAME_V1_FEATURE_NAMES,
    FrameV1FeatureCalculator,
)

# ---------------------------------------------------------------------------
# Constantes — iguales a features_frame_v1.py (fuente canónica)
# ---------------------------------------------------------------------------

OFF_HOURS: frozenset = frozenset({23, 0, 1, 2, 3, 4, 5, 6})

# Rutas de artefactos
STATS_PATH = MODELS / "facility_stats_v1.json"
FE_PATH = MODELS / "feature_engineer.joblib"
TZ_PARQUET = ROOT / "output" / "revision" / "facility_tz.parquet"

TRAIN_PARQUET = DATA / "train_features_enriched.parquet"
VAL_PARQUET = DATA / "val_features_enriched.parquet"

# Columnas necesarias del parquet (evitar cargar todo)
NEEDED_COLS = [
    "facility_id", "amount", "created_at", "currency",
    "discount_ratio", "has_tip", "time_since_last_txn",
    "user_amount_24h", "user_distinct_facilities_30d", "user_distinct_methods",
    "user_debit_count_30d", "user_debit_amount_30d", "credit_flow_ratio",
    "is_club_credit", "paid_by_manager", "user_role",
    "category_entropy_30d", "user_merchandise_ratio_30d",
    "gateway_change_recent", "is_main_gateway",
    "is_first_gateway_for_user", "source_change_recent",
    "is_off_hours",  # columna UTC pre-computada — para contraste en Gate 2
]


# ---------------------------------------------------------------------------
# Función vectorizada: add_frame_features_from_artifact
# Aritmética idéntica línea por línea a FrameV1FeatureCalculator._compute_frame_features
# ---------------------------------------------------------------------------

def _build_facility_lookup(stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Pre-computar arrays de lookup para todas las facilities conocidas.

    Retorna:
        fid_to_mean:    dict[int -> float]
        fid_to_iqrg:    dict[int -> float]
        fid_to_iana_tz: dict[int -> str]
        global_fallback: dict con keys mean, median, iqr_guarded
    """
    gfb = stats["global_fallback"]
    g_mean = float(gfb["mean"])
    g_median = float(gfb["median"])
    g_iqrg = float(gfb["iqr_guarded"])

    fid_to_mean: dict = {}
    fid_to_median: dict = {}
    fid_to_iqrg: dict = {}
    fid_to_iana_tz: dict = {}

    for fid_str, entry in stats["facilities"].items():
        fid = int(fid_str)
        iana_tz = entry.get("iana_tz", "Etc/UTC") or "Etc/UTC"
        fmean = float(entry.get("mean") or 0)
        fmedian = float(entry.get("median") or 0)
        fiqrg = float(entry.get("iqr_guarded") or 1.0)

        fid_to_iana_tz[fid] = iana_tz

        if fmean > 0:
            # Cubre facility y currency-fallback (ambos tienen mean>0)
            fid_to_mean[fid] = fmean
            fid_to_median[fid] = fmedian
            fid_to_iqrg[fid] = fiqrg
        else:
            # Facility con mean==0 → usar global_fallback para mean/median/iqr_guarded
            # (iana_tz se mantiene del artefacto para la facility conocida)
            fid_to_mean[fid] = g_mean
            fid_to_median[fid] = g_median
            fid_to_iqrg[fid] = g_iqrg

    return fid_to_mean, fid_to_median, fid_to_iqrg, fid_to_iana_tz, {
        "mean": g_mean, "median": g_median, "iqr_guarded": g_iqrg
    }


def add_frame_features_from_artifact(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Añade las 30 columnas de FRAME_V1_FEATURE_NAMES al DataFrame (vectorizado).

    Aritmética IDÉNTICA a FrameV1FeatureCalculator._compute_frame_features:
      - Lookup facility: facility → currency → global (según mean>0 en artefacto)
      - iqr_guarded del artefacto (NO iqr+1e-6 del prototipo)
      - log_amount_fac = log1p(amount / (fmean + 0.01))
      - amount_facility_ratio = amount / (fmean + 0.01)
      - user_amount_24h_fac = user_amount_24h / (fmean + 0.01)
      - user_debit_amount_30d_fac = user_debit_amount_30d / (fmean + 0.01)
      - Hora local: tz_localize("UTC").astimezone(ZoneInfo(iana_tz)) — idéntico a _local_hour_dow
      - OFF_HOURS = {23,0,1,2,3,4,5,6}, dow 0=lunes
      - off_hours_high_value_loc: is_off_hours_loc > 0 AND amount_facility_ratio > 3
      - staff_amount_zscore: usa currency del parquet (= currency_original en calculate_from_row)

    Args:
        df: DataFrame con columnas de NEEDED_COLS.
        stats: dict cargado de facility_stats_v1.json.

    Returns:
        Copia de df con las 30 columnas de FRAME_V1_FEATURE_NAMES añadidas.
    """
    df = df.copy()
    fid_arr = df["facility_id"].to_numpy(dtype=np.int32)
    amt_arr = df["amount"].to_numpy(dtype=np.float64)
    n = len(df)

    # --- Lookup facility stats (vectorizado con fallback chain) ---
    fid_to_mean, fid_to_median, fid_to_iqrg, fid_to_iana_tz, gfb = _build_facility_lookup(stats)
    g_mean = gfb["mean"]
    g_median = gfb["median"]
    g_iqrg = gfb["iqr_guarded"]

    fmean_arr = np.array([fid_to_mean.get(int(f), g_mean) for f in fid_arr], dtype=np.float64)
    fiqrg_arr = np.array([fid_to_iqrg.get(int(f), g_iqrg) for f in fid_arr], dtype=np.float64)

    # --- 2. Magnitud relativa a facility (aritmética de _compute_frame_features líneas 213-216) ---
    log_amount_fac = np.log1p(amt_arr / (fmean_arr + 0.01))
    amount_facility_ratio = amt_arr / (fmean_arr + 0.01)
    user_amount_24h_arr = df["user_amount_24h"].to_numpy(dtype=np.float64)
    user_amount_24h_fac = user_amount_24h_arr / (fmean_arr + 0.01)
    user_debit_amount_30d_arr = df["user_debit_amount_30d"].to_numpy(dtype=np.float64)
    user_debit_amount_30d_fac = user_debit_amount_30d_arr / (fmean_arr + 0.01)

    # --- 3. Temporales en hora local con DST ---
    # Idéntico a _local_hour_dow: tz_localize("UTC").astimezone(ZoneInfo(iana_tz))
    # Vectorizado por grupo de timezone para eficiencia
    created_at = pd.to_datetime(df["created_at"])
    # Asegurar que el timestamp sea naive (UTC naive) antes de localizar
    if created_at.dt.tz is not None:
        created_at = created_at.dt.tz_convert("UTC").dt.tz_localize(None)

    hour_arr = np.zeros(n, dtype=np.int32)
    dow_arr = np.zeros(n, dtype=np.int32)
    iana_tz_arr = np.array([fid_to_iana_tz.get(int(f), "Etc/UTC") for f in fid_arr])

    # Procesar por grupo de timezone (el tz_localize+astimezone es caro por fila)
    # Usar dt.tz_localize("UTC").dt.tz_convert(iana_tz) en pandas (equivalente a astimezone)
    for tz_name in np.unique(iana_tz_arr):
        mask = iana_tz_arr == tz_name
        ts_slice = created_at[mask]
        # Replicar: ts_utc_naive.tz_localize("UTC").astimezone(ZoneInfo(iana_tz))
        ts_local = ts_slice.dt.tz_localize("UTC").dt.tz_convert(tz_name)
        hour_arr[mask] = ts_local.dt.hour.to_numpy(dtype=np.int32)
        dow_arr[mask] = ts_local.dt.dayofweek.to_numpy(dtype=np.int32)  # 0=lunes

    hour_sin_loc = np.sin(2 * np.pi * hour_arr / 24)
    hour_cos_loc = np.cos(2 * np.pi * hour_arr / 24)
    dow_sin_loc = np.sin(2 * np.pi * dow_arr / 7)
    dow_cos_loc = np.cos(2 * np.pi * dow_arr / 7)
    is_weekend_loc = (dow_arr >= 5).astype(np.float32)
    is_off_hours_loc = np.isin(hour_arr, list(OFF_HOURS)).astype(np.float32)

    # --- 4. Interacciones derivadas (líneas 228-233 de _compute_frame_features) ---
    small_amount_at_facility = (amount_facility_ratio < 0.2).astype(np.float32)
    very_small_amount_at_facility = (amount_facility_ratio < 0.05).astype(np.float32)
    off_hours_high_value_loc = (
        (is_off_hours_loc > 0) & (amount_facility_ratio > 3)
    ).astype(np.float32)

    # --- 5. staff_amount_zscore (idéntico a _lookup_staff_zscore) ---
    # El parquet almacena currency = currency_original (igual que calculate_from_row)
    # Por eso usamos currency directamente como currency_original
    # staff zscore se computa fila a fila desde el feature_engineer.joblib
    # Para eficiencia, vectorizamos por (role, currency) como grupo
    fe = joblib.load(FE_PATH)
    staff_role_currency = fe._groups[6]._role_currency_stats
    staff_currency = fe._groups[6]._currency_stats
    g_staff_mean = float(fe._groups[6]._global_mean)
    g_staff_std = float(fe._groups[6]._global_std or 1.0)

    roles = df["user_role"].fillna("player").to_numpy(dtype=str)
    currencies = df["currency"].fillna("USD").str.upper().to_numpy(dtype=str)
    staff_zscore_arr = np.zeros(n, dtype=np.float64)

    for i in range(n):
        role = roles[i] if roles[i] else "player"
        currency = currencies[i]
        key_rc = (role, currency)
        if key_rc in staff_role_currency:
            s = staff_role_currency[key_rc]
        elif currency in staff_currency:
            s = staff_currency[currency]
        else:
            s = {"mean": g_staff_mean, "std": g_staff_std}
        s_mean = float(s["mean"])
        s_std = float(s.get("std") or 1.0) or 1.0
        staff_zscore_arr[i] = (float(amt_arr[i]) - s_mean) / s_std

    # --- 6. is_staff (línea 238 de _compute_frame_features) ---
    is_staff = np.array(
        [float(r in ("court_manager", "court_operator", "teacher")) for r in roles],
        dtype=np.float32
    )

    # --- 7. Ensamblar las 30 columnas en orden FRAME_V1_FEATURE_NAMES ---
    df["log_amount_fac"] = log_amount_fac.astype(np.float32)
    # discount_ratio y has_tip ya existen en el parquet — usarlos tal cual
    df["hour_sin_loc"] = hour_sin_loc.astype(np.float32)
    df["hour_cos_loc"] = hour_cos_loc.astype(np.float32)
    df["dow_sin_loc"] = dow_sin_loc.astype(np.float32)
    df["dow_cos_loc"] = dow_cos_loc.astype(np.float32)
    df["is_weekend_loc"] = is_weekend_loc
    df["is_off_hours_loc"] = is_off_hours_loc
    df["user_amount_24h_fac"] = user_amount_24h_fac.astype(np.float32)
    df["amount_facility_ratio"] = amount_facility_ratio.astype(np.float32)
    df["user_debit_amount_30d_fac"] = user_debit_amount_30d_fac.astype(np.float32)
    df["is_staff"] = is_staff
    df["staff_amount_zscore"] = staff_zscore_arr.astype(np.float32)
    df["small_amount_at_facility"] = small_amount_at_facility
    df["very_small_amount_at_facility"] = very_small_amount_at_facility
    df["off_hours_high_value_loc"] = off_hours_high_value_loc

    assert set(FRAME_V1_FEATURE_NAMES).issubset(set(df.columns)), (
        f"Faltan columnas de FRAME_V1_FEATURE_NAMES en el df: "
        f"{set(FRAME_V1_FEATURE_NAMES) - set(df.columns)}"
    )
    return df


# ---------------------------------------------------------------------------
# Verificación de paridad batch↔calculator
# ---------------------------------------------------------------------------

def check_batch_calculator_parity(
    sample_df: pd.DataFrame,
    stats: dict,
    fe_path: str = str(FE_PATH),
    tol: float = 1e-8,
    verbose: bool = True,
) -> float:
    """Verifica que add_frame_features_from_artifact ≡ FrameV1FeatureCalculator._compute_frame_features.

    Args:
        sample_df: DataFrame con ≥100 filas y ≥20 facilities del val set.
        stats: dict cargado de facility_stats_v1.json.
        fe_path: ruta a feature_engineer.joblib.
        tol: tolerancia máxima (diff < tol para PASS).
        verbose: imprimir resultado.

    Returns:
        max_diff: float — diff máximo observado sobre todas las filas y features.

    Raises:
        AssertionError: si max_diff >= tol (paridad fallida).
    """
    calculator = FrameV1FeatureCalculator(
        facility_stats=stats,
        feature_engineer_path=fe_path,
    )

    # Aplicar batch vectorizado
    sample_enriched = add_frame_features_from_artifact(sample_df, stats)

    max_diffs = []
    for idx in range(len(sample_df)):
        row = sample_df.iloc[idx]
        # Vector del calculator (misma lógica que calculate_from_row)
        calc_vec = calculator.calculate_from_row(row).astype(np.float64)

        # Vector del batch
        batch_vec = np.array(
            [sample_enriched.iloc[idx][feat] for feat in FRAME_V1_FEATURE_NAMES],
            dtype=np.float64
        )

        diff = np.max(np.abs(calc_vec - batch_vec))
        max_diffs.append(diff)

        if diff >= tol:
            # Imprimir diagnóstico detallado
            mismatches = np.where(np.abs(calc_vec - batch_vec) >= tol)[0]
            for midx in mismatches:
                print(f"  MISMATCH feature={FRAME_V1_FEATURE_NAMES[midx]} "
                      f"calc={calc_vec[midx]:.10e} batch={batch_vec[midx]:.10e} "
                      f"diff={abs(calc_vec[midx] - batch_vec[midx]):.2e}")

    max_diff = max(max_diffs) if max_diffs else 0.0
    n_facilities = sample_df["facility_id"].nunique()

    if verbose:
        status = "PASS" if max_diff < tol else "FAIL"
        print(f"Paridad batch↔calculator: {status} | "
              f"max_diff={max_diff:.2e} (tol={tol:.0e}) | "
              f"{len(sample_df)} filas, {n_facilities} facilities")

    assert max_diff < tol, (
        f"Paridad batch↔calculator FALLIDA: max_diff={max_diff:.2e} >= tol={tol:.0e} "
        f"sobre {len(sample_df)} filas, {n_facilities} facilities. "
        f"La aritmética batch diverge del calculator de 01-02. Abortar reentrenamiento."
    )
    return max_diff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()
    print("=" * 60)
    print("retrain_frame_v1.py — Reentrenamiento offline FS-frame-v1")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Cargar artefactos base
    # ------------------------------------------------------------------
    print("\n[1/5] Cargando artefactos...")
    t0 = time.perf_counter()
    with open(STATS_PATH) as f:
        stats = json.load(f)
    print(f"  facility_stats_v1.json: {len(stats['facilities'])} facilities "
          f"({time.perf_counter()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Cargar train set y añadir frame features
    # ------------------------------------------------------------------
    print("\n[2/5] Cargando train set y añadiendo frame features...")
    t0 = time.perf_counter()
    train_df = pd.read_parquet(TRAIN_PARQUET, columns=NEEDED_COLS)
    print(f"  train shape: {train_df.shape} ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    train_df = add_frame_features_from_artifact(train_df, stats)
    print(f"  frame features añadidas a train ({time.perf_counter()-t0:.1f}s)")

    X_train = train_df[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  X_train shape: {X_train.shape}")

    # ------------------------------------------------------------------
    # VERIFICACIÓN DE PARIDAD batch↔calculator
    # Antes de entrenar: verificar que add_frame_features_from_artifact
    # produce features idénticas a FrameV1FeatureCalculator.calculate_from_row
    # sobre ≥100 filas del val set. ABORTA si diff ≥ 1e-8.
    # ------------------------------------------------------------------
    print("\n[3/5] Verificación de paridad batch↔calculator (≥100 filas, ≥20 facilities)...")
    t0 = time.perf_counter()

    val_sample_full = pd.read_parquet(VAL_PARQUET, columns=NEEDED_COLS)
    # Estratificar: ≥20 facilities, ≥100 filas (mismo patrón que golden_rows en test_parity_phase1.py)
    rows_per_fac = max(1, 100 // 20)
    sampled = (
        val_sample_full.groupby("facility_id", group_keys=False)
        .apply(lambda g: g.head(rows_per_fac), include_groups=False)
    )
    if "facility_id" not in sampled.columns:
        sampled = sampled.join(val_sample_full[["facility_id"]])
    if len(sampled) < 100:
        sampled = val_sample_full.head(100)
    sampled = sampled.head(max(100, len(sampled))).reset_index(drop=True)

    assert sampled["facility_id"].nunique() >= 20, (
        f"Muestra de paridad cubre solo {sampled['facility_id'].nunique()} facilities (mín 20)"
    )
    assert len(sampled) >= 100, (
        f"Muestra de paridad tiene solo {len(sampled)} filas (mín 100)"
    )

    batch_calculator_parity_maxdiff = check_batch_calculator_parity(
        sampled, stats, fe_path=str(FE_PATH), tol=1e-8, verbose=True
    )
    print(f"  Paridad verificada en {time.perf_counter()-t0:.1f}s "
          f"(max_diff={batch_calculator_parity_maxdiff:.2e})")

    # ------------------------------------------------------------------
    # Entrenar scaler + modelo
    # ------------------------------------------------------------------
    print("\n[4/5] Entrenando scaler y IsolationForest...")
    t0 = time.perf_counter()
    scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(X_train)
    X_train_scaled = np.clip(scaler.transform(X_train), -10, 10).astype(np.float32)
    print(f"  Scaler fit. X_train_scaled shape: {X_train_scaled.shape} "
          f"({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    model = IsolationForest(
        n_estimators=200,
        max_samples=512,
        max_features=0.6,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    ).fit(X_train_scaled)
    train_time = time.perf_counter() - t0
    print(f"  IsolationForest fit. n_estimators={model.n_estimators}, "
          f"max_samples={model.max_samples}, max_features={model.max_features} "
          f"({train_time:.1f}s)")

    # Guardar modelo y scaler
    model_path = MODELS / "isolation_forest_frame_v1.joblib"
    scaler_path = MODELS / "scaler_frame_v1.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Guardado: {model_path.name}, {scaler_path.name}")

    # Metadata parcial (las métricas de sesgo se añaden en el siguiente paso)
    built_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "feature_version": "frame-v1",
        "feature_names": FRAME_V1_FEATURE_NAMES,
        "n_features": len(FRAME_V1_FEATURE_NAMES),
        "model_recipe": {
            "n_estimators": 200,
            "max_samples": 512,
            "max_features": 0.6,
            "contamination": "auto",
            "random_state": 42,
            "n_jobs": -1,
        },
        "scaler_config": {
            "class": "RobustScaler",
            "quantile_range": [5.0, 95.0],
            "post_transform_clip": [-10, 10],
        },
        "train_rows": int(X_train.shape[0]),
        "built_at": built_at,
        "stats_artifact": "facility_stats_v1.json",
        "batch_calculator_parity_maxdiff": float(batch_calculator_parity_maxdiff),
        "parity_check": {
            "status": "PASS",
            "tol": 1e-8,
            "n_rows": len(sampled),
            "n_facilities": int(sampled["facility_id"].nunique()),
            "max_diff": float(batch_calculator_parity_maxdiff),
        },
        "bias_metrics": None,  # placeholder — se rellena abajo
    }

    # Liberar train data
    del train_df, X_train, X_train_scaled

    # ------------------------------------------------------------------
    # Gate de sesgo: val set completo
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluando gates de sesgo sobre val set completo...")
    t0 = time.perf_counter()
    val_df = pd.read_parquet(VAL_PARQUET, columns=NEEDED_COLS)
    print(f"  val shape: {val_df.shape} ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    val_df = add_frame_features_from_artifact(val_df, stats)
    print(f"  frame features añadidas a val ({time.perf_counter()-t0:.1f}s)")

    X_val = val_df[FRAME_V1_FEATURE_NAMES].to_numpy(dtype=np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.clip(scaler.transform(X_val), -10, 10).astype(np.float32)

    t0 = time.perf_counter()
    scores = -model.decision_function(X_val_scaled)
    print(f"  Scores computados ({time.perf_counter()-t0:.1f}s)")

    n_val = len(scores)

    # --- Gate 1: sesgo de monto top-5% ---
    k5 = int(n_val * 0.05)
    top5_idx = np.argsort(scores)[-k5:]
    val_amount = val_df["amount"].to_numpy(dtype=np.float64)
    top5_amount_ratio = float(val_amount[top5_idx].mean() / val_amount.mean())
    gate1_pass = top5_amount_ratio < 4.0

    # --- Gate 2: off-hours local ---
    # is_off_hours_loc fue añadida por add_frame_features_from_artifact
    off_hours_local_pct = float(val_df["is_off_hours_loc"].mean())
    gate2_pass = 0.03 <= off_hours_local_pct <= 0.07  # banda operativa ~4-5%

    # off-hours UTC (columna is_off_hours del parquet, pre-computada en FeatureEngineer)
    off_hours_utc_pct = float(val_df["is_off_hours"].mean())

    # Resumen legible
    g1_label = "PASS" if gate1_pass else "FAIL"
    g2_label = "PASS" if gate2_pass else "FAIL"
    print(f"\n  GATES DE SESGO:")
    print(f"  top5 amount ratio: {top5_amount_ratio:.2f}x  (gate <4x: {g1_label})  "
          f"[baseline: 11.79x]")
    print(f"  off-hours local:   {off_hours_local_pct*100:.2f}%  (gate ~4-5%: {g2_label})  "
          f"[UTC baseline: {off_hours_utc_pct*100:.2f}%]")
    print(f"\n  Resumen: top5={top5_amount_ratio:.2f}x (gate <4×: {g1_label}) | "
          f"off-hours local: {off_hours_local_pct*100:.2f}% vs UTC {off_hours_utc_pct*100:.2f}% "
          f"(gate ~4-5%: {g2_label})")

    # Nota de debug si falla algún gate
    debug_note = (
        "Si un gate falla: correr ./venv/bin/python -m pytest tests/test_parity_phase1.py -q "
        "para verificar que las features de marco están bien computadas vía el calculator de 01-02. "
        "Revisar también currency_fallbacks del artefacto facility_stats_v1.json (01-01). "
        "Research open question 3: si top5_amount_ratio > 4x, considerar ablación "
        "(añadir amount_fac_z — research open question 1)."
    )

    # --- Escribir bias report ---
    bias_report = {
        "top5_amount_ratio": round(top5_amount_ratio, 6),
        "gate1_pass": gate1_pass,
        "gate1_criterion": "top5_amount_ratio < 4.0",
        "off_hours_local_pct": round(off_hours_local_pct, 6),
        "off_hours_utc_pct": round(off_hours_utc_pct, 6),
        "gate2_pass": gate2_pass,
        "gate2_criterion": "0.03 <= off_hours_local_pct <= 0.07 (banda ~4-5%)",
        "baseline_top5": 11.79,
        "baseline_offhours_utc": 0.2978,
        "n_val_rows": n_val,
        "built_at": built_at,
        "auc_note": "AUC vs pure_fraud NO evaluado aquí — diagnóstico circular, no gate. "
                    "Ver baseline_v0.json para clasificación diagnostic_circular_not_a_gate_metric.",
        "debug_if_gate_fails": debug_note,
        "methodology": "no_supervisado: modelo entrena SIN etiquetas; proxy SOLO para diagnóstico",
    }

    bias_report_path = ROOT / "output" / "frame_v1_bias_report.json"
    with open(bias_report_path, "w") as f:
        json.dump(bias_report, f, indent=2)
    print(f"  Guardado: {bias_report_path.name}")

    # --- Actualizar metadata con bias_metrics ---
    metadata["bias_metrics"] = {
        "top5_amount_ratio": round(top5_amount_ratio, 6),
        "gate1_pass": gate1_pass,
        "off_hours_local_pct": round(off_hours_local_pct, 6),
        "off_hours_utc_pct": round(off_hours_utc_pct, 6),
        "gate2_pass": gate2_pass,
        "n_val_rows": n_val,
        "all_gates_pass": gate1_pass and gate2_pass,
    }

    metadata_path = MODELS / "model_metadata_frame_v1.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Guardado: {metadata_path.name}")

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    total_time = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"COMPLETADO en {total_time:.0f}s")
    print(f"  Paridad batch↔calculator: {batch_calculator_parity_maxdiff:.2e} (tol 1e-8: PASS)")
    print(f"  Gate 1 — top5 ratio:       {top5_amount_ratio:.2f}x (<4x: {g1_label})")
    print(f"  Gate 2 — off-hours local:  {off_hours_local_pct*100:.2f}% (~4-5%: {g2_label})")
    print(f"  Artefactos frame-v1:")
    print(f"    {model_path}")
    print(f"    {scaler_path}")
    print(f"    {metadata_path}")
    print(f"    {bias_report_path}")
    print(f"{'='*60}")

    if not (gate1_pass and gate2_pass):
        print("\n  ADVERTENCIA: uno o más gates fallaron. Ver debug_if_gate_fails en bias_report.")
        print(f"  {debug_note}")


if __name__ == "__main__":
    main()
