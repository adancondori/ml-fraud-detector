"""
Simulate the Reserve Hudson Yards card-testing bot attack (2026-04-29)
against the trained Isolation Forest model to assess detectability.

Uses the actual trained model, scaler, and feature engineer from the thesis pipeline.
Constructs 47 synthetic feature vectors matching the case characteristics.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detector.features.engineering import FEATURE_NAMES

OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"

# ---------------------------------------------------------------------------
# 1. Load trained artifacts
# ---------------------------------------------------------------------------
print("=" * 70)
print("SIMULACIÓN: Caso Reserve Hudson Yards — Card-Testing Bot Attack")
print("=" * 70)

print("\n[1/5] Cargando artefactos entrenados...")
if_model = joblib.load(MODELS_DIR / "isolation_forest.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")

# Extract facility_avg_amount for facility 499 from the fitted ContextualFeatures
contextual_group = None
for group in feature_engineer._groups:
    if hasattr(group, "_facility_avg_amount"):
        contextual_group = group
        break

facility_499_avg = None
if contextual_group and 499 in contextual_group._facility_avg_amount:
    facility_499_avg = contextual_group._facility_avg_amount[499]
    print(f"  → Facility 499 encontrada en datos de entrenamiento: avg_amount = ${facility_499_avg:.2f}")
else:
    facility_499_avg = contextual_group._global_avg_amount if contextual_group else 368.61
    print(f"  → Facility 499 NO encontrada. Usando promedio global: ${facility_499_avg:.2f}")

# Extract global_avg_amount from TransactionalFeatures
global_avg_amount = 368.61  # fallback
for group in feature_engineer._groups:
    if hasattr(group, "_global_avg_amount") and not hasattr(group, "_facility_avg_amount"):
        global_avg_amount = group._global_avg_amount
        break
print(f"  → Promedio global de monto: ${global_avg_amount:.2f}")

# Extract staff role stats
staff_group = None
for group in feature_engineer._groups:
    if hasattr(group, "_role_currency_stats"):
        staff_group = group
        break

player_usd_stats = {"mean": global_avg_amount, "std": 1.0}
if staff_group:
    player_usd_stats = staff_group._role_currency_stats.get(
        ("player", "USD"),
        staff_group._currency_stats.get("USD", player_usd_stats),
    )
    print(f"  → Stats player/USD: mean=${player_usd_stats['mean']:.2f}, std=${player_usd_stats['std']:.2f}")

# ---------------------------------------------------------------------------
# 2. Build 47 synthetic transactions matching the case
# ---------------------------------------------------------------------------
print("\n[2/5] Construyendo 47 vectores de features sintéticos...")

# Exact timestamps from the case file (ET timezone)
timestamps_str = [
    "2026-04-29 05:11", "2026-04-29 05:18", "2026-04-29 05:18",
    "2026-04-29 05:22", "2026-04-29 05:22", "2026-04-29 05:25",
    "2026-04-29 05:27", "2026-04-29 05:32", "2026-04-29 05:32",
    "2026-04-29 05:34", "2026-04-29 05:44", "2026-04-29 05:44",
    "2026-04-29 05:47", "2026-04-29 05:47", "2026-04-29 05:49",
    "2026-04-29 05:51", "2026-04-29 05:51", "2026-04-29 05:53",
    "2026-04-29 06:26", "2026-04-29 06:31", "2026-04-29 06:38",
    "2026-04-29 06:39", "2026-04-29 06:42", "2026-04-29 06:43",
    "2026-04-29 06:45", "2026-04-29 06:45", "2026-04-29 06:48",
    "2026-04-29 06:51", "2026-04-29 06:56", "2026-04-29 09:51",
    "2026-04-29 09:55", "2026-04-29 09:55", "2026-04-29 09:57",
    "2026-04-29 09:59", "2026-04-29 09:59", "2026-04-29 10:00",
    "2026-04-29 10:03", "2026-04-29 10:18", "2026-04-29 10:31",
    "2026-04-29 10:51", "2026-04-29 10:51", "2026-04-29 11:02",
    "2026-04-29 12:05", "2026-04-29 14:46", "2026-04-29 17:39",
    "2026-04-29 18:24", "2026-04-29 18:38",
]

n_txn = len(timestamps_str)
assert n_txn == 47, f"Expected 47, got {n_txn}"

timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M") for ts in timestamps_str]
hours = [t.hour for t in timestamps]

# All bot characteristics from the case:
amount = 5.0  # USD tech fee only
discount = 0.0
tip = 0.0

rows = []
for i in range(n_txn):
    h = hours[i]
    row = {
        # A) Transactional
        "amount": amount,
        "log_amount": np.log1p(amount),
        "amount_usd_ratio": amount / max(global_avg_amount, 1e-8),
        "discount_ratio": discount / (amount + 0.01),
        "has_tip": 0,
        # B) Temporal
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "day_of_week": 3,  # Wednesday (2026-04-29 is Wednesday)
        "is_weekend": 0,
        "is_off_hours": 1 if h in [23, 0, 1, 2, 3, 4, 5, 6] else 0,
        # C) Velocity — each bot account has 1 successful txn (47 accounts, 47 txns)
        #    but some had MULTIPLE ATTEMPTS (488 declines + 47 captures = 535)
        #    The model only sees captures, so for first capture per bot: count = 0
        "user_txn_count_1h": 0.0,  # first successful txn for this bot account
        "user_txn_count_24h": 0.0,  # first successful txn for this bot account
        "time_since_last_txn": 0.0,  # no prior txn (fillna(0) in pipeline)
        "user_amount_24h": 0.0,  # no prior txn
        # D) Behavioral — brand new accounts, zero history
        "user_distinct_facilities_30d": 0.0,  # new account, no prior facilities
        "user_distinct_methods": 0.0,  # no prior methods seen (before current)
        "user_reversal_ratio_30d": 0.0,  # no prior reversals
        "user_account_age_days": 0,  # created same day
        "user_discount_ratio_30d": 0.0,  # no prior discounts
        # E) Contextual
        "facility_avg_amount": facility_499_avg,
        "amount_facility_ratio": amount / (facility_499_avg + 0.01),
        # F) Credit/Flow — no credit activity
        "is_club_credit": 0,
        "user_debit_count_30d": 0.0,
        "user_debit_amount_30d": 0.0,
        "credit_flow_ratio": 0.0,
        # G) Staff/Role — all are fake player accounts
        "is_staff": 0,
        "paid_by_manager": 0,  # self-service via white label web
        "staff_amount_zscore": (amount - player_usd_stats["mean"])
        / max(player_usd_stats["std"], 1e-8),
        # H) Operational Diversity — zero diversity
        "category_entropy_30d": 0.0,  # only one type
        "user_reversal_count_30d": 0.0,
        "user_merchandise_ratio_30d": 0.0,
    }
    rows.append(row)

df_bot = pd.DataFrame(rows, columns=FEATURE_NAMES)
print(f"  → {n_txn} transacciones construidas con {len(FEATURE_NAMES)} features")

# ---------------------------------------------------------------------------
# 3. Scale and score with trained IF model
# ---------------------------------------------------------------------------
print("\n[3/5] Escalando y puntuando con Isolation Forest entrenado...")

X_bot_scaled = scaler.transform(df_bot)

# score_samples(): lower = more anomalous (matches stored score_if in test set)
bot_scores = if_model.score_samples(X_bot_scaled)

print(f"\n  IF scores (score_samples — lower = more anomalous):")
print(f"    Mean:   {bot_scores.mean():.6f}")
print(f"    Std:    {bot_scores.std():.6f}")
print(f"    Min:    {bot_scores.min():.6f}  (most anomalous)")
print(f"    Max:    {bot_scores.max():.6f}  (least anomalous)")
print(f"    Median: {np.median(bot_scores):.6f}")

# ---------------------------------------------------------------------------
# 4. Compare against test set score distribution
# ---------------------------------------------------------------------------
print("\n[4/5] Comparando con distribución del test set (2.5M transacciones)...")

test_scores_path = OUTPUT_DIR / "scores" / "test_scores.parquet"
df_test = pd.read_parquet(test_scores_path)
test_if_scores = df_test["score_if"].values

# For score_samples: lower = more anomalous
# Percentile = % of test txns that score HIGHER (less anomalous) than bot txns
percentiles_anomaly = []
for s in bot_scores:
    pct = (test_if_scores > s).mean() * 100  # % more normal than this
    percentiles_anomaly.append(pct)
percentiles_anomaly = np.array(percentiles_anomaly)

print(f"\n  Distribución score_if del test set (n={len(test_if_scores):,}):")
print(f"    Mean:  {test_if_scores.mean():.6f}")
print(f"    Std:   {test_if_scores.std():.6f}")
print(f"    P1 (most anomalous):   {np.percentile(test_if_scores, 1):.6f}")
print(f"    P5:                    {np.percentile(test_if_scores, 5):.6f}")
print(f"    P50 (median):          {np.percentile(test_if_scores, 50):.6f}")
print(f"    P95:                   {np.percentile(test_if_scores, 95):.6f}")
print(f"    P99 (most normal):     {np.percentile(test_if_scores, 99):.6f}")

print(f"\n  Posición de las transacciones bot (% de txns test MÁS normales):")
print(f"    Mean:   {percentiles_anomaly.mean():.1f}% del test set es más normal")
print(f"    Min:    {percentiles_anomaly.min():.1f}%")
print(f"    Max:    {percentiles_anomaly.max():.1f}%")
print(f"    Median: {np.median(percentiles_anomaly):.1f}%")

# Top-k thresholds (lower score = more anomalous)
top5_threshold = np.percentile(test_if_scores, 5)   # bottom 5% = top 5% most anomalous
top1_threshold = np.percentile(test_if_scores, 1)   # bottom 1% = top 1% most anomalous
top10_threshold = np.percentile(test_if_scores, 10)

n_in_top5 = (bot_scores <= top5_threshold).sum()
n_in_top1 = (bot_scores <= top1_threshold).sum()
n_in_top10 = (bot_scores <= top10_threshold).sum()

print(f"\n  Detección con umbrales de anomalía:")
print(f"    Umbral top-10% anómalo: score <= {top10_threshold:.6f}")
print(f"    Umbral top-5%  anómalo: score <= {top5_threshold:.6f}")
print(f"    Umbral top-1%  anómalo: score <= {top1_threshold:.6f}")
print(f"    Bot txns en top-10%: {n_in_top10}/{n_txn} ({n_in_top10/n_txn*100:.1f}%)")
print(f"    Bot txns en top-5%:  {n_in_top5}/{n_txn} ({n_in_top5/n_txn*100:.1f}%)")
print(f"    Bot txns en top-1%:  {n_in_top1}/{n_txn} ({n_in_top1/n_txn*100:.1f}%)")

# Compare bot scores vs known proxy-anomalies in test set
# Load proxy info if available in processed data
proxy_path = PROJECT_ROOT / "data" / "processed" / "test.parquet"
if proxy_path.exists():
    df_test_full = pd.read_parquet(proxy_path, columns=["id", "status"])
    df_merged = df_test.merge(df_test_full, on="id", how="left")
    proxy_mask = df_merged["status"].isin(["totally_refunded", "refunded_to_credit"])
    proxy_pos_scores = df_merged.loc[proxy_mask, "score_if"].values
    proxy_neg_scores = df_merged.loc[~proxy_mask, "score_if"].values

    print(f"\n  Comparación con anomalías proxy del test set:")
    print(f"    Score medio proxy=1 (refunds, n={len(proxy_pos_scores):,}): {proxy_pos_scores.mean():.6f}")
    print(f"    Score medio proxy=0 (normales, n={len(proxy_neg_scores):,}):  {proxy_neg_scores.mean():.6f}")
    print(f"    Score medio BOT ATTACK (n={n_txn}):                  {bot_scores.mean():.6f}")
    if bot_scores.mean() < proxy_pos_scores.mean():
        print(f"    → Bot attack puntúa MÁS ANÓMALO que refunds promedio ✓")
    else:
        delta = bot_scores.mean() - proxy_pos_scores.mean()
        print(f"    → Bot attack puntúa {delta:.6f} más alto (menos anómalo) que refunds promedio")
else:
    print("\n  (No se encontró test.parquet para comparar con proxy)")

# ---------------------------------------------------------------------------
# 5. Feature contribution analysis
# ---------------------------------------------------------------------------
print("\n[5/5] Análisis de contribución por feature...")
print("\n  Valores medios de las features bot vs. estadísticas de entrenamiento:")
print(f"  {'Feature':<30} {'Bot Mean':>12} {'Train Mean':>12} {'Train Std':>12} {'Z-score':>10}")
print("  " + "-" * 76)

feature_stats = pd.read_csv(OUTPUT_DIR / "feature_statistics.csv", index_col=0)
z_scores = {}
for feat in FEATURE_NAMES:
    bot_val = df_bot[feat].mean()
    if feat in feature_stats.index:
        train_mean = feature_stats.loc[feat, "mean"]
        train_std = feature_stats.loc[feat, "std"]
        z = (bot_val - train_mean) / max(train_std, 1e-8)
        z_scores[feat] = abs(z)
        flag = " ◄◄◄" if abs(z) > 2.0 else " ◄" if abs(z) > 1.0 else ""
        print(f"  {feat:<30} {bot_val:>12.4f} {train_mean:>12.4f} {train_std:>12.4f} {z:>10.2f}{flag}")

print("\n  Features más desviadas (|z| > 1.0):")
sorted_z = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)
for feat, z in sorted_z:
    if z > 1.0:
        print(f"    {feat}: z = {z:.2f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)

detected = n_in_top5 > 0
detection_strength = "FUERTE" if n_in_top5 >= 40 else "MODERADA" if n_in_top5 >= 20 else "DÉBIL" if n_in_top5 > 0 else "NULA"

print(f"""
Caso: Card-testing bot attack — Reserve Hudson Yards (2026-04-29)
Transacciones simuladas: {n_txn}
Modelo: Isolation Forest (entrenado Ene-Jun 2025, {len(FEATURE_NAMES)} features)
Score IF medio bot: {bot_scores.mean():.6f} (test set mean: {test_if_scores.mean():.6f})

DETECCIÓN: {'SÍ' if detected else 'NO'} — Señal {detection_strength}
{percentiles_anomaly.mean():.1f}% del test set es más normal que estas transacciones

Transacciones en top-10% anómalo: {n_in_top10}/{n_txn} ({n_in_top10/n_txn*100:.1f}%)
Transacciones en top-5%  anómalo: {n_in_top5}/{n_txn} ({n_in_top5/n_txn*100:.1f}%)
Transacciones en top-1%  anómalo: {n_in_top1}/{n_txn} ({n_in_top1/n_txn*100:.1f}%)

LIMITACIÓN CRÍTICA: El modelo opera en batch (no real-time).
No habría PREVENIDO el ataque, solo lo habría DETECTADO después.
""")
