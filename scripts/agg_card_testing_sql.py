#!/usr/bin/env python3
"""Aggregation SQL builder — replica las definiciones de context.py en un
único pase batcheado (self-joins), para el conjunto de txns de un rango.

Devuelve un DataFrame con exactamente las columnas raw que consume
add_frame_features_from_artifact (NEEDED_COLS) + RULE_FIELDS + id/user_id/amount.

Definiciones replicadas 1:1 de src/fraud_detector/scoring/context.py:
  - amount (USD) = reservation_paid_out * rate_case  (ataque/baseline = USD)
  - time_since_last_txn = t - max(created_at | created_at < t)   (0 si no hay)
  - user_amount_24h  = sum USD en [t-24h, t)
  - user_txn_count_1h/24h = count en [t-1h/24h, t)
  - user_distinct_facilities_30d, user_distinct_methods  (30d, < t, no reversal/free)
  - user_debit_count_30d, user_debit_amount_30d, prepaid_spend_30d (30d, < t; credit incluye reversal/free segun CREDIT_SQL)
  - credit_flow_ratio = debit_amount / (prepaid + 0.01)
  - category_entropy_30d = Shannon sobre groupArray(category) (30d, < t, no reversal/free)
  - user_merchandise_ratio_30d = countIf(cat=merchandise)/count (30d, <t, no reversal/free)
  - gateway_change_recent / is_main_gateway / is_first_gateway_for_user / source_change_recent (< t, no reversal/free)
  - user_account_age_days = (t - user.created_at) en días
  - discount_ratio = discount_usd / max(amount_usd, 0.01)
  - has_tip = tip > 0
  - same_amount_count_1h/24h (para RULE_FIELDS, no feature del modelo)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_detector.scoring.context import AMOUNT_USD_SQL  # noqa: E402
from fraud_detector.utils.currency import clickhouse_rate_case  # noqa: E402

RATE = clickhouse_rate_case()
DISCOUNT_USD_SQL = f"(discount * {RATE})"

# Versión de AMOUNT_USD_SQL con columnas calificadas por alias 'h.' (para self-joins).
# AMOUNT_USD_SQL = "(reservation_paid_out * multiIf(upper(currency)=...))"
RATE_H = clickhouse_rate_case().replace("upper(currency)", "upper(h.currency)")
AMOUNT_USD_H = f"(h.reservation_paid_out * {RATE_H})"

DB = "pbp_productionDB_optimized"


def build_target_sql(fid, start, end):
    """Txns objetivo (depuradas) con campos directos por fila."""
    return f"""
    SELECT
        p.id AS id,
        p.user_id AS user_id,
        p.effective_user_id AS effective_user_id,
        p.facility_id AS facility_id,
        p.created_at AS created_at,
        p.status AS status,
        ({AMOUNT_USD_SQL}) AS amount,
        p.currency AS currency,
        p.gateway AS gateway,
        toString(p.source_enum) AS source_enum,
        p.payment_method AS payment_method,
        p.category AS category,
        toUInt8(p.club_credit_flag) AS is_club_credit,
        toUInt8(p.paid_by_manager) AS paid_by_manager,
        {DISCOUNT_USD_SQL} AS discount_usd,
        (p.tip) AS tip
    FROM {DB}.payments AS p FINAL
    WHERE p.facility_id = {{fid:Int64}}
      AND p.created_at >= {{start:DateTime}}
      AND p.created_at <  {{end:DateTime}}
      AND p.payment_method NOT IN ('reversal', 'free')
      AND p.user_id != 0
      AND p._peerdb_is_deleted = 0
    ORDER BY p.created_at, p.id
    """


def shannon(cats):
    if not cats:
        return 0.0
    from collections import Counter
    c = Counter(cats)
    tot = sum(c.values())
    return -sum((v / tot) * math.log2(v / tot) for v in c.values() if v > 0)


def compute_aggregates(client, target_df, fid):
    """Para cada txn objetivo computa los agregados vía SQL escaneando el
    historial de los usuarios involucrados. Un query por bloque de agregados,
    pero UN solo query por bloque para TODAS las txns (no por fila).
    """
    ids = target_df["id"].tolist()
    uids = sorted(set(int(u) for u in target_df["user_id"].tolist()))
    uids_sql = ",".join(str(u) for u in uids)
    id_list = ",".join(str(i) for i in ids)

    # Tabla temporal lógica: (id, user_id, ts, amt_usd) de las txns objetivo,
    # inyectada como subconsulta desde la misma tabla payments filtrando por id.
    tgt = f"""
        SELECT id AS tid, user_id AS tuid, created_at AS t,
               round(({AMOUNT_USD_SQL}), 2) AS tamt
        FROM {DB}.payments FINAL
        WHERE id IN ({id_list}) AND _peerdb_is_deleted = 0
    """

    # --- Velocidad + amount_24h + last_txn (self-join a historial no reversal/free) ---
    q_vel = f"""
    SELECT g.tid AS id,
        countIf(h.created_at >= g.t - INTERVAL 1 HOUR AND h.created_at < g.t) AS user_txn_count_1h,
        countIf(h.created_at >= g.t - INTERVAL 24 HOUR AND h.created_at < g.t) AS user_txn_count_24h,
        sumIf(({AMOUNT_USD_H}), h.created_at >= g.t - INTERVAL 24 HOUR AND h.created_at < g.t) AS user_amount_24h,
        maxIf(h.created_at, h.created_at < g.t) AS last_txn_at
    FROM ({tgt}) AS g
    LEFT JOIN {DB}.payments AS h FINAL
        ON h.user_id = g.tuid
    WHERE h.user_id IN ({uids_sql})
      AND h._peerdb_is_deleted = 0
      AND h.payment_method NOT IN ('reversal','free')
      AND h.created_at < g.t
    GROUP BY g.tid, g.t
    """

    # --- Behavior 30d (no reversal/free) ---
    q_beh = f"""
    SELECT g.tid AS id,
        uniqExactIf(h.facility_id, 1=1) AS user_distinct_facilities_30d,
        uniqExactIf(h.payment_method, 1=1) AS user_distinct_methods
    FROM ({tgt}) AS g
    LEFT JOIN {DB}.payments AS h FINAL ON h.user_id = g.tuid
    WHERE h.user_id IN ({uids_sql}) AND h._peerdb_is_deleted = 0
      AND h.payment_method NOT IN ('reversal','free')
      AND h.created_at >= g.t - INTERVAL 30 DAY AND h.created_at < g.t
    GROUP BY g.tid
    """

    # --- Credit 30d (CREDIT_SQL: SIN filtro payment_method) ---
    q_cred = f"""
    SELECT g.tid AS id,
        countIf(h.category = 'debit') AS user_debit_count_30d,
        sumIf(({AMOUNT_USD_H}), h.category = 'debit') AS user_debit_amount_30d,
        sumIf(({AMOUNT_USD_H}), h.payment_method = 'prepaid') AS prepaid_spend_30d
    FROM ({tgt}) AS g
    LEFT JOIN {DB}.payments AS h FINAL ON h.user_id = g.tuid
    WHERE h.user_id IN ({uids_sql}) AND h._peerdb_is_deleted = 0
      AND h.created_at >= g.t - INTERVAL 30 DAY AND h.created_at < g.t
    GROUP BY g.tid
    """

    # --- Diversity 30d (no reversal/free): categorias + merchandise ratio ---
    q_div = f"""
    SELECT g.tid AS id,
        groupArray(h.category) AS categories_30d,
        countIf(h.category = 'merchandise') * 1.0 / greatest(count(), 1) AS user_merchandise_ratio_30d
    FROM ({tgt}) AS g
    LEFT JOIN {DB}.payments AS h FINAL ON h.user_id = g.tuid
    WHERE h.user_id IN ({uids_sql}) AND h._peerdb_is_deleted = 0
      AND h.payment_method NOT IN ('reversal','free')
      AND h.created_at >= g.t - INTERVAL 30 DAY AND h.created_at < g.t
    GROUP BY g.tid
    """

    # --- Gateway (< t, no reversal/free) — replica GATEWAY_SQL con argMax por txn ---
    # Necesitamos gateway/source de la txn objetivo para comparar.
    q_gw = f"""
    SELECT g.tid AS id,
        if(count(h.created_at) = 0, 0, argMax(h.gateway, h.created_at) != g.gw) AS gateway_change_recent,
        if(count(h.created_at) = 0, 0, countIf(h.gateway = g.gw) * 1.0 / greatest(count(h.created_at),1) >= 0.5) AS is_main_gateway,
        if(count(h.created_at) = 0, 1, countIf(h.gateway = g.gw) = 0) AS is_first_gateway_for_user,
        if(count(h.created_at) = 0, 0, argMax(toString(h.source_enum), h.created_at) != g.src) AS source_change_recent
    FROM (
        SELECT id AS tid, user_id AS tuid, created_at AS t,
               if(gateway = '', 'unknown', gateway) AS gw,
               if(toString(source_enum) = '', 'unknown', toString(source_enum)) AS src
        FROM {DB}.payments FINAL WHERE id IN ({id_list}) AND _peerdb_is_deleted = 0
    ) AS g
    LEFT JOIN {DB}.payments AS h FINAL ON h.user_id = g.tuid
        AND h.created_at < g.t
        AND h.payment_method NOT IN ('reversal','free')
        AND h._peerdb_is_deleted = 0
    GROUP BY g.tid, g.gw, g.src
    """

    # --- same_amount_count_1h/24h (RULE, no feature) ---
    q_same = f"""
    SELECT g.tid AS id,
        countIf(round(({AMOUNT_USD_H}),2) = g.tamt AND h.created_at >= g.t - INTERVAL 1 HOUR AND h.created_at < g.t) AS same_amount_count_1h,
        countIf(round(({AMOUNT_USD_H}),2) = g.tamt AND h.created_at >= g.t - INTERVAL 24 HOUR AND h.created_at < g.t) AS same_amount_count_24h
    FROM ({tgt}) AS g
    LEFT JOIN {DB}.payments AS h FINAL ON h.user_id = g.tuid
    WHERE h.user_id IN ({uids_sql}) AND h._peerdb_is_deleted = 0
      AND h.payment_method NOT IN ('reversal','free')
      AND h.created_at < g.t
    GROUP BY g.tid, g.t, g.tamt
    """

    # --- user_created_at + role ---
    q_user = f"""
    SELECT id, coalesce(fu.role, 'player') AS user_role, u.created_at AS user_created_at
    FROM (SELECT id, user_id, facility_id FROM {DB}.payments FINAL WHERE id IN ({id_list}) AND _peerdb_is_deleted=0) AS p
    LEFT ANY JOIN (SELECT user_id, facility_id, role FROM {DB}.facilities_users FINAL WHERE _peerdb_is_deleted=0) AS fu
        ON p.user_id=fu.user_id AND p.facility_id=fu.facility_id
    LEFT ANY JOIN (SELECT id AS uid, created_at FROM {DB}.users FINAL WHERE _peerdb_is_deleted=0) AS u
        ON p.user_id=u.uid
    """

    print("      q_vel...", flush=True); d_vel = client.query_df(q_vel)
    print("      q_beh...", flush=True); d_beh = client.query_df(q_beh)
    print("      q_cred...", flush=True); d_cred = client.query_df(q_cred)
    print("      q_div...", flush=True); d_div = client.query_df(q_div)
    print("      q_gw...", flush=True); d_gw = client.query_df(q_gw)
    print("      q_same...", flush=True); d_same = client.query_df(q_same)
    print("      q_user...", flush=True); d_user = client.query_df(q_user)

    # category_entropy desde categories_30d
    d_div["category_entropy_30d"] = d_div["categories_30d"].apply(
        lambda a: shannon(list(a) if a is not None else [])
    )
    d_div = d_div.drop(columns=["categories_30d"])

    # credit_flow_ratio = debit_amount / (prepaid + 0.01)
    d_cred["credit_flow_ratio"] = d_cred["user_debit_amount_30d"] / (d_cred["prepaid_spend_30d"] + 0.01)
    d_cred = d_cred.drop(columns=["prepaid_spend_30d"])

    # merge todos por id
    out = target_df.copy()
    for d in [d_vel, d_beh, d_cred, d_div, d_gw, d_same, d_user]:
        out = out.merge(d, on="id", how="left")

    # time_since_last_txn (0 si sin previa)
    ts = pd.to_datetime(out["created_at"], utc=True)
    last = pd.to_datetime(out["last_txn_at"], utc=True, errors="coerce")
    tsl = (ts - last).dt.total_seconds()
    out["time_since_last_txn"] = tsl.fillna(0.0).clip(lower=0.0)

    # user_account_age_days
    uc = pd.to_datetime(out["user_created_at"], utc=True, errors="coerce")
    age = (ts - uc).dt.total_seconds() / 86400.0
    out["user_account_age_days"] = age.fillna(0.0).clip(lower=0.0).astype(int)

    # discount_ratio / has_tip
    out["discount_ratio"] = out["discount_usd"].astype(float) / np.maximum(out["amount"].astype(float), 0.01)
    out["has_tip"] = (out["tip"].astype(float) > 0).astype(int)

    # rellenar nulos de agregados (usuarios sin historial) con 0
    fill0 = ["user_txn_count_1h", "user_txn_count_24h", "user_amount_24h",
             "user_distinct_facilities_30d", "user_distinct_methods",
             "user_debit_count_30d", "user_debit_amount_30d", "credit_flow_ratio",
             "category_entropy_30d", "user_merchandise_ratio_30d",
             "gateway_change_recent", "is_main_gateway", "is_first_gateway_for_user",
             "source_change_recent", "same_amount_count_1h", "same_amount_count_24h"]
    for c in fill0:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out
