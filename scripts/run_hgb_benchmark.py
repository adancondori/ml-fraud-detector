#!/usr/bin/env python3
"""
Pre-benchmark V4-CLEAN obligatorio (Gate A0 del PLAN-FINAL-v4).

Reproduce el HGB-CLEAN minimo sobre el universo 2025 sin features circulares:
- sin historial directo de reembolso (grupo D);
- sin target encoding (solo frequency encoding train-only);
- features minimas: amount + tiempo + categoricos via frequency + reserva point-in-time + token.

Splits temporales:
- train:      2025-01-01 a 2025-06-30
- validation: 2025-07-01 a 2025-08-31
- test legacy: 2025-09-01 a 2025-12-31
- test final:  2025-10-01 a 2025-12-31

Salida:
- output/v4/benchmarks/baseline_hgb_{variant}.json
- output/v4/benchmarks/baseline_hgb_{variant}_scores.parquet
- output/v4/benchmarks/feature_list_{variant}.json
- output/v4/benchmarks/simple_rule_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import clickhouse_connect
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "output" / "v4" / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42


def ch_client():
    load_dotenv(REPO / ".env")
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 8443)),
        username=os.environ["CLICKHOUSE_USER"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
        database=os.environ.get("CLICKHOUSE_DATABASE", "pbp_productionDB_optimized"),
        secure=os.environ.get("CLICKHOUSE_SECURE", "true").lower() == "true",
        send_receive_timeout=600,
    )


BASE_SQL_TEMPLATE = """
WITH p AS (
    SELECT
        id AS payment_id,
        user_id,
        facility_id,
        created_at,
        status,
        source_enum,
        payment_method,
        gateway,
        card_brand,
        currency,
        category,
        paid_by_manager,
        reservation_paid_out,
        original_amount_paid_out,
        discount,
        tax,
        tip,
        reservation_id,
        user_token_id,
        captured_at,
        payment_source
    FROM pbp_productionDB_optimized.payments FINAL
    WHERE created_at >= %(start)s
      AND created_at <  %(end)s
      AND payment_method != 'reversal'
      AND payment_method != 'free'
      AND user_id != 0
      AND _peerdb_is_deleted = 0
),
r AS (
    SELECT
        id,
        created_at AS reservation_created_at,
        date AS res_date,
        admin_booked,
        booked_from,
        generated_by_court,
        kind_enum,
        reservation_type
    FROM pbp_productionDB_optimized.reservations FINAL
    WHERE created_at >= '2024-12-01' AND created_at < '2026-02-01'
      AND _peerdb_is_deleted = 0
),
{ru_cte}
t AS (
    SELECT
        id,
        created_at AS token_created_at,
        is_default AS token_is_default,
        accttype AS token_accttype,
        gateway AS token_gateway,
        length(last4) > 0 AS token_has_last4
    FROM pbp_productionDB_optimized.user_tokens FINAL
    WHERE _peerdb_is_deleted = 0
),
pd_disc AS (
    SELECT payment_id AS pd_payment_id, anyLast(discount_id) AS discount_id, anyLast(discount_type) AS discount_type
    FROM pbp_productionDB_optimized.payment_discounts FINAL
    WHERE _peerdb_is_deleted = 0
    GROUP BY pd_payment_id
)
SELECT
    p.payment_id,
    p.user_id,
    p.facility_id,
    p.created_at,
    p.status,
    p.source_enum,
    p.payment_method,
    p.gateway,
    p.card_brand,
    p.currency,
    p.category,
    p.paid_by_manager,
    p.reservation_paid_out,
    p.original_amount_paid_out,
    p.discount,
    p.tax,
    p.tip,
    p.reservation_id AS reservation_id,
    p.user_token_id AS user_token_id,
    p.captured_at,
    p.payment_source,
    r.reservation_created_at,
    r.res_date,
    r.admin_booked,
    r.booked_from,
    r.generated_by_court,
    r.kind_enum,
    r.reservation_type,
    {ru_select}
    t.token_created_at,
    t.token_is_default,
    t.token_accttype,
    t.token_gateway,
    t.token_has_last4,
    pd_disc.discount_type AS coupon_discount_type
FROM p
LEFT ANY JOIN r       ON p.reservation_id = r.id
{ru_join}
LEFT ANY JOIN t       ON p.user_token_id  = t.id
LEFT ANY JOIN pd_disc ON p.payment_id     = pd_disc.pd_payment_id
{group_by}
"""


RU_CURRENT_CTE = """
ru AS (
    SELECT
        reservation_id AS ru_reservation_id,
        uniqExact(user_id) AS participant_count,
        countIf(user_affiliation_enum = 'teacher') AS teacher_rows,
        maxIf(1, user_affiliation_enum = 'invited') AS has_invited_raw,
        maxIf(1, free_pass) AS has_free_pass_raw,
        maxIf(1, resident) AS has_resident_raw
    FROM pbp_productionDB_optimized.reservations_users FINAL
    WHERE created_at >= '2024-12-01' AND created_at < '2026-02-01'
      AND _peerdb_is_deleted = 0
    GROUP BY ru_reservation_id
),
"""


RU_STRICT_CTE = """
ru AS (
    SELECT
        reservation_id AS ru_reservation_id,
        user_id AS ru_user_id,
        created_at AS ru_created_at,
        user_affiliation_enum,
        free_pass,
        resident
    FROM pbp_productionDB_optimized.reservations_users FINAL
    WHERE created_at >= '2024-12-01'
      AND created_at < %(end)s
      AND _peerdb_is_deleted = 0
),
"""


RU_CURRENT_SELECT = """
    coalesce(ru.participant_count, 0) AS participant_count,
    coalesce(ru.teacher_rows, 0) AS teacher_rows,
    coalesce(ru.has_invited_raw, 0) AS has_invited,
    coalesce(ru.has_free_pass_raw, 0) AS has_free_pass,
    coalesce(ru.has_resident_raw, 0) AS has_resident,
"""


RU_STRICT_SELECT = """
    uniqExactIf(ru.ru_user_id, ru.ru_created_at <= p.created_at) AS participant_count,
    countIf(ru.user_affiliation_enum = 'teacher' AND ru.ru_created_at <= p.created_at) AS teacher_rows,
    maxIf(toUInt8(1), ru.user_affiliation_enum = 'invited' AND ru.ru_created_at <= p.created_at) AS has_invited,
    maxIf(toUInt8(1), ru.free_pass AND ru.ru_created_at <= p.created_at) AS has_free_pass,
    maxIf(toUInt8(1), ru.resident AND ru.ru_created_at <= p.created_at) AS has_resident,
"""


RU_ZERO_SELECT = """
    toUInt16(0) AS participant_count,
    toUInt16(0) AS teacher_rows,
    toUInt8(0) AS has_invited,
    toUInt8(0) AS has_free_pass,
    toUInt8(0) AS has_resident,
"""


STRICT_GROUP_BY = """
GROUP BY
    p.payment_id, p.user_id, p.facility_id, p.created_at, p.status,
    p.source_enum, p.payment_method, p.gateway, p.card_brand, p.currency,
    p.category, p.paid_by_manager, p.reservation_paid_out,
    p.original_amount_paid_out, p.discount, p.tax, p.tip, p.reservation_id,
    p.user_token_id, p.captured_at, p.payment_source,
    r.reservation_created_at, r.res_date, r.admin_booked, r.booked_from,
    r.generated_by_court, r.kind_enum, r.reservation_type,
    t.token_created_at, t.token_is_default, t.token_accttype,
    t.token_gateway, t.token_has_last4, pd_disc.discount_type
"""


def query_for_variant(variant: str) -> str:
    if variant == "clean":
        return BASE_SQL_TEMPLATE.format(
            ru_cte=RU_CURRENT_CTE,
            ru_select=RU_CURRENT_SELECT,
            ru_join="LEFT ANY JOIN ru      ON p.reservation_id = ru.ru_reservation_id",
            group_by="",
        )
    if variant == "clean-strict":
        return BASE_SQL_TEMPLATE.format(
            ru_cte=RU_STRICT_CTE,
            ru_select=RU_STRICT_SELECT,
            ru_join="LEFT JOIN ru          ON p.reservation_id = ru.ru_reservation_id",
            group_by=STRICT_GROUP_BY,
        )
    if variant == "clean-no-ru":
        return BASE_SQL_TEMPLATE.format(
            ru_cte="",
            ru_select=RU_ZERO_SELECT,
            ru_join="",
            group_by="",
        )
    raise ValueError(f"unknown variant: {variant}")


def variant_slug(variant: str) -> str:
    return variant.replace("-", "_")


def extract(client, start: str, end: str, label: str, sql: str) -> pd.DataFrame:
    print(f"  extracting {label} [{start} .. {end})")
    t0 = time.time()
    df = client.query_df(sql, parameters={"start": start, "end": end})
    print(f"    rows={len(df):,}  in {time.time()-t0:.1f}s")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # A. Transaccion base
    amount = df["reservation_paid_out"].astype("float64")
    fallback = df["original_amount_paid_out"].astype("float64")
    amount = amount.where(amount > 0, fallback)
    out["amount"] = amount
    out["log_amount"] = np.log1p(amount.clip(lower=0))
    safe = amount.replace(0, np.nan)
    out["discount_ratio"] = df["discount"].astype("float64") / safe
    out["tax_ratio"] = df["tax"].astype("float64") / safe
    out["tip_ratio"] = df["tip"].astype("float64") / safe
    out["has_tip"] = (df["tip"].astype("float64") > 0).astype("int8")

    # B. Tiempo
    ts = pd.to_datetime(df["created_at"])
    hours = ts.dt.hour.astype("float64")
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    out["day_of_week"] = ts.dt.dayofweek.astype("int8")
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    out["is_off_hours"] = ((hours < 6) | (hours > 22)).astype("int8")
    out["month"] = ts.dt.month.astype("int8")

    # F. Reserva point-in-time
    res_date = pd.to_datetime(df["res_date"], errors="coerce")
    pay_date = ts.dt.normalize()
    lead = (res_date - pay_date).dt.days.astype("float64")
    out["reservation_lead_days"] = lead.fillna(-1)
    out["reservation_lead_bucket"] = pd.cut(
        lead.fillna(-1),
        bins=[-2, -0.5, 0.5, 7, 30, 365],
        labels=[0, 1, 2, 3, 4],
    ).astype("float64").fillna(0)
    out["admin_booked"] = df["admin_booked"].fillna(False).astype("int8")
    out["generated_by_court"] = df["generated_by_court"].fillna(False).astype("int8")
    out["booked_from"] = df["booked_from"].fillna(-1).astype("int16")
    out["kind_enum_code"] = pd.Categorical(df["kind_enum"]).codes.astype("int16")
    out["reservation_type_code"] = df["reservation_type"].fillna(-1).astype("int16")
    out["has_reservation"] = (df["reservation_id"].fillna(0) > 0).astype("int8")
    res_created = pd.to_datetime(df["reservation_created_at"], errors="coerce")
    payment_after_booking = (ts - res_created).dt.total_seconds() / 60.0
    out["payment_after_booking_minutes"] = (
        payment_after_booking.fillna(-1).clip(lower=-1, upper=525600)
    )
    out["payment_after_booking_bucket"] = pd.cut(
        payment_after_booking.fillna(-1),
        bins=[-2, -0.5, 5, 60, 1440, 43200, 525600],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype("float64").fillna(0)
    out["participant_count"] = df["participant_count"].fillna(0).astype("int16")
    out["has_invited"] = df["has_invited"].fillna(0).astype("int8")
    out["has_free_pass"] = df["has_free_pass"].fillna(0).astype("int8")
    out["has_resident"] = df["has_resident"].fillna(0).astype("int8")
    out["teacher_rows"] = df["teacher_rows"].fillna(0).astype("int16")

    # G. Token / card
    tok_created = pd.to_datetime(df["token_created_at"], errors="coerce")
    token_age = (ts - tok_created).dt.total_seconds() / 86400.0
    out["has_user_token"] = (df["user_token_id"].fillna(0) > 0).astype("int8")
    out["token_age_days"] = token_age.fillna(-1).clip(lower=-1, upper=3650)
    out["token_is_default"] = df["token_is_default"].fillna(False).astype("int8")
    out["token_has_last4"] = df["token_has_last4"].fillna(False).astype("int8")
    out["token_gateway_mismatch"] = (
        df["gateway"].astype("string").fillna("") !=
        df["token_gateway"].astype("string").fillna("")
    ).astype("int8")

    # H. Cupon (via payment_discounts)
    out["has_coupon"] = df["coupon_discount_type"].notna().astype("int8")
    out["coupon_discount_type_code"] = pd.Categorical(
        df["coupon_discount_type"].fillna("none")
    ).codes.astype("int16")

    out["paid_by_manager"] = df["paid_by_manager"].fillna(False).astype("int8")

    # Categoricals for frequency encoding (computed in main)
    for col in ["source_enum", "payment_method", "gateway", "card_brand",
                "currency", "category", "payment_source", "facility_id"]:
        out[f"_cat_{col}"] = df[col].astype("string").fillna("UNKNOWN")

    return out


CAT_COLS = ["source_enum", "payment_method", "gateway", "card_brand",
            "currency", "category", "payment_source", "facility_id"]


def fit_frequency(X_train: pd.DataFrame) -> dict:
    mapping = {}
    for col in CAT_COLS:
        vc = X_train[f"_cat_{col}"].value_counts(normalize=True)
        mapping[col] = vc.to_dict()
    return mapping


def apply_frequency(X: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    X = X.copy()
    for col in CAT_COLS:
        m = mapping[col]
        X[f"freq_{col}"] = X[f"_cat_{col}"].map(m).astype("float64").fillna(0.0)
        X.drop(columns=[f"_cat_{col}"], inplace=True)
    return X


def target(df: pd.DataFrame) -> np.ndarray:
    return df["status"].isin(["totally_refunded", "refunded_to_credit"]).astype("int8").values


def simple_rule_scores(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(df["created_at"])
    res_date = pd.to_datetime(df["res_date"], errors="coerce")
    lead = (res_date - ts.dt.normalize()).dt.days.astype("float64")
    score = np.zeros(len(df), dtype="float64")
    score += df["source_enum"].isin(["pbp_app", "white_label_app"]).astype("float64")
    score += df["payment_method"].isin(["prepaid", "user_package"]).astype("float64")
    score += (lead >= 8).fillna(False).astype("float64")
    score += (~df["admin_booked"].fillna(False).astype(bool)).astype("float64")
    return score


def metrics_block(name: str, y, scores) -> dict:
    auc = float(roc_auc_score(y, scores))
    ap = float(average_precision_score(y, scores))
    base = float(y.mean())
    out = {"split": name, "n": int(len(y)), "positives": int(y.sum()),
           "base_rate": base, "auc_roc": auc, "ap": ap, "ap_over_base": ap / base if base > 0 else None}
    for k_pct in [0.1, 0.2, 0.5, 1, 2, 5]:
        k = max(1, int(len(y) * k_pct / 100))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        precision = float(y[top_idx].mean())
        recall = float(y[top_idx].sum() / max(1, y.sum()))
        out[f"precision_at_{k_pct}pct"] = precision
        out[f"recall_at_{k_pct}pct"] = recall
    print(f"  [{name}] n={out['n']:,} pos={out['positives']:,} base={base*100:.3f}% "
          f"AUC={auc:.4f} AP={ap:.4f} AP/base={out['ap_over_base']:.2f} "
          f"P@1%={out['precision_at_1pct']*100:.1f}%")
    return out


def monthly_metrics(df: pd.DataFrame, y, scores, prefix: str) -> list[dict]:
    months = pd.to_datetime(df["created_at"]).dt.to_period("M").astype(str)
    out = []
    for month in sorted(months.unique()):
        mask = months == month
        if mask.sum() == 0:
            continue
        out.append(metrics_block(f"{prefix}_{month}", y[mask.values], scores[mask.values]))
    return out


def simple_rule_payload(raw_splits: dict[str, pd.DataFrame]) -> dict:
    payload = {
        "variant": "SIMPLE-RULE",
        "seed": SEED,
        "definition": {
            "source_app": "source_enum IN ('pbp_app', 'white_label_app')",
            "prepaid_or_package": "payment_method IN ('prepaid', 'user_package')",
            "lead_ge_8_days": "reservation_lead_days >= 8",
            "not_admin_booked": "admin_booked = false",
        },
        "metrics": [],
        "monthly_metrics": [],
    }
    for split, raw in raw_splits.items():
        y = target(raw)
        scores = simple_rule_scores(raw)
        payload["metrics"].append(metrics_block(split, y, scores))
        if split == "test_sep_dec_legacy":
            is_octdec = pd.to_datetime(raw["created_at"]) >= pd.Timestamp("2025-10-01")
            payload["metrics"].append(
                metrics_block("test_oct_dec_final", y[is_octdec.values], scores[is_octdec.values])
            )
            is_sep = ~is_octdec
            payload["metrics"].append(
                metrics_block("discovery_sep", y[is_sep.values], scores[is_sep.values])
            )
            payload["monthly_metrics"].extend(monthly_metrics(raw, y, scores, "simple_rule_month"))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4 HGB Gate A0 benchmark.")
    parser.add_argument(
        "--variant",
        choices=["clean", "clean-strict", "clean-no-ru"],
        default="clean-strict",
        help="Participant feature policy. clean uses current-state RU and is exploratory only.",
    )
    parser.add_argument(
        "--baseline",
        choices=["none", "simple-rule"],
        default="simple-rule",
        help="Optional manual baseline to write alongside the HGB benchmark.",
    )
    parser.add_argument(
        "--period",
        choices=["legacy_sep_dec", "final_oct_dec"],
        default="final_oct_dec",
        help="Reporting period hint stored in metadata; extraction always keeps Sep-Dec for comparability.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    slug = variant_slug(args.variant)
    sql = query_for_variant(args.variant)

    print("=" * 80)
    print(f"V4-CLEAN PRE-BENCHMARK (Gate A0) - {args.variant}")
    print("=" * 80)

    client = ch_client()

    t0 = time.time()
    train_raw = extract(client, "2025-01-01", "2025-07-01", "train_jan_jun", sql)
    val_raw   = extract(client, "2025-07-01", "2025-09-01", "val_jul_aug", sql)
    test_raw  = extract(client, "2025-09-01", "2026-01-01", "test_sep_dec", sql)

    print(f"extracted in {time.time()-t0:.1f}s - total rows {len(train_raw)+len(val_raw)+len(test_raw):,}")

    print("\nbuilding features...")
    X_train = build_features(train_raw)
    X_val   = build_features(val_raw)
    X_test  = build_features(test_raw)

    y_train = target(train_raw)
    y_val   = target(val_raw)
    y_test  = target(test_raw)

    print(f"  train base rate {y_train.mean()*100:.3f}%")
    print(f"  val   base rate {y_val.mean()*100:.3f}%")
    print(f"  test  base rate {y_test.mean()*100:.3f}%")

    print("\nfitting frequency encoder on train...")
    freq_map = fit_frequency(X_train)

    X_train = apply_frequency(X_train, freq_map)
    X_val   = apply_frequency(X_val, freq_map)
    X_test  = apply_frequency(X_test, freq_map)

    feature_list = X_train.columns.tolist()
    print(f"feature count: {len(feature_list)}")

    if args.baseline == "simple-rule":
        print("\nevaluating SIMPLE-RULE baseline...")
        simple_payload = simple_rule_payload({
            "train_jan_jun": train_raw,
            "val_jul_aug": val_raw,
            "test_sep_dec_legacy": test_raw,
        })
        (OUT / "simple_rule_baseline.json").write_text(
            json.dumps(simple_payload, indent=2, default=str)
        )
        print(f"  {OUT}/simple_rule_baseline.json")

    print(f"\ntraining HistGradientBoostingClassifier ({args.variant})...")
    t1 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        min_samples_leaf=50,
        random_state=SEED,
    )
    model.fit(X_train.values.astype("float32"), y_train)
    print(f"  fitted in {time.time()-t1:.1f}s")

    s_train = model.predict_proba(X_train.values.astype("float32"))[:, 1]
    s_val   = model.predict_proba(X_val.values.astype("float32"))[:, 1]
    s_test  = model.predict_proba(X_test.values.astype("float32"))[:, 1]

    print("\nmetrics:")
    train_m = metrics_block("train_jan_jun", y_train, s_train)
    val_m   = metrics_block("val_jul_aug", y_val, s_val)
    test_legacy = metrics_block("test_sep_dec_legacy", y_test, s_test)

    is_octdec = pd.to_datetime(test_raw["created_at"]) >= pd.Timestamp("2025-10-01")
    test_final = metrics_block(
        "test_oct_dec_final",
        y_test[is_octdec.values],
        s_test[is_octdec.values],
    )

    is_sep = ~is_octdec
    test_disc = metrics_block(
        "discovery_sep",
        y_test[is_sep.values],
        s_test[is_sep.values],
    )

    payload = {
        "variant": args.variant,
        "model": "HistGradientBoostingClassifier",
        "params": model.get_params(),
        "seed": SEED,
        "period_hint": args.period,
        "participant_feature_policy": args.variant,
        "amount_policy": "nominal_amount_benchmark; final pipeline must use USD-normalized features",
        "feature_count": len(feature_list),
        "metrics": [train_m, val_m, test_disc, test_final, test_legacy],
        "monthly_metrics": monthly_metrics(test_raw, y_test, s_test, "hgb_month"),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    metrics_path = OUT / f"baseline_hgb_{slug}.json"
    features_path = OUT / f"feature_list_{slug}.json"
    scores_path = OUT / f"baseline_hgb_{slug}_scores.parquet"

    metrics_path.write_text(json.dumps(payload, indent=2, default=str))
    features_path.write_text(json.dumps(feature_list, indent=2))

    scores_df = pd.concat([
        pd.DataFrame({"split": "val_jul_aug",
                      "payment_id": val_raw["payment_id"].values,
                      "score": s_val, "y": y_val}),
        pd.DataFrame({"split": "test_sep_dec",
                      "payment_id": test_raw["payment_id"].values,
                      "score": s_test, "y": y_test}),
    ], ignore_index=True)
    scores_df.to_parquet(scores_path, index=False)

    print("\nartifacts written:")
    print(f"  {metrics_path}")
    print(f"  {features_path}")
    print(f"  {scores_path}")

    print("\n" + "=" * 80)
    print("GATE A0 SUMMARY")
    print("=" * 80)
    print(f"{args.variant} val AUC      : {val_m['auc_roc']:.4f}")
    print(f"{args.variant} test Sep-Dic : {test_legacy['auc_roc']:.4f}")
    print(f"{args.variant} test Oct-Dic : {test_final['auc_roc']:.4f}   (gate >= 0.70)")
    print(f"{args.variant} P@1% Sep-Dic : {test_legacy['precision_at_1pct']*100:.1f}%")


if __name__ == "__main__":
    main()
