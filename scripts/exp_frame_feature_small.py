"""
Small-sample experiment: can reference-frame fixes improve ML ranking?

This is an exploratory train -> validation test. It does not touch the final
test set and does not replace the confirmatory thesis metrics.

Variants:
  - clean_29_current: current non-leaky FS-clean-A-29 feature set.
  - no_absolute_amounts: removes absolute amount/scale features.
  - frame_magnitude_v1: relative/robust facility magnitude features.
  - frame_magnitude_plus_cyclic_dow: same as v1 with cyclic day-of-week.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


SEEDS = [42, 52, 62]
TRAIN_N = 120_000
VAL_N = 60_000
OUT = Path("output/revision/frame_feature_small_experiment.json")
PROXY_POSITIVE = {"totally_refunded", "refunded_to_credit"}
COHORTS = {
    "all_currencies": None,
    "usd_only": "USD",
}

FS_CLEAN_A_29 = [
    "amount",
    "log_amount",
    "amount_usd_ratio",
    "discount_ratio",
    "has_tip",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "is_off_hours",
    "user_txn_count_1h",
    "user_txn_count_24h",
    "time_since_last_txn",
    "user_amount_24h",
    "user_distinct_facilities_30d",
    "user_distinct_methods",
    "user_account_age_days",
    "user_discount_ratio_30d",
    "facility_avg_amount",
    "amount_facility_ratio",
    "is_club_credit",
    "user_debit_count_30d",
    "user_debit_amount_30d",
    "credit_flow_ratio",
    "is_staff",
    "paid_by_manager",
    "staff_amount_zscore",
    "category_entropy_30d",
    "user_merchandise_ratio_30d",
]

ABSOLUTE_MAGNITUDE_FEATURES = {
    "amount",
    "log_amount",
    "amount_usd_ratio",
    "facility_avg_amount",
    "user_amount_24h",
    "user_debit_amount_30d",
}


def deterministic_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).sort_values("created_at").reset_index(drop=True)


def load_split(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path("data/processed") / f"{name}_features.parquet", columns=columns)


def add_frame_features(train: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    val = val.copy()

    grouped = train.groupby("facility_id")["amount"]
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    stats = pd.DataFrame(
        {
            "facility_amount_median": grouped.median(),
            "facility_amount_iqr": iqr.fillna(iqr.median()),
        }
    )
    global_median = float(train["amount"].median())
    global_iqr = float((train["amount"].quantile(0.75) - train["amount"].quantile(0.25)) or 1.0)
    if not np.isfinite(global_iqr) or global_iqr <= 0:
        global_iqr = 1.0

    for df in (train, val):
        df["facility_amount_median"] = (
            df["facility_id"].map(stats["facility_amount_median"]).fillna(global_median)
        )
        df["facility_amount_iqr"] = (
            df["facility_id"].map(stats["facility_amount_iqr"]).fillna(global_iqr).clip(lower=1e-6)
        )
        df["amount_facility_z"] = (
            (df["amount"] - df["facility_amount_median"]) / df["facility_amount_iqr"]
        ).astype(np.float32)
        df["user_amount_24h_facility_ratio"] = (
            df["user_amount_24h"] / (df["facility_avg_amount"] + 0.01)
        ).astype(np.float32)
        df["user_debit_amount_30d_facility_ratio"] = (
            df["user_debit_amount_30d"] / (df["facility_avg_amount"] + 0.01)
        ).astype(np.float32)

        dow_zero_based = (df["day_of_week"].astype(float) - 1.0).clip(lower=0, upper=6)
        df["day_of_week_sin"] = np.sin(2 * np.pi * dow_zero_based / 7).astype(np.float32)
        df["day_of_week_cos"] = np.cos(2 * np.pi * dow_zero_based / 7).astype(np.float32)

    return train, val


def feature_sets() -> dict[str, list[str]]:
    no_abs = [f for f in FS_CLEAN_A_29 if f not in ABSOLUTE_MAGNITUDE_FEATURES]
    frame_v1 = no_abs + [
        "amount_facility_z",
        "user_amount_24h_facility_ratio",
        "user_debit_amount_30d_facility_ratio",
    ]
    frame_plus_cyclic = [
        f for f in frame_v1 if f != "day_of_week"
    ] + ["day_of_week_sin", "day_of_week_cos"]
    return {
        "clean_29_current": FS_CLEAN_A_29,
        "no_absolute_amounts": no_abs,
        "frame_magnitude_v1": frame_v1,
        "frame_magnitude_plus_cyclic_dow": frame_plus_cyclic,
    }


def evaluate(y: np.ndarray, scores: np.ndarray, amount: np.ndarray, off_hours: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, scores))
    ap = float(average_precision_score(y, scores))
    k = max(1, int(np.ceil(len(scores) * 0.05)))
    order = np.argsort(scores)[-k:]
    precision = float(y[order].mean())
    base_rate = float(y.mean())
    corr = spearmanr(scores, amount).statistic
    return {
        "auc_roc": auc,
        "auc_pr": ap,
        "precision_at_5pct": precision,
        "ef_at_5pct": precision / base_rate if base_rate else 0.0,
        "spearman_score_vs_amount": float(corr) if np.isfinite(corr) else None,
        "top5_amount_x_avg": float(amount[order].mean() / max(amount.mean(), 1e-9)),
        "top5_offhours_rate_utc": float(off_hours[order].mean()),
    }


def fit_score(train: pd.DataFrame, val: pd.DataFrame, features: list[str], seed: int) -> dict:
    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    X_val = val[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    model = IsolationForest(
        n_estimators=100,
        max_samples=512,
        max_features=1.0,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train_s)
    scores = -model.score_samples(X_val_s)
    y = val["status"].isin(PROXY_POSITIVE).astype(np.int8).to_numpy()
    amount = val["amount"].astype(float).to_numpy()
    off_hours = val["is_off_hours"].astype(float).to_numpy()
    return evaluate(y, scores, amount, off_hours)


def summarize(rows: list[dict]) -> dict:
    metrics = [
        "auc_roc",
        "auc_pr",
        "precision_at_5pct",
        "ef_at_5pct",
        "spearman_score_vs_amount",
        "top5_amount_x_avg",
        "top5_offhours_rate_utc",
    ]
    out = {}
    for metric in metrics:
        values = np.array([r[metric] for r in rows if r[metric] is not None], dtype=float)
        out[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def run_cohort(train_full: pd.DataFrame, val_full: pd.DataFrame, cohort_name: str, currency: str | None) -> dict:
    if currency is None:
        train_base = train_full
        val_base = val_full
    else:
        train_base = train_full[train_full["currency"] == currency]
        val_base = val_full[val_full["currency"] == currency]

    train = deterministic_sample(train_base, TRAIN_N, seed=20260703)
    val = deterministic_sample(val_base, VAL_N, seed=20260703)
    train, val = add_frame_features(train, val)

    raw_rows = []
    for variant, features in feature_sets().items():
        for seed in SEEDS:
            metrics = fit_score(train, val, features, seed)
            raw_rows.append(
                {
                    "cohort": cohort_name,
                    "variant": variant,
                    "seed": seed,
                    "n_features": len(features),
                    **metrics,
                }
            )

    return {
        "currency_filter": currency,
        "train_n": int(len(train)),
        "val_n": int(len(val)),
        "base_rate_val_tipo_a": float(val["status"].isin(PROXY_POSITIVE).mean()),
        "variants": {
            variant: summarize([r for r in raw_rows if r["variant"] == variant])
            for variant in feature_sets()
        },
        "raw_rows": raw_rows,
    }


def main() -> None:
    required = sorted(set(FS_CLEAN_A_29 + ["id", "created_at", "status", "facility_id", "currency"]))
    train_full = load_split("train", columns=required)
    val_full = load_split("val", columns=required)

    cohorts = {
        name: run_cohort(train_full, val_full, name, currency)
        for name, currency in COHORTS.items()
    }
    result = {
        "experiment": "frame_feature_small",
        "date": "2026-07-03",
        "split": "train_sample_to_val_sample",
        "train_n_max_per_cohort": TRAIN_N,
        "val_n_max_per_cohort": VAL_N,
        "seeds": SEEDS,
        "model": {
            "class": "IsolationForest",
            "n_estimators": 100,
            "max_samples": 512,
            "max_features": 1.0,
            "contamination": "auto",
            "scaler": "StandardScaler fit on train sample only",
        },
        "cohorts": cohorts,
        "feature_sets": feature_sets(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["experiment", "train_n_max_per_cohort", "val_n_max_per_cohort"]}, indent=2))
    for cohort_name, cohort in cohorts.items():
        print(f"\n[{cohort_name}] train={cohort['train_n']:,} val={cohort['val_n']:,} base={cohort['base_rate_val_tipo_a']:.4f}")
        for variant, metrics in cohort["variants"].items():
            auc = metrics["auc_roc"]["mean"]
            ef5 = metrics["ef_at_5pct"]["mean"]
            top_amt = metrics["top5_amount_x_avg"]["mean"]
            print(f"{variant:32s} AUC={auc:.4f} EF5={ef5:.3f} top5_amt_x={top_amt:.2f}")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
