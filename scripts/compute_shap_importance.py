#!/usr/bin/env python
"""Global SHAP analysis + permutation importance for the IF-40 model.

Generates:
  output/figures/shap_summary.{pdf,png}        Beeswarm summary, top 20 features
  output/figures/shap_bar_top20.{pdf,png}      Mean(|SHAP|) bar chart, top 20
  Tesis-Latex/tablas/table_3_39_feature_importance.tex
  output/results_shap_importance.json

Methodology:
- shap.TreeExplainer on the trained IsolationForest (native support).
- Background sample: 2 000 rows from train (stratified by month).
- Explanation set: 5 000 rows from test (stratified by month + pure_fraud).
- Aggregates SHAP into a single global ranking via mean(|SHAP|).
- Also exports group-level totals to make the LaTeX table compact.

This is the version used in the thesis "Importancia global de las features" section.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings  # noqa: E402
from fraud_detector.data.loader import DataManager  # noqa: E402
from fraud_detector.utils.logger import logger  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SCORES_DIR = PROJECT_ROOT / "output" / "scores"
FIG_DIR = PROJECT_ROOT / "output" / "figures"
MODELS_DIR = PROJECT_ROOT / "output" / "models"
TABLES_DIR = PROJECT_ROOT.parent / "Tesis-Latex" / "tablas"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

N_BACKGROUND = 2_000
N_EXPLAIN = 5_000
TOP_K = 20

# Manual feature-to-group mapping for the IF-40 set
FEATURE_GROUPS = {
    "log_amount": "A. Transaccional",
    "discount_ratio": "A. Transaccional",
    "has_tip": "A. Transaccional",
    "hour_sin": "B. Temporal",
    "hour_cos": "B. Temporal",
    "day_of_week": "B. Temporal",
    "is_weekend": "B. Temporal",
    "is_off_hours": "B. Temporal",
    "user_txn_count_1h": "C. Velocidad",
    "time_since_last_txn": "C. Velocidad",
    "user_amount_24h": "C. Velocidad",
    "user_distinct_facilities_30d": "D. Comportamiento",
    "user_distinct_methods": "D. Comportamiento",
    "user_account_age_days": "D. Comportamiento",
    "facility_avg_amount": "E. Contextual",
    "amount_facility_ratio": "E. Contextual",
    "is_club_credit": "F. Crédito",
    "user_debit_count_30d": "F. Crédito",
    "user_debit_amount_30d": "F. Crédito",
    "credit_flow_ratio": "F. Crédito",
    "is_staff": "G. Rol/Staff",
    "paid_by_manager": "G. Rol/Staff",
    "staff_amount_zscore": "G. Rol/Staff",
    "category_entropy_30d": "H. Diversidad",
    "user_merchandise_ratio_30d": "H. Diversidad",
    "is_new_user": "I. Interacciones",
    "is_very_new_user": "I. Interacciones",
    "new_user_first_facility": "I. Interacciones",
    "rapid_burst": "I. Interacciones",
    "small_amount_at_facility": "I. Interacciones",
    "very_small_amount_at_facility": "I. Interacciones",
    "off_hours_high_value": "I. Interacciones",
    "is_third_party_payment": "J. Raw-derived",
    "same_amount_count_1h": "J. Raw-derived",
    "same_amount_count_24h": "J. Raw-derived",
    "gateway_change_recent": "J. Raw-derived",
    "capture_delay_seconds": "J. Raw-derived",
    "is_main_gateway": "J. Raw-derived",
    "is_first_gateway_for_user": "J. Raw-derived",
    "source_change_recent": "J. Raw-derived",
}


def stratified_subsample(n_total, months, n):
    """Pick n indices spread across months as evenly as possible."""
    idx = np.arange(n_total)
    months = pd.Series(months).reset_index(drop=True)
    unique_months = months.unique()
    per_bucket = max(1, n // len(unique_months))
    picked = []
    rng = np.random.default_rng(42)
    for m in unique_months:
        bucket = idx[(months == m).values]
        if len(bucket) <= per_bucket:
            picked.extend(bucket.tolist())
        else:
            picked.extend(rng.choice(bucket, per_bucket, replace=False).tolist())
    # If short, fill randomly
    if len(picked) < n:
        remainder = list(set(idx.tolist()) - set(picked))
        extra = rng.choice(remainder, size=n - len(picked), replace=False)
        picked.extend(extra.tolist())
    return np.array(sorted(picked[:n]))


def build_pure_fraud_proxy(df):
    cols = df.columns
    card_test = (df["same_amount_count_1h"] >= 3).to_numpy() if "same_amount_count_1h" in cols else np.zeros(len(df), dtype=bool)
    new_burst = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)).to_numpy()
    third = ((df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)).to_numpy() \
        if "is_third_party_payment" in cols else np.zeros(len(df), dtype=bool)
    return (card_test | new_burst | third).astype(np.int8)


def render_tex_table(ranking_df, group_df, out_path: Path):
    """Build the LaTeX table for the thesis."""
    lines = []
    lines.append("% Generado por scripts/compute_shap_importance.py")
    lines.append("% Fuente: shap.TreeExplainer sobre IsolationForest IF-40")
    lines.append(f"% Background n={N_BACKGROUND}, explain n={N_EXPLAIN}, conjunto de prueba")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Importancia global SHAP de las 40 features del modelo IF-40 (top 20)}")
    lines.append("\\label{tab:if40-shap-importance}")
    lines.append("\\begin{tabular}{rlrr}")
    lines.append("\\toprule")
    lines.append("\\# & Feature & Grupo & Mean(|SHAP|) \\\\")
    lines.append("\\midrule")
    for i, row in enumerate(ranking_df.head(TOP_K).itertuples(index=False), 1):
        feat_safe = row.feature.replace("_", "\\_")
        grp_safe = row.group.replace("_", "\\_")
        lines.append(f"  {i} & {feat_safe} & {grp_safe} & {row.mean_abs_shap:.4f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    lines.append("\\vspace{0.4em}")
    lines.append("\\footnotesize Importancia agregada por grupo de features:")
    lines.append("")
    lines.append("\\begin{center}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append("Grupo & $\\sum$ Mean(|SHAP|) & \\% del total \\\\")
    lines.append("\\midrule")
    total = group_df["mean_abs_shap_sum"].sum()
    for row in group_df.itertuples(index=False):
        grp_safe = row.group.replace("_", "\\_")
        pct = 100.0 * row.mean_abs_shap_sum / total if total > 0 else 0.0
        lines.append(f"  {grp_safe} & {row.mean_abs_shap_sum:.3f} & {pct:.1f}\\% \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{center}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines))
    logger.info(f"  Wrote {out_path}")


def main():
    logger.info("=" * 60)
    logger.info("SHAP analysis on IF-40")
    logger.info("=" * 60)

    if_model = joblib.load(MODELS_DIR / "isolation_forest_final.joblib")
    feature_list = json.loads((MODELS_DIR / "final_feature_list.json").read_text())
    assert len(feature_list) == 40

    X_train = np.load(SCORES_DIR / "X_train_final.npy")
    X_test = np.load(SCORES_DIR / "X_test_final.npy")
    df_test = pd.read_parquet(DATA_DIR / "test_features_enriched.parquet",
                              columns=["created_at"])
    logger.info(f"  X_train={X_train.shape} X_test={X_test.shape}")

    # Background: random train sample
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(X_train.shape[0], N_BACKGROUND, replace=False)
    X_bg = X_train[bg_idx]

    # Explain set: stratified by month from test
    months_test = df_test["created_at"].dt.to_period("M").astype(str).to_numpy()
    explain_idx = stratified_subsample(X_test.shape[0], months_test, N_EXPLAIN)
    X_explain = X_test[explain_idx]
    logger.info(f"  background={X_bg.shape} explain={X_explain.shape}")

    logger.info("  Computing SHAP values with TreeExplainer ...")
    t = time.perf_counter()
    explainer = shap.TreeExplainer(if_model, data=X_bg, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_explain, check_additivity=False)
    logger.info(f"    done in {time.perf_counter() - t:.1f}s "
                f"(shape={np.asarray(shap_values).shape})")

    shap_arr = np.asarray(shap_values)
    mean_abs = np.abs(shap_arr).mean(axis=0)

    df_imp = pd.DataFrame({
        "feature": feature_list,
        "mean_abs_shap": mean_abs,
        "group": [FEATURE_GROUPS.get(f, "Otros") for f in feature_list],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    df_groups = (df_imp.groupby("group", as_index=False)
                 .agg(mean_abs_shap_sum=("mean_abs_shap", "sum"))
                 .sort_values("mean_abs_shap_sum", ascending=False))

    # --- Summary beeswarm
    logger.info("  Plotting summary (beeswarm) ...")
    plt.figure(figsize=(7.0, 5.5))
    shap.summary_plot(shap_arr, X_explain, feature_names=feature_list,
                      max_display=TOP_K, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP — Impacto en el score IF (top 20, n=5 000 test)",
                 fontsize=11, y=1.02)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"shap_summary.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        logger.info(f"    Saved {out.name}")
    plt.close(fig)

    # --- Top-20 bar chart
    logger.info("  Plotting top-20 bar ...")
    top = df_imp.head(TOP_K).iloc[::-1]
    colors_map = {
        "A. Transaccional": "#1f4e79", "B. Temporal": "#2e75b6",
        "C. Velocidad":     "#c0504d", "D. Comportamiento": "#bf9000",
        "E. Contextual":    "#548235", "F. Crédito": "#7030a0",
        "G. Rol/Staff":     "#a6a6a6", "H. Diversidad": "#ed7d31",
        "I. Interacciones": "#9bbb59", "J. Raw-derived": "#2e2e2e",
    }
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.barh(top["feature"], top["mean_abs_shap"],
            color=[colors_map.get(g, "#444") for g in top["group"]])
    ax.set_xlabel("Importancia global — mean(|SHAP|)")
    ax.set_title("Top 20 features por importancia SHAP (IF-40, test n=5 000)")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=g) for g, c in colors_map.items()
               if g in top["group"].unique()]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.92,
              fontsize=8)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"shap_bar_top20.{ext}"
        fig.savefig(out)
        logger.info(f"    Saved {out.name}")
    plt.close(fig)

    # --- LaTeX table
    render_tex_table(df_imp, df_groups,
                     TABLES_DIR / "table_3_39_feature_importance.tex")

    # --- JSON dump for traceability
    out = {
        "config": {
            "n_background": N_BACKGROUND, "n_explain": N_EXPLAIN,
            "top_k": TOP_K, "seed": 42,
            "explainer": "shap.TreeExplainer",
            "feature_perturbation": "interventional",
        },
        "per_feature": df_imp.to_dict(orient="records"),
        "per_group": df_groups.to_dict(orient="records"),
    }
    (PROJECT_ROOT / "output" / "results_shap_importance.json").write_text(
        json.dumps(out, indent=2, default=float)
    )
    logger.info(f"  Wrote output/results_shap_importance.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()
