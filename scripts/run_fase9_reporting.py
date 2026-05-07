#!/usr/bin/env python
"""Fase 9: Reporting — Generate all LaTeX tables and figures for Cap 3.

Reads results.json, results_sensitivity.json, results_posthoc.json
and generates ~20 tables + ~11 figures.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings
from fraud_detector.reporting import figures as fig_gen
from fraud_detector.reporting import latex_tables as tbl
from fraud_detector.utils.logger import logger

TOTAL_STEPS = 4


def log_step(step: int, name: str, status: str = "START"):
    logger.info(f"[STEP {step}/{TOTAL_STEPS}] {name} — {status}")


def load_all_results():
    """Load all JSON results from Fases 7 and 8."""
    results = json.loads((settings.project_root / "output" / "results.json").read_text())
    sensitivity = json.loads((settings.project_root / "output" / "results_sensitivity.json").read_text())
    posthoc = json.loads((settings.project_root / "output" / "results_posthoc.json").read_text())
    return results, sensitivity, posthoc


def step1_tables(results, sensitivity, posthoc):
    """Generate all LaTeX tables for Cap 3."""
    log_step(1, "LaTeX Tables", "START")
    t0 = time.perf_counter()
    tables_dir = settings.tables_dir
    count = 0

    # HE1-HE4
    tbl.table_model_comparison(results, tables_dir / "table_3_10_model_comparison.tex")
    tbl.table_he1_results(results, tables_dir / "table_3_11_he1_results.tex")
    tbl.table_he2_results(results, tables_dir / "table_3_12_he2_results.tex")
    tbl.table_he3_results(results, tables_dir / "table_3_13_he3_results.tex")
    tbl.table_he4_comparison(results, tables_dir / "table_3_14_he4_comparison.tex")
    tbl.table_bootstrap_ci(results, tables_dir / "table_3_15_bootstrap_ci.tex")
    count += 6

    # Sensitivity
    tbl.table_sensitivity_proxy(sensitivity, tables_dir / "table_3_16_sensitivity_proxy.tex")
    tbl.table_sensitivity_per_type(sensitivity, tables_dir / "table_3_17_sensitivity_per_type.tex")
    tbl.table_sensitivity_feature18(sensitivity, tables_dir / "table_3_18_sensitivity_feature18.tex")
    tbl.table_ablation_31vs21(sensitivity, tables_dir / "table_3_19_ablation_31vs21.tex")
    count += 4

    # Temporal
    tbl.table_temporal_stability(results, tables_dir / "table_3_20_temporal_stability.tex")
    count += 1

    # Summary
    tbl.table_hypothesis_summary(results, tables_dir / "table_3_21_hypothesis_summary.tex")
    count += 1

    # Segments
    tbl.table_metrics_by_segment(
        sensitivity, "by_role",
        "M\\'{e}tricas por rol de usuario", "tab:metrics-by-role",
        tables_dir / "table_3_22_metrics_by_role.tex",
    )
    tbl.table_metrics_by_segment(
        sensitivity, "by_category",
        "M\\'{e}tricas por categor\\'{i}a de pago", "tab:metrics-by-category",
        tables_dir / "table_3_23_metrics_by_category.tex",
    )
    count += 2

    # Typology + User profiles
    tbl.table_anomaly_types(sensitivity, tables_dir / "table_3_24_anomaly_types.tex")
    tbl.table_user_risk_profile(sensitivity, tables_dir / "table_3_25_user_risk_profile.tex")
    count += 2

    # Post-hoc
    tbl.table_posthoc_facility(posthoc, tables_dir / "table_3_26_posthoc_facility.tex")
    tbl.table_posthoc_manager(posthoc, tables_dir / "table_3_27_posthoc_manager.tex")
    tbl.table_posthoc_currency(posthoc, tables_dir / "table_3_28_posthoc_currency.tex")
    count += 3

    # Grid search top 10
    grid_path = settings.project_root / "output" / "grid_search_if.csv"
    if grid_path.exists():
        grid_df = pd.read_csv(grid_path).nlargest(10, "auc_roc")
        rows = []
        for _, r in grid_df.iterrows():
            rows.append(
                f"  {int(r['n_estimators'])} & {int(r['max_samples'])} & "
                f"{r['max_features']:.2f} & {r['contamination']:.2f} & {r['auc_roc']:.6f} \\\\"
            )
        body = (
            "\\begin{tabular}{rrrrr}\n\\toprule\n"
            "n\\_est & max\\_s & max\\_f & cont & AUC-ROC \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        )
        tex = (
            "\\begin{table}[htbp]\n\\centering\n\\small\n"
            "\\caption{Top 10 configuraciones del grid search IF}\n"
            "\\label{tab:grid-search-top10}\n"
            f"{body}"
            "\\end{table}\n"
        )
        (tables_dir / "table_3_08_grid_search_top10.tex").write_text(tex)
        count += 1

    log_step(1, "LaTeX Tables", f"DONE ({time.perf_counter()-t0:.1f}s) — {count} tables generated")
    return count


def step2_figures(results, sensitivity, posthoc):
    """Generate all figures for Cap 3."""
    log_step(2, "Figures", "START")
    t0 = time.perf_counter()
    fig_dir = settings.figures_dir
    count = 0

    # Load scores for figure generation
    scores_df = pd.read_parquet(settings.scores_dir / "test_scores.parquet")
    df_test = pd.read_parquet(settings.processed_dir / "test_features.parquet")
    from fraud_detector.data.loader import DataManager
    proxy = DataManager.assign_proxy_labels(df_test, "unified", settings).values

    scores_dict = {
        "isolation_forest": scores_df["score_if"].values,
        "lof": scores_df["score_lof"].values,
        "ocsvm": scores_df["score_ocsvm"].values,
    }

    # ROC curves
    fig_gen.fig_roc_curves(scores_dict, proxy, fig_dir)
    count += 1

    # PR curves
    fig_gen.fig_pr_curves(scores_dict, proxy, fig_dir)
    count += 1

    # Score distributions
    fig_gen.fig_score_distributions(scores_dict, proxy, fig_dir)
    count += 1

    # Enrichment curve
    fig_gen.fig_enrichment_curve(scores_dict, proxy, fig_dir)
    count += 1

    # Temporal stability
    fig_gen.fig_temporal_stability(results, fig_dir)
    count += 1

    # Grid search heatmap
    grid_path = settings.project_root / "output" / "grid_search_if.csv"
    if grid_path.exists():
        grid_df = pd.read_csv(grid_path)
        fig_gen.fig_grid_search_heatmap(grid_df, fig_dir)
        count += 1

    # Anomaly type distribution
    fig_gen.fig_anomaly_type_distribution(sensitivity, fig_dir)
    count += 1

    # Post-hoc facility
    fig_gen.fig_posthoc_facility(posthoc, fig_dir)
    count += 1

    # Post-hoc currency
    fig_gen.fig_posthoc_currency(posthoc, fig_dir)
    count += 1

    log_step(2, "Figures", f"DONE ({time.perf_counter()-t0:.1f}s) — {count} figures generated")
    return count


def step3_feature_tables():
    """Generate dataset summary and feature description tables."""
    log_step(3, "Feature & Dataset Tables", "START")
    t0 = time.perf_counter()
    tables_dir = settings.tables_dir
    count = 0

    # Feature statistics from train
    from fraud_detector.features.engineering import FEATURE_NAMES
    train_df = pd.read_parquet(settings.processed_dir / "train_features.parquet", columns=FEATURE_NAMES)
    stats = train_df.describe().T[["mean", "std", "min", "max"]]

    rows = []
    for feat in FEATURE_NAMES:
        r = stats.loc[feat]
        fname = feat.replace("_", "\\_")
        rows.append(f"  {fname} & {r['mean']:.4f} & {r['std']:.4f} & {r['min']:.4f} & {r['max']:.4f} \\\\")
    body = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Feature & Media & Desv. est. & M\\'{i}n & M\\'{a}x \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = (
        "\\begin{table}[htbp]\n\\centering\n\\scriptsize\n"
        "\\caption{Estad\\'{i}sticas descriptivas de las 31 features (train set)}\n"
        "\\label{tab:feature-statistics}\n"
        f"{body}"
        "\\end{table}\n"
    )
    (tables_dir / "table_3_07_feature_statistics.tex").write_text(tex)
    count += 1

    # Dataset summary
    from fraud_detector.data.loader import DataManager
    split_rows = []
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(settings.processed_dir / f"{split}_features.parquet", columns=["created_at", "status"])
        proxy = DataManager.assign_proxy_labels(df, "unified", settings)
        split_rows.append(
            f"  {split.capitalize()} & {len(df):,} & {proxy.mean()*100:.2f}\\% & "
            f"{str(df['created_at'].min())[:10]} & {str(df['created_at'].max())[:10]} \\\\"
        )
        del df
    body = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Split & N & Proxy unif. & Inicio & Fin \\\\\n\\midrule\n"
        + "\n".join(split_rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = (
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\caption{Resumen del dataset por split temporal}\n"
        "\\label{tab:dataset-summary}\n"
        f"{body}"
        "\\end{table}\n"
    )
    (tables_dir / "table_3_05_dataset_summary.tex").write_text(tex)
    count += 1

    del train_df
    log_step(3, "Feature & Dataset Tables", f"DONE ({time.perf_counter()-t0:.1f}s) — {count} tables")
    return count


def step4_correlation_matrix():
    """Generate correlation matrix figure."""
    log_step(4, "Correlation Matrix", "START")
    t0 = time.perf_counter()

    from fraud_detector.features.engineering import FEATURE_NAMES
    X_train = np.load(settings.scores_dir / "X_train.npy")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"

    corr = np.corrcoef(X_train, rowvar=False)
    short_names = [f.replace("user_", "u_").replace("_30d", "").replace("_24h", "")[:15] for f in FEATURE_NAMES]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_title("Matriz de correlacion (31 features, train set)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    fig_dir = settings.figures_dir
    fig.savefig(fig_dir / "correlation_matrix.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(fig_dir / "correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    del X_train
    log_step(4, "Correlation Matrix", f"DONE ({time.perf_counter()-t0:.1f}s)")
    return 1


def main():
    t_total = time.perf_counter()

    results, sensitivity, posthoc = load_all_results()

    n_tables = step1_tables(results, sensitivity, posthoc)
    n_figures = step2_figures(results, sensitivity, posthoc)
    n_feat = step3_feature_tables()
    n_corr = step4_correlation_matrix()

    total_tables = n_tables + n_feat
    total_figures = n_figures + n_corr
    elapsed = time.perf_counter() - t_total

    logger.info("=" * 60)
    logger.info(f"Fase 9 completada en {elapsed:.1f}s")
    logger.info(f"  Tablas LaTeX: {total_tables}")
    logger.info(f"  Figuras PDF+PNG: {total_figures}")


if __name__ == "__main__":
    main()
