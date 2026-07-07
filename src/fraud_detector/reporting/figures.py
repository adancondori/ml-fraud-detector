"""Figure generator for thesis Cap 3. PDF + PNG, Spanish labels, serif font."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from fraud_detector.utils.logger import logger

MODEL_NAMES = {"isolation_forest": "Isolation Forest", "lof": "LOF", "ocsvm": "OC-SVM"}
MODEL_COLORS = {"isolation_forest": "#1f77b4", "lof": "#ff7f0e", "ocsvm": "#2ca02c"}


def _setup_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def _save(fig, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {name}.pdf + .png")


def fig_roc_curves(scores_dict: Dict[str, np.ndarray], proxy: np.ndarray, output_dir: Path):
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    for model, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(proxy, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS.get(model, "gray"),
            label=f"{MODEL_NAMES.get(model, model)} (AUC={roc_auc:.4f})",
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatorio (0.50)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC")
    ax.legend(loc="lower right")
    _save(fig, output_dir, "roc_curves")


def fig_pr_curves(scores_dict: Dict[str, np.ndarray], proxy: np.ndarray, output_dir: Path):
    _setup_style()
    base_rate = proxy.mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    for model, scores in scores_dict.items():
        prec, rec, _ = precision_recall_curve(proxy, scores)
        ap = average_precision_score(proxy, scores)
        ax.plot(
            rec,
            prec,
            color=MODEL_COLORS.get(model, "gray"),
            label=f"{MODEL_NAMES.get(model, model)} (AP={ap:.4f})",
        )
    ax.axhline(
        y=base_rate, color="gray", linestyle="--", alpha=0.5, label=f"Tasa base ({base_rate:.4f})"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall")
    ax.legend(loc="upper right")
    _save(fig, output_dir, "pr_curves")


def fig_score_distributions(
    scores_dict: Dict[str, np.ndarray], proxy: np.ndarray, output_dir: Path
):
    _setup_style()
    models = list(scores_dict.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        s = scores_dict[model]
        # Clip extreme outliers for visualization
        p1, p99 = np.percentile(s, [1, 99])
        bins = np.linspace(p1, p99, 80)
        ax.hist(s[proxy == 0], bins=bins, alpha=0.5, color="blue", label="Normal", density=True)
        ax.hist(s[proxy == 1], bins=bins, alpha=0.5, color="orange", label="Proxy+", density=True)
        ax.set_title(MODEL_NAMES.get(model, model))
        ax.set_xlabel("Score de anomalia")
        ax.set_ylabel("Densidad")
        ax.legend()
    fig.suptitle("Distribucion de scores: proxy+ vs proxy-", y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, "score_distributions")


def fig_enrichment_curve(scores_dict: Dict[str, np.ndarray], proxy: np.ndarray, output_dir: Path):
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    k_values = np.arange(0.005, 1.005, 0.005)
    base_rate = proxy.mean()
    for model, scores in scores_dict.items():
        efs = []
        sorted_idx = np.argsort(scores)[::-1]
        sorted_proxy = proxy[sorted_idx]
        cumsum = np.cumsum(sorted_proxy)
        n = len(proxy)
        for k in k_values:
            top_n = max(1, int(k * n))
            prec_at_k = cumsum[top_n - 1] / top_n
            efs.append(prec_at_k / base_rate if base_rate > 0 else 0)
        ax.plot(
            k_values * 100,
            efs,
            color=MODEL_COLORS.get(model, "gray"),
            label=MODEL_NAMES.get(model, model),
        )
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="EF = 1 (aleatorio)")
    ax.set_xlabel("% del dataset revisado")
    ax.set_ylabel("Factor de enriquecimiento")
    ax.set_title("Curva de enriquecimiento")
    ax.legend()
    ax.set_xlim(0, 50)
    _save(fig, output_dir, "enrichment_curve")


def fig_temporal_stability(results: Dict, output_dir: Path):
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    months = ["2025-09", "2025-10", "2025-11", "2025-12"]
    month_labels = ["Sep", "Oct", "Nov", "Dic"]
    for model in ["isolation_forest", "lof", "ocsvm"]:
        ts = results.get(model, {}).get("temporal_stability", {}).get("monthly_auc", {})
        aucs = [ts.get(m, {}).get("auc_roc", np.nan) for m in months]
        ax.plot(
            month_labels,
            aucs,
            "o-",
            color=MODEL_COLORS.get(model, "gray"),
            label=MODEL_NAMES.get(model, model),
        )
    ax.set_xlabel("Mes (2025)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Estabilidad temporal")
    ax.legend()
    _save(fig, output_dir, "temporal_stability")


def fig_grid_search_heatmap(grid_df, output_dir: Path):
    _setup_style()
    import pandas as pd

    # Aggregate across contamination (invariant to ranking)
    agg = grid_df.groupby(["n_estimators", "max_samples"])["auc_roc"].mean().reset_index()
    pivot = agg.pivot(index="max_samples", columns="n_estimators", values="auc_roc")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_samples")
    ax.set_title("Grid Search IF: AUC-ROC medio")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="AUC-ROC")
    _save(fig, output_dir, "grid_search_heatmap")


def fig_anomaly_type_distribution(sens: Dict, output_dir: Path):
    _setup_style()
    typo = sens.get("anomaly_typology", {}).get("type_distribution", {})
    types = sorted(typo.keys(), key=lambda t: typo[t].get("count", 0), reverse=True)
    counts = [typo[t].get("count", 0) for t in types]
    pcts = [typo[t].get("pct", 0) for t in types]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(types, counts, color="#4c72b0")
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Cantidad de transacciones")
    ax.set_title("Tipologia de anomalias (top-5%, SHAP)")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, output_dir, "anomaly_type_distribution")


def fig_posthoc_facility(posthoc: Dict, output_dir: Path):
    _setup_style()
    facs = (
        posthoc.get("posthoc_analysis", {})
        .get("facility_concentration", {})
        .get("top_10_facilities", [])
    )
    if not facs:
        return
    labels = [str(f.get("facility_id", "")) for f in facs[:15]]
    rates = [f.get("anomaly_rate", 0) for f in facs[:15]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, rates, color="#c44e52")
    ax.axvline(x=0.05, color="gray", linestyle="--", alpha=0.7, label="Tasa base (5%)")
    ax.set_xlabel("Tasa de anomalias")
    ax.set_title("Top centros por concentracion de anomalias")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, output_dir, "posthoc_facility_anomaly_rate")


def fig_posthoc_currency(posthoc: Dict, output_dir: Path):
    _setup_style()
    currs = (
        posthoc.get("posthoc_analysis", {})
        .get("currency_concentration", {})
        .get("currencies_affected", [])
    )
    if not currs:
        return
    # Top 10 by n_transactions
    currs_sorted = sorted(currs, key=lambda x: x.get("n_transactions", 0), reverse=True)[:10]
    labels = [c.get("currency", "") for c in currs_sorted]
    rates = [c.get("anomaly_rate", 0) for c in currs_sorted]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, rates, color="#55a868")
    ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.7, label="Tasa base (5%)")
    ax.set_ylabel("Tasa de anomalias")
    ax.set_title("Tasa de anomalias por moneda")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "posthoc_currency_distribution")
