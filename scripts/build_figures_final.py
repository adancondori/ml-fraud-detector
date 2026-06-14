#!/usr/bin/env python
"""Build the four canonical figures for the thesis from the FINAL IF-40 model.

Overwrites output/figures/{roc_curves,pr_curves,temporal_stability,score_distributions}.pdf
(plus .png twins) with the IF-40 results. Also saves LOF / OC-SVM scores so SHAP
and future plots do not pay the training cost again.

Inputs:
  output/scores/X_{train,val,test}_final.npy
  output/scores/if_{val,test}_scores_final.npy
  data/processed/{val,test}_features_enriched.parquet

Outputs:
  output/figures/roc_curves.{pdf,png}            IF vs LOF vs OC-SVM in pure_fraud
  output/figures/pr_curves.{pdf,png}             IF vs LOF vs OC-SVM in pure_fraud
  output/figures/temporal_stability.{pdf,png}    AUC monthly across 4 proxies
  output/figures/score_distributions.{pdf,png}   IF score KDE split by pure_fraud
  output/scores/lof_test_scores_final.npy
  output/scores/ocsvm_test_scores_final.npy
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings  # noqa: E402
from fraud_detector.data.loader import DataManager  # noqa: E402
from fraud_detector.utils.logger import logger  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SCORES_DIR = PROJECT_ROOT / "output" / "scores"
FIG_DIR = PROJECT_ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Thesis-friendly defaults (serif, vector, no top/right spines)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

COLORS = {"if": "#1f4e79", "lof": "#c0504d", "ocsvm": "#548235"}
LABELS = {"if": "Isolation Forest", "lof": "LOF", "ocsvm": "OC-SVM"}


def temporal_subsample(X, n=100_000):
    if X.shape[0] <= n:
        return X
    idx = np.linspace(0, X.shape[0] - 1, n, dtype=int)
    return X[idx]


def build_extended_proxy(df):
    tipo_a = DataManager.assign_proxy_labels(df, "tipo_a", settings).to_numpy()
    tipo_c = DataManager.assign_proxy_labels(df, "tipo_c", settings).to_numpy()
    tipo_d = DataManager.assign_proxy_labels(df, "tipo_d", settings).to_numpy()
    new_burst = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 2)).to_numpy()
    small_extreme = (df["amount_facility_ratio"] < 0.05).to_numpy()
    return (tipo_a | tipo_c | tipo_d | new_burst | small_extreme).astype(np.int8)


def build_pure_fraud_proxy(df):
    cols = df.columns
    card_test = (df["same_amount_count_1h"] >= 3).to_numpy() if "same_amount_count_1h" in cols else np.zeros(len(df), dtype=bool)
    new_burst = ((df["user_account_age_days"] < 14) & (df["user_txn_count_1h"] >= 3)).to_numpy()
    third_party_burst = (
        (df["is_third_party_payment"] == 1) & (df["user_txn_count_1h"] >= 2)
    ).to_numpy() if "is_third_party_payment" in cols else np.zeros(len(df), dtype=bool)
    return (card_test | new_burst | third_party_burst).astype(np.int8)


def get_or_train_scores(X_train, X_val, X_test):
    """Return (if_test, lof_test, ocsvm_test). Reuses cached scores when present."""
    if_path = SCORES_DIR / "if_test_scores_final.npy"
    lof_path = SCORES_DIR / "lof_test_scores_final.npy"
    oc_path = SCORES_DIR / "ocsvm_test_scores_final.npy"

    if_test = np.load(if_path)
    logger.info(f"  IF scores loaded ({len(if_test):,})")

    if lof_path.exists():
        lof_test = np.load(lof_path)
        logger.info(f"  LOF scores loaded ({len(lof_test):,})")
    else:
        logger.info("  Training LOF on 200k temporal subsample ...")
        t = time.perf_counter()
        X_lof = temporal_subsample(X_train, 200_000)
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
        lof.fit(X_lof)
        lof_test = -lof.decision_function(X_test)
        np.save(lof_path, lof_test)
        logger.info(f"    LOF done in {time.perf_counter() - t:.1f}s, scores saved")

    if oc_path.exists():
        oc_test = np.load(oc_path)
        logger.info(f"  OC-SVM scores loaded ({len(oc_test):,})")
    else:
        logger.info("  Training OC-SVM on 50k temporal subsample ...")
        t = time.perf_counter()
        X_oc = temporal_subsample(X_train, 50_000)
        oc = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        oc.fit(X_oc)
        oc_test = -oc.decision_function(X_test)
        np.save(oc_path, oc_test)
        logger.info(f"    OC-SVM done in {time.perf_counter() - t:.1f}s, scores saved")

    return if_test, lof_test, oc_test


def plot_roc(scores_by_model, y, fig_path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for name in ("if", "lof", "ocsvm"):
        fpr, tpr, _ = roc_curve(y, scores_by_model[name])
        auc = roc_auc_score(y, scores_by_model[name])
        ax.plot(fpr, tpr, color=COLORS[name], lw=1.8,
                label=f"{LABELS[name]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle=":", label="Aleatorio")
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title("Curvas ROC — IF vs. LOF vs. OC-SVM (proxy pure_fraud, test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    for ext in ("pdf", "png"):
        out = fig_path.with_suffix(f".{ext}")
        fig.savefig(out)
        logger.info(f"  Saved {out.name}")
    plt.close(fig)


def plot_pr(scores_by_model, y, fig_path):
    base = float(y.mean())
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for name in ("if", "lof", "ocsvm"):
        prec, rec, _ = precision_recall_curve(y, scores_by_model[name])
        ap = average_precision_score(y, scores_by_model[name])
        ax.plot(rec, prec, color=COLORS[name], lw=1.8,
                label=f"{LABELS[name]} (AP={ap:.3f})")
    ax.axhline(base, color="gray", lw=0.8, linestyle=":",
               label=f"Tasa base ({base * 100:.1f}%)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_title("Curvas Precisión–Recall (proxy pure_fraud, test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.5, prec.max() * 1.1))
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    for ext in ("pdf", "png"):
        out = fig_path.with_suffix(f".{ext}")
        fig.savefig(out)
        logger.info(f"  Saved {out.name}")
    plt.close(fig)


def plot_temporal_stability(temporal_dict, fig_path):
    proxies = ["unified", "tipo_a", "extended", "pure_fraud"]
    colors = {"unified": "#1f4e79", "tipo_a": "#7f7f7f",
              "extended": "#bf9000", "pure_fraud": "#c0504d"}
    months_pretty = ["Sep", "Oct", "Nov", "Dic"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for p in proxies:
        aucs = [temporal_dict[p][m]["auc"] for m in
                ("2025-09", "2025-10", "2025-11", "2025-12")]
        ax.plot(months_pretty, aucs, marker="o", lw=1.8,
                color=colors[p], label=p)
    ax.axhline(0.70, color="black", lw=0.7, linestyle="--", alpha=0.6,
               label="Umbral HE2 (AUC=0,70)")
    ax.axhline(0.50, color="gray", lw=0.7, linestyle=":", alpha=0.6,
               label="Aleatorio")
    ax.set_xlabel("Mes (2025)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Estabilidad temporal mensual del IF-40 (test)")
    ax.set_ylim(0.45, 0.90)
    ax.legend(loc="center right", frameon=True, framealpha=0.92, ncol=1)
    for ext in ("pdf", "png"):
        out = fig_path.with_suffix(f".{ext}")
        fig.savefig(out)
        logger.info(f"  Saved {out.name}")
    plt.close(fig)


def plot_score_distributions(if_scores, y_pure, y_unified, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=False)

    for ax, y, title in [
        (axes[0], y_pure, "Proxy pure_fraud"),
        (axes[1], y_unified, "Proxy unified"),
    ]:
        pos = if_scores[y == 1]
        neg = if_scores[y == 0]
        bins = np.linspace(if_scores.min(), if_scores.max(), 80)
        ax.hist(neg, bins=bins, density=True, color="#1f4e79", alpha=0.55,
                label=f"Normal (n={len(neg):,})")
        ax.hist(pos, bins=bins, density=True, color="#c0504d", alpha=0.65,
                label=f"Anómalo (n={len(pos):,})")
        ax.set_xlabel("Score IF (mayor = más anómalo)")
        ax.set_ylabel("Densidad")
        ax.set_title(title)
        ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    fig.suptitle("Distribución del score IF-40 separada por etiqueta proxy (test)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        out = fig_path.with_suffix(f".{ext}")
        fig.savefig(out)
        logger.info(f"  Saved {out.name}")
    plt.close(fig)


def compute_temporal_dict(if_scores, df_test):
    """Recreate the temporal-stability dict directly from this run's scores."""
    out = {p: {} for p in ("unified", "tipo_a", "extended", "pure_fraud")}
    months = df_test["created_at"].dt.to_period("M").astype(str)
    proxy_arrays = {
        "unified": DataManager.assign_proxy_labels(df_test, "unified", settings).to_numpy(),
        "tipo_a": DataManager.assign_proxy_labels(df_test, "tipo_a", settings).to_numpy(),
        "extended": build_extended_proxy(df_test),
        "pure_fraud": build_pure_fraud_proxy(df_test),
    }
    for m in ("2025-09", "2025-10", "2025-11", "2025-12"):
        mask = (months == m).to_numpy()
        for p, y_all in proxy_arrays.items():
            y_m = y_all[mask]
            s_m = if_scores[mask]
            if y_m.sum() == 0 or y_m.sum() == len(y_m):
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_m, s_m))
            out[p][m] = {"n": int(mask.sum()), "rate": float(y_m.mean()),
                         "auc": auc}
    return out


def main():
    logger.info("=" * 60)
    logger.info("Building final figures (IF-40)")
    logger.info("=" * 60)

    X_train = np.load(SCORES_DIR / "X_train_final.npy")
    X_val = np.load(SCORES_DIR / "X_val_final.npy")
    X_test = np.load(SCORES_DIR / "X_test_final.npy")
    df_test = pd.read_parquet(DATA_DIR / "test_features_enriched.parquet")
    logger.info(f"  train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    if_test, lof_test, oc_test = get_or_train_scores(X_train, X_val, X_test)

    y_pure = build_pure_fraud_proxy(df_test)
    y_unified = DataManager.assign_proxy_labels(df_test, "unified", settings).to_numpy()

    scores_by_model = {"if": if_test, "lof": lof_test, "ocsvm": oc_test}

    logger.info("Plotting ROC ...")
    plot_roc(scores_by_model, y_pure, FIG_DIR / "roc_curves")
    logger.info("Plotting PR ...")
    plot_pr(scores_by_model, y_pure, FIG_DIR / "pr_curves")

    logger.info("Computing temporal stability ...")
    temporal_dict = compute_temporal_dict(if_test, df_test)
    plot_temporal_stability(temporal_dict, FIG_DIR / "temporal_stability")

    logger.info("Plotting score distributions ...")
    plot_score_distributions(if_test, y_pure, y_unified,
                             FIG_DIR / "score_distributions")

    logger.info("Done.")


if __name__ == "__main__":
    main()
