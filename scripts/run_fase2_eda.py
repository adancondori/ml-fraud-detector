"""
Fase 2 — EDA y Diagnóstico para Capítulo 2 de la tesis.

Genera programáticamente todas las tablas LaTeX, figuras y métricas
requeridas por OE2 (diagnóstico del estado transaccional).

Uso:
    python scripts/run_fase2_eda.py
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.config import settings
from fraud_detector.data.loader import DataManager

# ── Plot style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

FIGURES = settings.figures_dir
TABLES = settings.tables_dir
METRICS = settings.metrics_dir
for d in (FIGURES, TABLES, METRICS):
    d.mkdir(parents=True, exist_ok=True)

dm = DataManager(settings)


def save_fig(name: str) -> None:
    for fmt in ("pdf", "png"):
        plt.savefig(FIGURES / f"{name}.{fmt}", bbox_inches="tight", dpi=150)
    plt.close()


def save_tex(df: pd.DataFrame, name: str, **kwargs) -> None:
    path = TABLES / f"{name}.tex"
    path.write_text(df.to_latex(**kwargs))


# ── Load all splits into one DF for global stats ────────────────────
# Memory: ~5 GB peak. Load sequentially, concat, then work.
print("Loading splits...")
frames = []
split_meta = {}
for split in ("train", "val", "test"):
    df = dm.load_split(split)
    proxy_a = DataManager.assign_proxy_labels(df, "tipo_a", settings)
    proxy_w = DataManager.assign_proxy_labels(df, "wide", settings)
    split_meta[split] = {
        "rows": len(df),
        "proxy_strict_rate": round(float(proxy_a.mean()), 6),
        "proxy_wide_rate": round(float(proxy_w.mean()), 6),
        "date_min": str(df["created_at"].min()),
        "date_max": str(df["created_at"].max()),
    }
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
del frames
gc.collect()

N = len(all_df)
proxy_strict = DataManager.assign_proxy_labels(all_df, "tipo_a", settings)
proxy_wide = DataManager.assign_proxy_labels(all_df, "wide", settings)
all_df["proxy_strict"] = proxy_strict
all_df["proxy_wide"] = proxy_wide
all_df["month"] = all_df["created_at"].dt.to_period("M")
all_df["hour"] = all_df["created_at"].dt.hour
all_df["day_of_week"] = all_df["created_at"].dt.dayofweek + 1  # 1=Mon

print(f"Total: {N:,} rows")


# ═══════════════════════════════════════════════════════════════════
# 1. Status distribution
# ═══════════════════════════════════════════════════════════════════
print("1. Status distribution...")
status_counts = all_df["status"].value_counts()
status_df = pd.DataFrame({
    "Status": status_counts.index,
    "N": status_counts.values,
    "%": (status_counts.values / N * 100).round(2),
})
save_tex(status_df, "cap2_status_distribution", index=False, escape=False)

fig, ax = plt.subplots(figsize=(10, 5))
status_df_plot = status_df.head(10)
ax.barh(status_df_plot["Status"], status_df_plot["N"], color="steelblue")
ax.set_xlabel("Transactions")
ax.set_title("Transaction Status Distribution")
ax.invert_yaxis()
for i, (n, pct) in enumerate(zip(status_df_plot["N"], status_df_plot["%"])):
    ax.text(n + N * 0.005, i, f"{n:,} ({pct}%)", va="center", fontsize=9)
save_fig("cap2_status_distribution")


# ═══════════════════════════════════════════════════════════════════
# 2. Channel (source_enum) distribution
# ═══════════════════════════════════════════════════════════════════
print("2. Channel distribution...")
if "source_enum" in all_df.columns:
    ch = all_df["source_enum"].value_counts()
    ch_df = pd.DataFrame({"Channel": ch.index, "N": ch.values, "%": (ch.values / N * 100).round(2)})
    save_tex(ch_df, "cap2_channel_distribution", index=False, escape=False)


# ═══════════════════════════════════════════════════════════════════
# 3. Gateway distribution
# ═══════════════════════════════════════════════════════════════════
print("3. Gateway distribution...")
gw = all_df["gateway"].value_counts()
gw_df = pd.DataFrame({"Gateway": gw.index, "N": gw.values, "%": (gw.values / N * 100).round(2)})
save_tex(gw_df, "cap2_gateway_distribution", index=False, escape=False)


# ═══════════════════════════════════════════════════════════════════
# 4. Payment method distribution
# ═══════════════════════════════════════════════════════════════════
print("4. Payment method distribution...")
pm = all_df["payment_method"].value_counts()
pm_df = pd.DataFrame({"Method": pm.index, "N": pm.values, "%": (pm.values / N * 100).round(2)})
save_tex(pm_df, "cap2_payment_method_distribution", index=False, escape=False)

fig, ax = plt.subplots(figsize=(10, 5))
pm_top = pm_df.head(10)
ax.barh(pm_top["Method"], pm_top["N"], color="darkorange")
ax.set_xlabel("Transactions")
ax.set_title("Top 10 Payment Methods")
ax.invert_yaxis()
save_fig("cap2_payment_methods")


# ═══════════════════════════════════════════════════════════════════
# 5. Amount descriptive statistics
# ═══════════════════════════════════════════════════════════════════
print("5. Amount descriptive stats...")
amt = all_df["amount"]
desc = {
    "N": f"{N:,}",
    "Mean (USD)": f"{amt.mean():.2f}",
    "Median (USD)": f"{amt.median():.2f}",
    "Std": f"{amt.std():.2f}",
    "Min": f"{amt.min():.2f}",
    "Max": f"{amt.max():.2f}",
    "P95": f"{amt.quantile(0.95):.2f}",
    "P99": f"{amt.quantile(0.99):.2f}",
    "Zero amount %": f"{(amt == 0).mean() * 100:.1f}",
}
desc_df = pd.DataFrame(list(desc.items()), columns=["Metric", "Value"])
save_tex(desc_df, "cap2_amount_descriptive", index=False, escape=False)

# Amount by status
status_groups = ["captured", "totally_refunded", "refunded_to_credit", "partially_refunded"]
amt_by_status_rows = []
for s in status_groups:
    sub = all_df.loc[all_df["status"] == s, "amount"]
    if len(sub) == 0:
        continue
    amt_by_status_rows.append({
        "Status": s,
        "N": f"{len(sub):,}",
        "Mean": f"{sub.mean():.2f}",
        "Median": f"{sub.median():.2f}",
        "P95": f"{sub.quantile(0.95):.2f}",
        "P99": f"{sub.quantile(0.99):.2f}",
    })
if amt_by_status_rows:
    save_tex(
        pd.DataFrame(amt_by_status_rows),
        "cap2_amount_by_status",
        index=False,
        escape=False,
    )

# Histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(amt[amt <= 500].values, bins=100, color="steelblue", edgecolor="none")
axes[0].set_xlabel("Amount (USD)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Amount Distribution (truncated at $500)")

log_amt = np.log1p(amt.values)
axes[1].hist(log_amt, bins=100, color="darkorange", edgecolor="none")
axes[1].set_xlabel("log(1 + Amount)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Log-Amount Distribution")
plt.tight_layout()
save_fig("cap2_amount_distribution")


# ═══════════════════════════════════════════════════════════════════
# 6. Monthly volume
# ═══════════════════════════════════════════════════════════════════
print("6. Monthly volume...")
monthly = all_df.groupby("month").agg(
    N=("id", "count"),
    proxy_rate=("proxy_strict", "mean"),
).reset_index()
monthly["month_str"] = monthly["month"].astype(str)
monthly["proxy_rate_pct"] = (monthly["proxy_rate"] * 100).round(2)

save_tex(
    monthly[["month_str", "N", "proxy_rate_pct"]].rename(
        columns={"month_str": "Month", "proxy_rate_pct": "Proxy Rate (%)"}
    ),
    "cap2_monthly_volume",
    index=False,
    escape=False,
)


# ═══════════════════════════════════════════════════════════════════
# 7. Temporal patterns (hour, day, monthly trend)
# ═══════════════════════════════════════════════════════════════════
print("7. Temporal patterns...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Hourly
hourly = all_df.groupby("hour")["id"].count()
axes[0].bar(hourly.index, hourly.values, color="steelblue")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Transactions")
axes[0].set_title("Hourly Volume")
axes[0].set_xticks(range(0, 24, 2))

# Day of week
dow = all_df.groupby("day_of_week")["id"].count()
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
axes[1].bar(dow.index, dow.values, color="darkorange")
axes[1].set_xlabel("Day of Week")
axes[1].set_ylabel("Transactions")
axes[1].set_title("Daily Volume")
axes[1].set_xticks(range(1, 8))
axes[1].set_xticklabels(days)

# Monthly
axes[2].bar(range(len(monthly)), monthly["N"].values, color="seagreen")
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Transactions")
axes[2].set_title("Monthly Volume (2025)")
axes[2].set_xticks(range(len(monthly)))
axes[2].set_xticklabels(monthly["month_str"].values, rotation=45, ha="right")

plt.tight_layout()
save_fig("cap2_temporal_patterns")


# ═══════════════════════════════════════════════════════════════════
# 8. Proxy rate monthly
# ═══════════════════════════════════════════════════════════════════
print("8. Proxy rate monthly...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    monthly["month_str"].values,
    monthly["proxy_rate_pct"].values,
    marker="o",
    color="crimson",
    linewidth=2,
)
avg_rate = proxy_strict.mean() * 100
ax.axhline(avg_rate, color="gray", linestyle="--", label=f"Average: {avg_rate:.2f}%")
ax.set_xlabel("Month")
ax.set_ylabel("Proxy Rate (%)")
ax.set_title("Monthly Proxy Rate (Tipo A)")
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
save_fig("cap2_proxy_rate_monthly")


# ═══════════════════════════════════════════════════════════════════
# 9. Proxy profiles (proxy+ vs proxy-)
# ═══════════════════════════════════════════════════════════════════
print("9. Proxy profiles...")
pos = all_df[all_df["proxy_strict"] == 1]["amount"]
neg = all_df[all_df["proxy_strict"] == 0]["amount"]

profiles = pd.DataFrame({
    "Metric": ["N", "% of Total", "Mean (USD)", "Median (USD)", "Std", "P95", "P99"],
    "Proxy+": [
        f"{len(pos):,}",
        f"{len(pos)/N*100:.2f}",
        f"{pos.mean():.2f}",
        f"{pos.median():.2f}",
        f"{pos.std():.2f}",
        f"{pos.quantile(0.95):.2f}",
        f"{pos.quantile(0.99):.2f}",
    ],
    "Proxy-": [
        f"{len(neg):,}",
        f"{len(neg)/N*100:.2f}",
        f"{neg.mean():.2f}",
        f"{neg.median():.2f}",
        f"{neg.std():.2f}",
        f"{neg.quantile(0.95):.2f}",
        f"{neg.quantile(0.99):.2f}",
    ],
})
save_tex(profiles, "cap2_proxy_profiles", index=False, escape=False)

# Box plot
fig, ax = plt.subplots(figsize=(8, 5))
plot_data = pd.DataFrame({
    "Amount (USD)": pd.concat([pos.clip(upper=500), neg.clip(upper=500)]),
    "Group": ["Proxy+"] * len(pos) + ["Proxy-"] * len(neg),
})
sns.boxplot(data=plot_data, x="Group", y="Amount (USD)", ax=ax, palette=["crimson", "steelblue"])
ax.set_title("Amount Distribution: Proxy+ vs Proxy- (capped at $500)")
save_fig("cap2_proxy_amounts")
del plot_data
gc.collect()


# ═══════════════════════════════════════════════════════════════════
# 10. Hourly proxy pattern
# ═══════════════════════════════════════════════════════════════════
print("10. Hourly proxy pattern...")
hourly_proxy = all_df.groupby("hour").agg(
    total=("id", "count"),
    proxy_n=("proxy_strict", "sum"),
).reset_index()
hourly_proxy["proxy_rate"] = (hourly_proxy["proxy_n"] / hourly_proxy["total"] * 100).round(2)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(hourly_proxy["hour"], hourly_proxy["total"], color="steelblue", alpha=0.5, label="Total")
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Total Transactions", color="steelblue")
ax2 = ax1.twinx()
ax2.plot(hourly_proxy["hour"], hourly_proxy["proxy_rate"], color="crimson", marker="o", linewidth=2, label="Proxy Rate %")
ax2.set_ylabel("Proxy Rate (%)", color="crimson")
ax1.set_title("Hourly Volume and Proxy Rate")
ax1.set_xticks(range(0, 24, 2))
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
save_fig("cap2_hourly_proxy")


# ═══════════════════════════════════════════════════════════════════
# 11. Summary JSON
# ═══════════════════════════════════════════════════════════════════
print("11. Saving summary JSON...")
summary = {
    "N_total": N,
    "split_stats": split_meta,
    "proxy_strict_n": int(proxy_strict.sum()),
    "proxy_strict_rate": round(float(proxy_strict.mean()) * 100, 2),
    "proxy_wide_n": int(proxy_wide.sum()),
    "proxy_wide_rate": round(float(proxy_wide.mean()) * 100, 2),
    "amount_mean": round(float(amt.mean()), 2),
    "amount_median": round(float(amt.median()), 2),
    "amount_p95": round(float(amt.quantile(0.95)), 2),
    "amount_p99": round(float(amt.quantile(0.99)), 2),
    "amount_zero_pct": round(float((amt == 0).mean()) * 100, 1),
    "amount_gt_1M": int((amt > 1_000_000).sum()),
    "status_distribution": status_counts.to_dict(),
    "monthly_proxy_rates": dict(zip(monthly["month_str"].tolist(), monthly["proxy_rate_pct"].tolist())),
    "n_currencies": int(all_df["currency"].nunique()),
    "n_gateways": int(all_df["gateway"].nunique()),
    "n_payment_methods": int(all_df["payment_method"].nunique()),
}

summary_path = METRICS / "cap2_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, default=str))

print(f"""
{'='*60}
EDA Capítulo 2 — Artefactos generados
{'='*60}
Tablas:  {len(list(TABLES.glob('cap2_*.tex')))} archivos en {TABLES}
Figuras: {len(list(FIGURES.glob('cap2_*')))} archivos en {FIGURES}
Métricas: {summary_path}

N total:         {N:,}
Proxy estricto:  {summary['proxy_strict_n']:,} ({summary['proxy_strict_rate']}%)
Proxy amplio:    {summary['proxy_wide_n']:,} ({summary['proxy_wide_rate']}%)
Amount mean:     ${summary['amount_mean']:.2f}
Amount zero:     {summary['amount_zero_pct']}%
{'='*60}
""")
