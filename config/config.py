"""
Thesis-aligned configuration for anomaly detection pipeline.

All supervised-era parameters removed. Configuration matches the technical
contract in PLAN-FINAL/01_CONTRATO_ALCANCE.md.

Validated counts (ClickHouse FINAL, 2026-03-17):
  - N=6,784,695  train=3,137,086  val=1,130,118  test=2,517,491  warm=419,820
  - proxy_strict=429,469 (6.33%)  proxy_wide=512,644 (7.55%)

Proxy taxonomy (5 types, updated 2026-04-15):
  - Tipo A: reembolso (status-based)
  - Tipo B: circuito de credito (circuit_closure > 80%, cash_loaded > $500)
  - Tipo C: descuento anomalo (discount_ratio_30d > 100%)
  - Tipo D: velocidad extrema (txn_count_1d > 100)
  - Tipo E: gratuitas sistematicas (free_pct_30d > 25%, free_count_30d > 10)
  - Proxy unificado: OR(A, B, C, D, E) — evaluacion principal
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pipeline settings — single source of truth for the thesis study."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignores old supervised env vars silently
    )

    # ── Environment ──────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"

    # ── Paths ────────────────────────────────────────────────────
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: str = "data"
    output_dir: str = "output"
    logs_dir: str = "logs"

    # ── Random seed ──────────────────────────────────────────────
    random_seed: int = 42
    multi_seeds: str = "42,52,62"

    # ── Logging ──────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["text", "json"] = "json"

    # ── Parallelism ──────────────────────────────────────────────
    n_jobs: int = -1

    # ── Currency Normalization ─────────────────────────────────
    exchange_rates_path: str = "data/external/exchange_rates.csv"
    reference_currency: str = "USD"

    # ── ClickHouse ───────────────────────────────────────────────
    clickhouse_host: str = "localhost"
    clickhouse_port: int = Field(default=8443, ge=1, le=65535)
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "pbp_productionDB_optimized"
    clickhouse_table: str = "payments"
    clickhouse_secure: bool = False

    # ── Temporal Split Boundaries ────────────────────────────────
    warm_start: str = "2024-12-01"
    train_start: str = "2025-01-01"
    train_end: str = "2025-07-01"  # exclusive
    val_end: str = "2025-09-01"  # exclusive
    test_end: str = "2026-01-01"  # exclusive

    # ── Proxy Label Definitions ──────────────────────────────────
    # Tipo A: reembolso (status-based)
    strict_proxy_statuses: str = "totally_refunded,refunded_to_credit"
    wide_proxy_statuses: str = "totally_refunded,refunded_to_credit,partially_refunded"

    # Tipo B: circuito de credito (rolling 30d aggregates)
    tipo_b_circuit_closure_threshold: float = 0.80
    tipo_b_cash_loaded_threshold: float = 500.0

    # Tipo C: descuento anomalo (rolling 30d)
    tipo_c_discount_ratio_threshold: float = 1.00

    # Tipo D: velocidad extrema (daily count)
    tipo_d_txn_count_1d_threshold: int = 100

    # Tipo E: gratuitas sistematicas (rolling 30d)
    tipo_e_free_pct_threshold: float = 0.25
    tipo_e_free_count_threshold: int = 10

    # ── Isolation Forest Grid ────────────────────────────────────
    if_n_estimators: str = "100,200,300,500"
    if_max_samples: str = "256,512,1024,2048"
    if_max_features: str = "0.5,0.75,1.0"
    if_contamination: str = "0.01,0.03,0.05,0.06,0.08"

    # ── LOF Grid ─────────────────────────────────────────────────
    lof_n_neighbors: str = "20,50,100"

    # ── OC-SVM Grid ──────────────────────────────────────────────
    ocsvm_nu: str = "0.01,0.05,0.10"
    ocsvm_gamma: str = "scale,auto"
    ocsvm_subsample: int = 100_000

    # ── Evaluation ───────────────────────────────────────────────
    bootstrap_n: int = 1000
    top_k_percents: str = "0.01,0.02,0.05,0.10"
    shap_sample_size: int = 5000

    # ── Hypothesis Thresholds (HE1-HE4) ─────────────────────────
    he1_alpha: float = 0.05
    he1_min_rank_biserial: float = 0.10
    he2_min_auc_roc: float = 0.70
    he2_min_ap_above_base_rate: bool = True  # AP > proxy unificado base rate
    he3_min_enrichment_factor: float = 1.0
    he4_min_metrics_won: int = 3  # IF must win >= 3 of 4 metrics vs competitors

    # ── Proxy parsing properties ─────────────────────────────────

    @property
    def strict_proxy_list(self) -> List[str]:
        """Tipo A statuses (backward compat alias)."""
        return [s.strip() for s in self.strict_proxy_statuses.split(",")]

    @property
    def tipo_a_list(self) -> List[str]:
        """Tipo A: reembolso statuses."""
        return self.strict_proxy_list

    @property
    def wide_proxy_list(self) -> List[str]:
        return [s.strip() for s in self.wide_proxy_statuses.split(",")]

    # ── Grid parsing properties ──────────────────────────────────

    @property
    def if_n_estimators_list(self) -> List[int]:
        return [int(x.strip()) for x in self.if_n_estimators.split(",")]

    @property
    def if_max_samples_list(self) -> List[int]:
        return [int(x.strip()) for x in self.if_max_samples.split(",")]

    @property
    def if_max_features_list(self) -> List[float]:
        return [float(x.strip()) for x in self.if_max_features.split(",")]

    @property
    def if_contamination_list(self) -> List[float]:
        return [float(x.strip()) for x in self.if_contamination.split(",")]

    @property
    def lof_n_neighbors_list(self) -> List[int]:
        return [int(x.strip()) for x in self.lof_n_neighbors.split(",")]

    @property
    def ocsvm_nu_list(self) -> List[float]:
        return [float(x.strip()) for x in self.ocsvm_nu.split(",")]

    @property
    def ocsvm_gamma_list(self) -> List[str]:
        return [x.strip() for x in self.ocsvm_gamma.split(",")]

    @property
    def top_k_percents_list(self) -> List[float]:
        return [float(x.strip()) for x in self.top_k_percents.split(",")]

    @property
    def multi_seeds_list(self) -> List[int]:
        return [int(x.strip()) for x in self.multi_seeds.split(",")]

    # ── Directory properties ─────────────────────────────────────

    @property
    def processed_dir(self) -> Path:
        return self.project_root / self.data_dir / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.project_root / self.output_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.project_root / self.output_dir / "tables"

    @property
    def scores_dir(self) -> Path:
        return self.project_root / self.output_dir / "scores"

    @property
    def models_output_dir(self) -> Path:
        return self.project_root / self.output_dir / "models"

    @property
    def manifests_dir(self) -> Path:
        return self.project_root / self.output_dir / "manifests"

    @property
    def metrics_dir(self) -> Path:
        return self.project_root / self.output_dir / "metrics"

    @property
    def logs_path(self) -> Path:
        return self.project_root / self.logs_dir

    @property
    def exchange_rates_file(self) -> Path:
        return self.project_root / self.exchange_rates_path

    # ── Path helpers ─────────────────────────────────────────────

    @field_validator("project_root", mode="before")
    @classmethod
    def _resolve_root(cls, v):
        return Path(v) if isinstance(v, str) else v

    def get_absolute_path(self, path) -> Path:
        p = Path(path) if isinstance(path, str) else path
        if p.is_absolute():
            return p
        return self.project_root / p

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    # ── Directory creation ───────────────────────────────────────

    def ensure_directories(self) -> None:
        """Create all output directories needed by the pipeline."""
        dirs = [
            self.processed_dir,
            self.figures_dir,
            self.tables_dir,
            self.scores_dir,
            self.models_output_dir,
            self.manifests_dir,
            self.metrics_dir,
            self.logs_path,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Global singleton — importable as `from config.config import settings`
settings = Settings()
