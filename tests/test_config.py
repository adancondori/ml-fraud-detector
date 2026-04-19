"""Tests for thesis-aligned configuration module.

TDD: These tests define the CONTRACT for Settings.
Written FIRST (red), then config.py is implemented to pass them (green).
"""
import pytest
from pathlib import Path


class TestSettingsCore:
    """Core settings initialization and validation."""

    def test_settings_loads_without_error(self):
        from config.config import Settings
        s = Settings()
        assert s.environment in ("development", "staging", "production")

    def test_random_seed_default_42(self):
        from config.config import Settings
        s = Settings()
        assert s.random_seed == 42

    def test_project_root_is_path(self):
        from config.config import Settings
        s = Settings()
        assert isinstance(s.project_root, Path)

    def test_no_supervised_params(self):
        """Settings must NOT contain supervised-era parameters."""
        from config.config import Settings
        s = Settings()
        for attr in [
            "model_type", "test_size", "validation_size",
            "mlflow_tracking_uri", "mlflow_experiment_name",
            "use_gpu", "fraud_threshold", "high_risk_threshold",
            "auto_decline_threshold", "api_host", "api_port",
            "database_url", "use_smote", "smote_sampling_strategy",
            "use_class_weights", "fraud_cost_per_transaction",
            "false_positive_cost", "review_capacity_per_day",
            "precision_target", "recall_target",
            "enable_drift_detection", "drift_threshold",
            "performance_degradation_threshold",
        ]:
            assert not hasattr(s, attr), f"Supervised param '{attr}' must be removed"


class TestTemporalSplits:
    """Temporal split boundaries must match thesis contract."""

    def test_train_boundaries(self):
        from config.config import Settings
        s = Settings()
        assert s.train_start == "2025-01-01"
        assert s.train_end == "2025-07-01"

    def test_val_boundaries(self):
        from config.config import Settings
        s = Settings()
        assert s.val_end == "2025-09-01"

    def test_test_boundaries(self):
        from config.config import Settings
        s = Settings()
        assert s.test_end == "2026-01-01"

    def test_warm_start(self):
        from config.config import Settings
        s = Settings()
        assert s.warm_start == "2024-12-01"


class TestProxyDefinitions:
    """Proxy label definitions must match thesis."""

    def test_strict_proxy_list(self):
        from config.config import Settings
        s = Settings()
        assert s.strict_proxy_list == ["totally_refunded", "refunded_to_credit"]

    def test_wide_proxy_list(self):
        from config.config import Settings
        s = Settings()
        assert s.wide_proxy_list == [
            "totally_refunded", "refunded_to_credit", "partially_refunded"
        ]


class TestGridSearchParams:
    """Grid search parameters for IF, LOF, OC-SVM."""

    def test_if_n_estimators_grid(self):
        from config.config import Settings
        s = Settings()
        assert s.if_n_estimators_list == [100, 200, 300, 500]

    def test_if_max_samples_grid(self):
        from config.config import Settings
        s = Settings()
        assert s.if_max_samples_list == [256, 512, 1024, 2048]

    def test_if_max_features_grid(self):
        from config.config import Settings
        s = Settings()
        assert "0.5" in s.if_max_features or 0.5 in s.if_max_features_list

    def test_lof_n_neighbors_grid(self):
        from config.config import Settings
        s = Settings()
        assert s.lof_n_neighbors_list == [20, 50, 100]

    def test_ocsvm_nu_grid(self):
        from config.config import Settings
        s = Settings()
        assert s.ocsvm_nu_list == [0.01, 0.05, 0.10]

    def test_ocsvm_gamma_grid(self):
        from config.config import Settings
        s = Settings()
        assert s.ocsvm_gamma_list == ["scale", "auto"]


class TestEvaluationParams:
    """Evaluation parameters for hypothesis testing."""

    def test_bootstrap_iterations(self):
        from config.config import Settings
        s = Settings()
        assert s.bootstrap_n == 1000

    def test_top_k_percents(self):
        from config.config import Settings
        s = Settings()
        assert 0.05 in s.top_k_percents_list

    def test_shap_sample_size(self):
        from config.config import Settings
        s = Settings()
        assert s.shap_sample_size == 5000

    def test_multi_seeds(self):
        from config.config import Settings
        s = Settings()
        assert s.multi_seeds_list == [42, 52, 62]


class TestDirectoryProperties:
    """Output directory properties."""

    def test_processed_dir(self):
        from config.config import Settings
        s = Settings()
        assert s.processed_dir.name == "processed"

    def test_output_dir_property(self):
        from config.config import Settings
        s = Settings()
        assert s.output_dir == "output" or hasattr(s, "output_dir")

    def test_figures_dir(self):
        from config.config import Settings
        s = Settings()
        assert "figures" in str(s.figures_dir)

    def test_tables_dir(self):
        from config.config import Settings
        s = Settings()
        assert "tables" in str(s.tables_dir)

    def test_scores_dir(self):
        from config.config import Settings
        s = Settings()
        assert "scores" in str(s.scores_dir)

    def test_models_output_dir(self):
        from config.config import Settings
        s = Settings()
        assert "models" in str(s.models_output_dir)

    def test_manifests_dir(self):
        from config.config import Settings
        s = Settings()
        assert "manifests" in str(s.manifests_dir)

    def test_ensure_directories_creates_all(self, tmp_path):
        from config.config import Settings
        s = Settings(project_root=tmp_path, data_dir="data", output_dir="output")
        s.ensure_directories()
        assert (tmp_path / "data" / "processed").is_dir()
        assert (tmp_path / "output" / "figures").is_dir()
        assert (tmp_path / "output" / "tables").is_dir()
        assert (tmp_path / "output" / "scores").is_dir()
        assert (tmp_path / "output" / "models").is_dir()
        assert (tmp_path / "output" / "manifests").is_dir()


class TestClickHouseConfig:
    """ClickHouse connection settings preserved."""

    def test_clickhouse_host_exists(self):
        from config.config import Settings
        s = Settings()
        assert hasattr(s, "clickhouse_host")

    def test_clickhouse_database(self):
        from config.config import Settings
        s = Settings()
        assert s.clickhouse_database == "pbp_productionDB_optimized"

    def test_clickhouse_table(self):
        from config.config import Settings
        s = Settings()
        assert s.clickhouse_table == "payments"
