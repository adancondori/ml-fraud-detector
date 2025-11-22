"""Tests for configuration module."""
import pytest
from pathlib import Path

from config.config import Settings


def test_settings_initialization():
    """Test settings can be initialized with defaults."""
    settings = Settings()

    assert settings.environment in ["development", "staging", "production"]
    assert settings.random_seed == 42
    assert isinstance(settings.project_root, Path)


def test_settings_validation():
    """Test settings validation works."""
    # Valid test_size
    settings = Settings(test_size=0.2)
    assert settings.test_size == 0.2

    # Invalid test_size should raise error
    with pytest.raises(Exception):
        Settings(test_size=1.5)


def test_path_resolution():
    """Test path resolution methods."""
    settings = Settings()

    # Test absolute path
    abs_path = Path("/tmp/test")
    resolved = settings.get_absolute_path(abs_path)
    assert resolved == abs_path

    # Test relative path
    rel_path = Path("data/raw")
    resolved = settings.get_absolute_path(rel_path)
    assert resolved.is_absolute()


def test_environment_properties():
    """Test environment property helpers."""
    dev_settings = Settings(environment="development")
    assert dev_settings.is_development is True
    assert dev_settings.is_production is False

    prod_settings = Settings(environment="production")
    assert prod_settings.is_development is False
    assert prod_settings.is_production is True
