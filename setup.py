"""Setup script for fraud-detector package."""
from setuptools import find_packages, setup

setup(
    name="fraud-detector",
    version="0.1.0",
    description="Unsupervised anomaly detection pipeline for digital payments",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "clickhouse-connect>=0.7.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "shap>=0.42.0",
    ],
)
