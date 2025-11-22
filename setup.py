"""Setup script for fraud-detector package."""
from setuptools import find_packages, setup

setup(
    name="fraud-detector",
    version="0.1.0",
    description="Machine Learning Fraud Detection System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "imbalanced-learn>=0.11.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "mlflow>=2.9.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.5.0",
            "ipykernel>=6.25.0",
            "jupyter>=1.0.0",
        ]
    },
)
