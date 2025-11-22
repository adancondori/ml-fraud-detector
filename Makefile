.PHONY: install install-dev setup clean test lint format type-check pre-commit notebook mlflow git-check git-status verify help

help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make setup          - Setup environment (.env + pre-commit)"
	@echo ""
	@echo "Development:"
	@echo "  make notebook       - Start Jupyter notebook"
	@echo "  make mlflow         - Start MLflow UI"
	@echo "  make verify         - Verify project setup"
	@echo ""
	@echo "Code Quality:"
	@echo "  make test           - Run tests with coverage"
	@echo "  make lint           - Run linting (flake8)"
	@echo "  make format         - Format code (black + isort)"
	@echo "  make type-check     - Run type checking (mypy)"
	@echo "  make pre-commit     - Run pre-commit on all files"
	@echo ""
	@echo "Git & Version Control:"
	@echo "  make git-check      - Check files before commit"
	@echo "  make git-status     - Show Git status with warnings"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Clean temporary files"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from .env.example"; \
	else \
		echo ".env file already exists"; \
	fi
	pre-commit install
	@echo "Setup complete! Don't forget to edit .env with your configuration"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	@echo "Cleaned temporary files"

test:
	pytest --cov=src/fraud_detector --cov-report=html --cov-report=term-missing -v

lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503

format:
	black src/ tests/ config/ --line-length=100
	isort src/ tests/ config/ --profile black

type-check:
	mypy src/ --ignore-missing-imports

pre-commit:
	pre-commit run --all-files

notebook:
	jupyter notebook notebooks/

mlflow:
	mlflow ui --port 5000

git-check:
	@echo "🔍 Verificando archivos antes de commit..."
	@python3 check_git_status.py

git-status:
	@echo "📋 Git Status:"
	@git status
	@echo ""
	@python3 check_git_status.py

verify:
	@echo "✅ Verificando configuración del proyecto..."
	@python3 verify_setup.py
