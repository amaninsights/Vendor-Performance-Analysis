# Makefile for Vendor Performance Analytics
# ==========================================
#
# Usage:
#   make install     Install all dependencies
#   make test        Run test suite
#   make lint        Run linting checks
#   make format      Format code with black
#   make run         Run the full pipeline
#   make clean       Clean build artifacts

.PHONY: help install install-dev test lint format run clean docs

# Default target
help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║       Vendor Performance Analytics - Makefile Commands         ║"
	@echo "╠════════════════════════════════════════════════════════════════╣"
	@echo "║  make install      Install production dependencies             ║"
	@echo "║  make install-dev  Install development dependencies            ║"
	@echo "║  make test         Run pytest test suite                       ║"
	@echo "║  make test-cov     Run tests with coverage report              ║"
	@echo "║  make lint         Run flake8 linting                          ║"
	@echo "║  make format       Format code with black and isort            ║"
	@echo "║  make run          Run the full analytics pipeline             ║"
	@echo "║  make ingest       Run data ingestion only                     ║"
	@echo "║  make charts       Generate all charts                         ║"
	@echo "║  make clean        Clean build artifacts and cache             ║"
	@echo "║  make build        Build distribution package                  ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black isort flake8 mypy

# Testing targets
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# Code quality targets
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503

format:
	black src/ tests/ main.py
	isort src/ tests/ main.py

type-check:
	mypy src/ --ignore-missing-imports

# Run targets
run:
	python main.py

run-top20:
	python main.py --top 20

ingest:
	python main.py --ingest-only

charts:
	python main.py --export-charts

# Build targets
build:
	pip install build
	python -m build

# Cleanup targets
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage 2>/dev/null || true
	@echo "Cleaned build artifacts and cache files"

clean-data:
	rm -rf data/processed/*.db 2>/dev/null || true
	rm -rf reports/figures/*.png 2>/dev/null || true
	@echo "Cleaned processed data and generated charts"

# Development workflow
dev: install-dev format lint test
	@echo "Development checks complete!"

# Full pipeline with all outputs
full: install run charts
	@echo "Full pipeline complete! Check reports/ for outputs."
