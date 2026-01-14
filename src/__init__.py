"""
Vendor Performance Analytics
============================

A production-grade data analytics pipeline for retail vendor performance evaluation.

This package provides:
- Data ingestion and ETL pipelines
- SQL-based transformations with CTEs
- Statistical analysis and metrics computation
- Automated visualization generation
- Comprehensive logging and error handling

Author: Aman Saroha
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Aman Saroha"

from src.data import loader, transformer
from src.analysis import metrics
from src.visualization import charts

__all__ = ["loader", "transformer", "metrics", "charts"]
