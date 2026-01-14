<div align="center">

#  Vendor Performance Analytics

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/amaninsights/Vendor-Performance-Analysis/actions)

**A production-grade data analytics pipeline for retail vendor performance evaluation**

*Analyze 2GB+ of retail data to identify top performers, calculate profitability metrics, and generate actionable insights*

[Features](#-features)  [Quick Start](#-quick-start)  [Architecture](#-architecture)  [Documentation](#-documentation)  [Results](#-key-findings)

</div>

---

##  Overview

This project implements an **end-to-end ETL pipeline** that transforms raw retail transaction data into actionable vendor performance insights. Built with production-grade code practices including:

-  Modular architecture with separation of concerns
-  Comprehensive logging and error handling
-  Unit tests with pytest
-  CI/CD pipeline with GitHub Actions
-  Type hints and docstrings throughout
-  Configuration-driven design

##  Features

| Feature | Description |
|---------|-------------|
|  **ETL Pipeline** | Ingest CSV files into SQLite with automated schema detection |
|  **SQL Transformations** | Complex CTEs for multi-table aggregations |
|  **Derived Metrics** | Gross profit, profit margin, stock turnover, inventory change |
|  **Visualizations** | Auto-generated charts including dashboards, heatmaps, scatter plots |
|  **Performance Scoring** | Weighted scoring algorithm with vendor segmentation (A/B/C/D tiers) |
|  **Executive Reports** | Formatted summary reports with KPIs |

##  Quick Start

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/amaninsights/Vendor-Performance-Analysis.git
cd Vendor-Performance-Analysis

# Install dependencies
pip install -r requirements.txt

# Or using make
make install
```

### Run the Pipeline

```bash
# Run full pipeline
python main.py

# Run with options
python main.py --top 20              # Show top 20 vendors
python main.py --export-charts       # Generate all visualizations
python main.py --ingest-only         # Only run data ingestion

# Using make
make run                             # Full pipeline
make charts                          # Generate charts
make test                            # Run tests
```

##  Project Structure

```
Vendor-Performance-Analysis/

  src/                          # Source code
     data/                     # Data processing modules
       loader.py                # CSV to SQLite ingestion
       transformer.py           # SQL CTEs & data cleaning
     analysis/                 # Analytics modules
       metrics.py               # KPIs & performance scoring
     visualization/            # Chart generation
       charts.py                # Publication-ready visualizations
     utils/                    # Utilities
        logger.py                # Centralized logging

  tests/                        # Unit tests
    conftest.py                  # Pytest fixtures
    test_transformer.py          # Transformer tests
    test_metrics.py              # Metrics tests

  config/                       # Configuration
    config.yaml                  # Pipeline settings

  notebooks/                    # Jupyter notebooks
    Vendor Performance Analysis.ipynb

  data/                         # Data directories
    raw/                         # Input CSV files
    processed/                   # SQLite database

  reports/                      # Output reports
    figures/                     # Generated charts

  Dashboard/                    # Power BI dashboard
    Dashboard.pbix

  .github/workflows/            # CI/CD
    ci.yml                       # GitHub Actions workflow

 main.py                          # Pipeline entry point
 pyproject.toml                   # Modern Python packaging
 Makefile                         # Automation commands
 requirements.txt                 # Dependencies
 README.md                        # This file
```

##  Architecture

```
          
   RAW DATA              TRANSFORM              ANALYZE      
   (CSV Files)      (SQL CTEs)       (Metrics)     
                                                             
   purchases            Aggregation          KPIs         
   sales                Cleaning             Scoring      
   inventory            Derived cols         Segmentation 
          
                                                        
                                                        
          
   POWER BI              EXPORT                VISUALIZE     
   Dashboard        (CSV/DB)         (Charts)      
                                                             
   Interactive          vendor_scores        Bar charts   
   Drill-down           SQLite DB            Heatmaps     
          
```

##  Key Findings

<table>
<tr>
<td width="50%">

###  Performance Metrics

| Metric | Value |
|--------|-------|
| Total Vendors Analyzed | 300+ |
| Total Revenue | $50M+ |
| Avg Profit Margin | 18.5% |
| Stock Turnover | 3.2x |

</td>
<td width="50%">

###  Top Insights

- **Top 10 vendors** contribute **65%** of total revenue
- **A-tier vendors** (top 25%) have **3x higher** profit margins
- Strong correlation between stock turnover and profitability
- 15% of vendors account for negative margins

</td>
</tr>
</table>

##  Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_transformer.py -v
```

##  Documentation

### Configuration

Edit `config/config.yaml` to customize:

```yaml
database:
  path: "data/processed/inventory.db"
  
analysis:
  min_profit_margin: 0
  top_vendors_count: 10
  
visualization:
  figure_dpi: 300
  color_palette: "husl"
```

### API Reference

```python
from src.data.loader import DataLoader
from src.data.transformer import DataTransformer
from src.analysis.metrics import VendorMetrics
from src.visualization.charts import VendorCharts

# Load and transform data
loader = DataLoader("config/config.yaml")
loader.ingest_all()

transformer = DataTransformer("config/config.yaml")
df = transformer.create_vendor_summary()
df = transformer.clean_data(df)

# Analyze
metrics = VendorMetrics(df)
kpis = metrics.calculate_kpis()
scores = metrics.calculate_performance_scores()

# Visualize
charts = VendorCharts(df)
charts.save_all_charts("reports/figures/")
```

##  Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Database** | SQLite, SQLAlchemy |
| **Visualization** | Matplotlib, Seaborn |
| **Statistics** | SciPy |
| **Dashboard** | Power BI |
| **Testing** | pytest, pytest-cov |
| **CI/CD** | GitHub Actions |
| **Code Quality** | black, isort, flake8 |

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Author

**Aman Saroha**

[![GitHub](https://img.shields.io/badge/GitHub-amaninsights-181717?style=flat-square&logo=github)](https://github.com/amaninsights)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/amaninsights)

---

<div align="center">

** Star this repo if you find it useful!**

</div>
