"""
Test Configuration
==================

Pytest fixtures and configuration for testing.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_vendor_data():
    """Create sample vendor DataFrame for testing."""
    return pd.DataFrame({
        'VendorNumber': [1, 2, 3, 4, 5],
        'VendorName': ['Vendor A', 'Vendor B', 'Vendor C!!', 'Vendor D', 'Vendor E'],
        'TotalPurchases': [100, 200, 150, 300, 250],
        'TotalPurchaseDollars': [10000, 20000, 15000, 30000, 25000],
        'TotalPurchaseQty': [500, 1000, 750, 1500, 1250],
        'TotalSales': [90, 180, 130, 280, 230],
        'TotalSalesDollars': [12000, 25000, 18000, 35000, 30000],
        'TotalSalesQty': [450, 900, 650, 1400, 1150],
        'AvgSalesPrice': [26.67, 27.78, 27.69, 25.00, 26.09],
        'BeginInventory': [100, 200, 150, 250, 200],
        'EndInventory': [150, 300, 200, 350, 300],
        'AvgPurchasePrice': [20.0, 20.0, 20.0, 20.0, 20.0],
        'MinPrice': [18.0, 18.0, 18.0, 18.0, 18.0],
        'MaxPrice': [22.0, 22.0, 22.0, 22.0, 22.0]
    })


@pytest.fixture
def sample_csv_data(tmp_path):
    """Create sample CSV file for testing."""
    df = pd.DataFrame({
        'VendorNumber': [1, 2, 3],
        'VendorName': ['Test A', 'Test B', 'Test C'],
        'Amount': [100, 200, 300]
    })
    
    filepath = tmp_path / "test_data.csv"
    df.to_csv(filepath, index=False)
    
    return filepath


@pytest.fixture
def sample_config(tmp_path):
    """Create sample configuration file."""
    config_content = """
project:
  name: "Test Project"
  version: "1.0.0"

database:
  type: sqlite
  path: "{db_path}"

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  files:
    - test_data.csv

analysis:
  min_profit_margin: 0
  top_vendors_count: 10

logging:
  level: "INFO"
  file: "logs/test.log"
"""
    db_path = (tmp_path / "test.db").as_posix()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content.format(db_path=db_path))
    
    return config_path


@pytest.fixture
def temp_database(tmp_path):
    """Create temporary SQLite database."""
    from sqlalchemy import create_engine
    
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    
    return engine, db_path
