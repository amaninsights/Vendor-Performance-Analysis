"""
Data Transformer Module
=======================

Provides SQL-based data transformations using CTEs for vendor analytics.
Creates derived metrics, handles data cleaning, and generates summary tables.

Example Usage:
    >>> from src.data.transformer import DataTransformer
    >>> transformer = DataTransformer("config/config.yaml")
    >>> vendor_summary = transformer.create_vendor_summary()
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataTransformer:
    """
    Production-grade data transformer for vendor analytics.
    
    Uses SQL CTEs for complex transformations and provides
    comprehensive data cleaning and metric derivation.
    
    Attributes:
        config (dict): Configuration loaded from YAML
        engine (Engine): SQLAlchemy database engine
    """
    
    # SQL query for vendor summary using Common Table Expressions (CTEs)
    VENDOR_SUMMARY_QUERY = """
    WITH purchases_agg AS (
        SELECT 
            VendorNumber,
            VendorName,
            COUNT(*) as TotalPurchases,
            SUM(Dollars) as TotalPurchaseDollars,
            SUM(Quantity) as TotalPurchaseQty
        FROM purchases
        GROUP BY VendorNumber, VendorName
    ),
    sales_agg AS (
        SELECT 
            VendorNo,
            COUNT(*) as TotalSales,
            SUM(SalesDollars) as TotalSalesDollars,
            SUM(SalesQuantity) as TotalSalesQty,
            AVG(SalesPrice) as AvgSalesPrice
        FROM sales
        GROUP BY VendorNo
    ),
    inventory_begin AS (
        SELECT 
            VendorNumber,
            SUM(onHand) as BeginInventory
        FROM begin_inventory
        GROUP BY VendorNumber
    ),
    inventory_end AS (
        SELECT 
            VendorNumber,
            SUM(onHand) as EndInventory
        FROM end_inventory
        GROUP BY VendorNumber
    ),
    pricing AS (
        SELECT 
            VendorNumber,
            AVG(Price) as AvgPurchasePrice,
            MIN(Price) as MinPrice,
            MAX(Price) as MaxPrice
        FROM purchase_prices
        GROUP BY VendorNumber
    )
    SELECT 
        p.VendorNumber,
        p.VendorName,
        p.TotalPurchases,
        p.TotalPurchaseDollars,
        p.TotalPurchaseQty,
        COALESCE(s.TotalSales, 0) as TotalSales,
        COALESCE(s.TotalSalesDollars, 0) as TotalSalesDollars,
        COALESCE(s.TotalSalesQty, 0) as TotalSalesQty,
        COALESCE(s.AvgSalesPrice, 0) as AvgSalesPrice,
        COALESCE(ib.BeginInventory, 0) as BeginInventory,
        COALESCE(ie.EndInventory, 0) as EndInventory,
        COALESCE(pr.AvgPurchasePrice, 0) as AvgPurchasePrice,
        COALESCE(pr.MinPrice, 0) as MinPrice,
        COALESCE(pr.MaxPrice, 0) as MaxPrice
    FROM purchases_agg p
    LEFT JOIN sales_agg s ON p.VendorNumber = s.VendorNo
    LEFT JOIN inventory_begin ib ON p.VendorNumber = ib.VendorNumber
    LEFT JOIN inventory_end ie ON p.VendorNumber = ie.VendorNumber
    LEFT JOIN pricing pr ON p.VendorNumber = pr.VendorNumber
    ORDER BY p.TotalSalesDollars DESC
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataTransformer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.engine = self._create_engine()
        logger.info(f"DataTransformer initialized")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine from configuration."""
        db_path = self.config.get('database', {}).get('path', 'data/processed/inventory.db')
        return create_engine(f"sqlite:///{db_path}", echo=False)
        
    def create_vendor_summary(self) -> pd.DataFrame:
        """
        Create vendor summary using SQL CTEs.
        
        Returns:
            DataFrame with aggregated vendor metrics
        """
        logger.info("Creating vendor summary with CTE-based aggregations")
        
        try:
            df = pd.read_sql(self.VENDOR_SUMMARY_QUERY, self.engine)
            logger.info(f"Vendor summary created: {len(df)} vendors")
            return df
        except Exception as e:
            logger.error(f"Error creating vendor summary: {e}")
            raise
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive data cleaning transformations.
        
        Cleaning steps:
        1. Clean vendor names (remove special characters, standardize)
        2. Handle missing values
        3. Remove duplicate entries
        4. Calculate derived metrics
        
        Args:
            df: Raw DataFrame to clean
            
        Returns:
            Cleaned DataFrame with derived metrics
        """
        logger.info(f"Cleaning data: {len(df)} rows input")
        original_rows = len(df)
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Step 1: Clean vendor names
        if 'VendorName' in df.columns:
            df['VendorName'] = df['VendorName'].apply(self._clean_vendor_name)
            
        # Step 2: Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Step 3: Remove duplicates based on VendorNumber
        if 'VendorNumber' in df.columns:
            df = df.drop_duplicates(subset=['VendorNumber'], keep='first')
            
        # Step 4: Calculate derived metrics
        df = self._calculate_derived_metrics(df)
        
        logger.info(f"Cleaning complete: {original_rows} → {len(df)} rows")
        return df
        
    def _clean_vendor_name(self, name: str) -> str:
        """
        Clean and standardize vendor name.
        
        Args:
            name: Raw vendor name
            
        Returns:
            Cleaned vendor name
        """
        if pd.isna(name):
            return "Unknown"
            
        name = str(name)
        
        # Remove special characters (keep letters, numbers, spaces)
        name = re.sub(r'[^a-zA-Z0-9\s\-&]', '', name)
        
        # Standardize whitespace
        name = ' '.join(name.split())
        
        # Title case
        name = name.title()
        
        return name.strip() or "Unknown"
        
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived business metrics.
        
        Metrics calculated:
        - GrossProfit: Sales - Purchases
        - ProfitMargin: (GrossProfit / Sales) * 100
        - StockTurnover: Sales / Average Inventory
        - InventoryChange: End - Begin inventory
        
        Args:
            df: DataFrame with base metrics
            
        Returns:
            DataFrame with derived metrics added
        """
        df = df.copy()
        
        # Gross Profit
        if 'TotalSalesDollars' in df.columns and 'TotalPurchaseDollars' in df.columns:
            df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
            
        # Profit Margin (percentage)
        if 'GrossProfit' in df.columns and 'TotalSalesDollars' in df.columns:
            df['ProfitMargin'] = np.where(
                df['TotalSalesDollars'] > 0,
                (df['GrossProfit'] / df['TotalSalesDollars']) * 100,
                0
            )
            
        # Stock Turnover Ratio
        if all(col in df.columns for col in ['TotalSalesQty', 'BeginInventory', 'EndInventory']):
            avg_inventory = (df['BeginInventory'] + df['EndInventory']) / 2
            df['StockTurnover'] = np.where(
                avg_inventory > 0,
                df['TotalSalesQty'] / avg_inventory,
                0
            )
            
        # Inventory Change
        if 'BeginInventory' in df.columns and 'EndInventory' in df.columns:
            df['InventoryChange'] = df['EndInventory'] - df['BeginInventory']
            
        # Price Spread
        if 'MaxPrice' in df.columns and 'MinPrice' in df.columns:
            df['PriceSpread'] = df['MaxPrice'] - df['MinPrice']
            
        logger.info("Derived metrics calculated: GrossProfit, ProfitMargin, StockTurnover, InventoryChange, PriceSpread")
        return df
        
    def remove_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers using standard deviation method.
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            n_std: Number of standard deviations for threshold
            
        Returns:
            DataFrame with outliers removed
        """
        if column not in df.columns:
            logger.warning(f"Column {column} not found, skipping outlier removal")
            return df
            
        mean = df[column].mean()
        std = df[column].std()
        
        lower = mean - (n_std * std)
        upper = mean + (n_std * std)
        
        original_len = len(df)
        df = df[(df[column] >= lower) & (df[column] <= upper)]
        removed = original_len - len(df)
        
        logger.info(f"Removed {removed} outliers from {column} (±{n_std} std)")
        return df
        
    def aggregate_by_category(
        self,
        df: pd.DataFrame,
        group_col: str,
        agg_cols: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Aggregate data by category with custom aggregations.
        
        Args:
            df: Input DataFrame
            group_col: Column to group by
            agg_cols: Dictionary mapping columns to aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        logger.info(f"Aggregating by {group_col}")
        return df.groupby(group_col).agg(agg_cols).reset_index()
        
    def get_top_vendors(
        self,
        df: pd.DataFrame,
        metric: str = 'TotalSalesDollars',
        n: int = 10
    ) -> pd.DataFrame:
        """
        Get top N vendors by specified metric.
        
        Args:
            df: Vendor summary DataFrame
            metric: Column to rank by
            n: Number of top vendors to return
            
        Returns:
            Top N vendors sorted by metric
        """
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found")
            return df.head(n)
            
        return df.nlargest(n, metric)
        
    def save_to_db(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Save transformed DataFrame to database.
        
        Args:
            df: DataFrame to save
            table_name: Target table name
        """
        df.to_sql(table_name, self.engine, if_exists='replace', index=False)
        logger.info(f"Saved {len(df)} rows to table: {table_name}")
        
    def export_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Export DataFrame to CSV file.
        
        Args:
            df: DataFrame to export
            filepath: Output file path
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} rows to: {filepath}")


def main():
    """Example usage of DataTransformer."""
    transformer = DataTransformer()
    
    # Create and clean vendor summary
    df = transformer.create_vendor_summary()
    df = transformer.clean_data(df)
    
    # Get top performers
    top_vendors = transformer.get_top_vendors(df, 'TotalSalesDollars', 10)
    
    print("\n🏆 Top 10 Vendors by Sales:")
    print("-" * 60)
    for _, row in top_vendors.iterrows():
        print(f"  {row['VendorName']}: ${row['TotalSalesDollars']:,.2f}")
    print("-" * 60)
    

if __name__ == "__main__":
    main()
