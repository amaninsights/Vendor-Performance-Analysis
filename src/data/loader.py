"""
Data Loader Module
==================

Provides utilities for loading raw data files and ingesting into SQLite database.
Supports multiple file formats with comprehensive error handling and logging.

Example Usage:
    >>> from src.data.loader import DataLoader
    >>> loader = DataLoader("config/config.yaml")
    >>> loader.ingest_all()
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """
    Production-grade data loader for ingesting CSV files into SQLite.
    
    Attributes:
        config (dict): Configuration loaded from YAML file
        engine (Engine): SQLAlchemy database engine
        
    Example:
        >>> loader = DataLoader("config/config.yaml")
        >>> loader.load_csv("data/raw/sales.csv")
        >>> loader.ingest_all()
    """
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.parquet', '.json']
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.engine = self._create_engine()
        logger.info(f"DataLoader initialized with config: {config_path}")
        
    def _load_config(self, config_path: str) -> dict:
        """Load and validate configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.debug(f"Configuration loaded: {config.get('project', {}).get('name')}")
        return config
        
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine from configuration."""
        db_path = self.config.get('database', {}).get('path', 'data/processed/inventory.db')
        
        # Ensure directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        logger.info(f"Database engine created: {db_path}")
        return engine
        
    def load_csv(
        self,
        filepath: Union[str, Path],
        parse_dates: Optional[List[str]] = None,
        dtype: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame with optional type hints.
        
        Args:
            filepath: Path to CSV file
            parse_dates: List of columns to parse as dates
            dtype: Dictionary of column data types
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.ParserError: If CSV parsing fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        logger.info(f"Loading CSV: {filepath}")
        
        df = pd.read_csv(
            filepath,
            parse_dates=parse_dates,
            dtype=dtype,
            **kwargs
        )
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns from {filepath.name}")
        return df
        
    def ingest_to_db(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = 'replace',
        index: bool = False
    ) -> int:
        """
        Ingest DataFrame into SQLite database table.
        
        Args:
            df: DataFrame to ingest
            table_name: Target table name
            if_exists: How to handle existing table ('replace', 'append', 'fail')
            index: Whether to include DataFrame index
            
        Returns:
            Number of rows inserted
        """
        logger.info(f"Ingesting {len(df):,} rows into table: {table_name}")
        
        rows_affected = df.to_sql(
            table_name,
            self.engine,
            if_exists=if_exists,
            index=index
        )
        
        logger.info(f"Successfully ingested {rows_affected or len(df):,} rows into {table_name}")
        return rows_affected or len(df)
        
    def ingest_all(self, data_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Ingest all configured data files into database.
        
        Args:
            data_dir: Optional override for data directory
            
        Returns:
            Dictionary mapping table names to row counts
        """
        data_dir = data_dir or self.config.get('data', {}).get('raw_dir', 'data/raw')
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return {}
            
        results = {}
        files = self.config.get('data', {}).get('files', [])
        
        for filename in files:
            filepath = data_path / filename
            if filepath.exists():
                table_name = filepath.stem  # Use filename without extension
                df = self.load_csv(filepath)
                rows = self.ingest_to_db(df, table_name)
                results[table_name] = rows
            else:
                logger.warning(f"Configured file not found: {filepath}")
                
        logger.info(f"Ingestion complete. Tables created: {list(results.keys())}")
        return results
        
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            sql: SQL query string
            
        Returns:
            Query results as DataFrame
        """
        logger.debug(f"Executing query: {sql[:100]}...")
        return pd.read_sql(sql, self.engine)
        
    def execute(self, sql: str) -> None:
        """
        Execute SQL statement (for DDL/DML operations).
        
        Args:
            sql: SQL statement to execute
        """
        with self.engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
        logger.info("SQL statement executed successfully")
        
    def get_table_info(self) -> pd.DataFrame:
        """Get list of all tables in the database."""
        sql = "SELECT name FROM sqlite_master WHERE type='table'"
        return self.query(sql)
        
    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Get schema information for a specific table."""
        sql = f"PRAGMA table_info({table_name})"
        return self.query(sql)


def main():
    """Example usage of DataLoader."""
    loader = DataLoader()
    
    # Ingest all data files
    results = loader.ingest_all()
    
    # Print summary
    print("\n📊 Ingestion Summary:")
    print("-" * 40)
    for table, rows in results.items():
        print(f"  {table}: {rows:,} rows")
    print("-" * 40)
    print(f"  Total tables: {len(results)}")
    

if __name__ == "__main__":
    main()
