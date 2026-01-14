"""
Test Data Transformer
=====================

Unit tests for the DataTransformer class.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.transformer import DataTransformer


class TestDataTransformerCleanVendorName:
    """Tests for vendor name cleaning."""
    
    def test_clean_normal_name(self, sample_vendor_data):
        """Test cleaning a normal vendor name."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer._clean_vendor_name("ABC Company")
        assert result == "Abc Company"
        
    def test_clean_name_with_special_chars(self, sample_vendor_data):
        """Test removing special characters from name."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer._clean_vendor_name("Company@#$%123")
        assert "@" not in result
        assert "#" not in result
        assert "123" in result
        
    def test_clean_name_with_none(self, sample_vendor_data):
        """Test handling None values."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer._clean_vendor_name(None)
        assert result == "Unknown"
        
    def test_clean_name_with_empty_string(self, sample_vendor_data):
        """Test handling empty strings."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer._clean_vendor_name("")
        assert result == "Unknown"
        
    def test_clean_name_preserves_ampersand(self, sample_vendor_data):
        """Test that & is preserved in names."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer._clean_vendor_name("Johnson & Johnson")
        assert "&" in result


class TestDataTransformerDerivedMetrics:
    """Tests for derived metrics calculation."""
    
    def test_gross_profit_calculation(self, sample_vendor_data):
        """Test gross profit is calculated correctly."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer._calculate_derived_metrics(sample_vendor_data)
        
        assert 'GrossProfit' in result.columns
        # First vendor: 12000 - 10000 = 2000
        assert result.loc[0, 'GrossProfit'] == 2000
        
    def test_profit_margin_calculation(self, sample_vendor_data):
        """Test profit margin percentage is calculated correctly."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer._calculate_derived_metrics(sample_vendor_data)
        
        assert 'ProfitMargin' in result.columns
        # First vendor: (2000 / 12000) * 100 = 16.67%
        expected = (2000 / 12000) * 100
        assert abs(result.loc[0, 'ProfitMargin'] - expected) < 0.01
        
    def test_stock_turnover_calculation(self, sample_vendor_data):
        """Test stock turnover ratio is calculated correctly."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer._calculate_derived_metrics(sample_vendor_data)
        
        assert 'StockTurnover' in result.columns
        # First vendor: 450 / ((100 + 150) / 2) = 450 / 125 = 3.6
        expected = 450 / 125
        assert abs(result.loc[0, 'StockTurnover'] - expected) < 0.01
        
    def test_inventory_change_calculation(self, sample_vendor_data):
        """Test inventory change is calculated correctly."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer._calculate_derived_metrics(sample_vendor_data)
        
        assert 'InventoryChange' in result.columns
        # First vendor: 150 - 100 = 50
        assert result.loc[0, 'InventoryChange'] == 50
        
    def test_price_spread_calculation(self, sample_vendor_data):
        """Test price spread is calculated correctly."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer._calculate_derived_metrics(sample_vendor_data)
        
        assert 'PriceSpread' in result.columns
        # All vendors: 22 - 18 = 4
        assert result.loc[0, 'PriceSpread'] == 4.0


class TestDataTransformerCleanData:
    """Tests for complete data cleaning pipeline."""
    
    def test_clean_data_removes_duplicates(self, sample_vendor_data):
        """Test that duplicate vendors are removed."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        # Add duplicate
        df_with_dupe = pd.concat([sample_vendor_data, sample_vendor_data.iloc[[0]]])
        
        result = transformer.clean_data(df_with_dupe)
        
        assert len(result) == len(sample_vendor_data)
        
    def test_clean_data_fills_na(self, sample_vendor_data):
        """Test that NA values are filled with 0."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        # Add NA value
        df_with_na = sample_vendor_data.copy()
        df_with_na.loc[0, 'TotalSalesDollars'] = np.nan
        
        result = transformer.clean_data(df_with_na)
        
        assert not result['TotalSalesDollars'].isna().any()
        
    def test_clean_data_cleans_vendor_names(self, sample_vendor_data):
        """Test that vendor names are cleaned."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer.clean_data(sample_vendor_data)
        
        # Vendor C!! should be cleaned to Vendor C
        vendor_c_name = result[result['VendorNumber'] == 3]['VendorName'].iloc[0]
        assert '!' not in vendor_c_name


class TestDataTransformerRemoveOutliers:
    """Tests for outlier removal."""
    
    def test_remove_outliers_basic(self):
        """Test basic outlier removal."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        df = pd.DataFrame({
            'VendorNumber': [1, 2, 3, 4, 5],
            'Value': [10, 12, 11, 100, 13]  # 100 is an outlier
        })
        
        result = transformer.remove_outliers(df, 'Value', n_std=2)
        
        assert len(result) < len(df)
        assert 100 not in result['Value'].values
        
    def test_remove_outliers_missing_column(self):
        """Test handling of missing column."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        df = pd.DataFrame({'VendorNumber': [1, 2, 3]})
        
        result = transformer.remove_outliers(df, 'NonExistent', n_std=2)
        
        assert len(result) == len(df)


class TestDataTransformerGetTopVendors:
    """Tests for top vendors retrieval."""
    
    def test_get_top_vendors(self, sample_vendor_data):
        """Test getting top N vendors by metric."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer.get_top_vendors(
            sample_vendor_data, 
            metric='TotalSalesDollars', 
            n=3
        )
        
        assert len(result) == 3
        assert result.iloc[0]['TotalSalesDollars'] >= result.iloc[1]['TotalSalesDollars']
        
    def test_get_top_vendors_missing_metric(self, sample_vendor_data):
        """Test with missing metric column."""
        transformer = DataTransformer.__new__(DataTransformer)
        
        result = transformer.get_top_vendors(
            sample_vendor_data, 
            metric='NonExistent', 
            n=3
        )
        
        # Should return first n rows when metric not found
        assert len(result) == 3
