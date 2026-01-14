"""
Test Vendor Metrics
===================

Unit tests for the VendorMetrics class.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.metrics import VendorMetrics, KPISummary


class TestVendorMetricsKPIs:
    """Tests for KPI calculations."""
    
    def test_calculate_kpis(self, sample_vendor_data):
        """Test KPI calculation returns correct values."""
        # Add derived metrics that would be added by transformer
        df = sample_vendor_data.copy()
        df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
        df['ProfitMargin'] = (df['GrossProfit'] / df['TotalSalesDollars']) * 100
        df['StockTurnover'] = df['TotalSalesQty'] / ((df['BeginInventory'] + df['EndInventory']) / 2)
        
        metrics = VendorMetrics(df)
        kpis = metrics.calculate_kpis()
        
        assert isinstance(kpis, KPISummary)
        assert kpis.total_vendors == 5
        assert kpis.total_revenue == df['TotalSalesDollars'].sum()
        assert kpis.total_profit == df['GrossProfit'].sum()
        
    def test_top_performer_identification(self, sample_vendor_data):
        """Test correct identification of top performer."""
        df = sample_vendor_data.copy()
        df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
        
        metrics = VendorMetrics(df)
        kpis = metrics.calculate_kpis()
        
        # Vendor D has highest gross profit (35000 - 30000 = 5000)
        assert kpis.top_performer == 'Vendor D'


class TestVendorMetricsPerformanceScores:
    """Tests for performance scoring."""
    
    def test_performance_scores_normalization(self, sample_vendor_data):
        """Test that scores are normalized to 0-100."""
        df = sample_vendor_data.copy()
        df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
        df['ProfitMargin'] = (df['GrossProfit'] / df['TotalSalesDollars']) * 100
        df['StockTurnover'] = df['TotalSalesQty'] / ((df['BeginInventory'] + df['EndInventory']) / 2)
        df['PriceSpread'] = df['MaxPrice'] - df['MinPrice']
        
        metrics = VendorMetrics(df)
        result = metrics.calculate_performance_scores()
        
        assert 'OverallScore' in result.columns
        assert result['OverallScore'].min() >= 0
        assert result['OverallScore'].max() <= 100
        
    def test_performance_scores_ranking(self, sample_vendor_data):
        """Test that vendors are ranked correctly."""
        df = sample_vendor_data.copy()
        df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
        df['ProfitMargin'] = (df['GrossProfit'] / df['TotalSalesDollars']) * 100
        df['StockTurnover'] = df['TotalSalesQty'] / ((df['BeginInventory'] + df['EndInventory']) / 2)
        df['PriceSpread'] = df['MaxPrice'] - df['MinPrice']
        
        metrics = VendorMetrics(df)
        result = metrics.calculate_performance_scores()
        
        assert 'Rank' in result.columns
        assert result['Rank'].min() == 1
        assert result['Rank'].max() == len(df)


class TestVendorMetricsSegmentation:
    """Tests for vendor segmentation."""
    
    def test_segment_vendors(self, sample_vendor_data):
        """Test vendor segmentation into tiers."""
        df = sample_vendor_data.copy()
        df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
        df['ProfitMargin'] = (df['GrossProfit'] / df['TotalSalesDollars']) * 100
        df['StockTurnover'] = df['TotalSalesQty'] / ((df['BeginInventory'] + df['EndInventory']) / 2)
        df['PriceSpread'] = df['MaxPrice'] - df['MinPrice']
        
        metrics = VendorMetrics(df)
        result = metrics.segment_vendors(n_segments=4)
        
        assert 'Segment' in result.columns
        assert set(result['Segment'].unique()).issubset({'A', 'B', 'C', 'D'})


class TestVendorMetricsCorrelation:
    """Tests for correlation analysis."""
    
    def test_correlation_analysis(self, sample_vendor_data):
        """Test correlation matrix generation."""
        metrics = VendorMetrics(sample_vendor_data)
        corr = metrics.correlation_analysis()
        
        assert isinstance(corr, pd.DataFrame)
        # Correlation matrix should be square
        assert corr.shape[0] == corr.shape[1]
        # Diagonal should be 1.0
        assert all(abs(corr.iloc[i, i] - 1.0) < 0.01 for i in range(len(corr)))
        
    def test_strong_correlations(self, sample_vendor_data):
        """Test identification of strong correlations."""
        metrics = VendorMetrics(sample_vendor_data)
        strong = metrics.get_strong_correlations(threshold=0.7)
        
        assert isinstance(strong, list)
        # All returned correlations should be >= threshold
        for col1, col2, corr in strong:
            assert abs(corr) >= 0.7


class TestVendorMetricsProfitAnalysis:
    """Tests for profit analysis."""
    
    def test_profit_analysis(self, sample_vendor_data):
        """Test profit analysis results."""
        df = sample_vendor_data.copy()
        df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
        
        metrics = VendorMetrics(df)
        analysis = metrics.profit_analysis()
        
        assert 'profitable_vendors' in analysis
        assert 'unprofitable_vendors' in analysis
        assert 'total_profit' in analysis
        assert analysis['profitable_vendors'] + analysis['unprofitable_vendors'] == len(df)


class TestVendorMetricsSummaryReport:
    """Tests for summary report generation."""
    
    def test_summary_report_format(self, sample_vendor_data):
        """Test summary report is formatted correctly."""
        df = sample_vendor_data.copy()
        df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
        df['ProfitMargin'] = (df['GrossProfit'] / df['TotalSalesDollars']) * 100
        df['StockTurnover'] = df['TotalSalesQty'] / ((df['BeginInventory'] + df['EndInventory']) / 2)
        
        metrics = VendorMetrics(df)
        report = metrics.get_summary_report()
        
        assert isinstance(report, str)
        assert 'VENDOR PERFORMANCE SUMMARY REPORT' in report
        assert 'Total Vendors Analyzed' in report
        assert 'Total Revenue' in report
