"""
Vendor Metrics Module
=====================

Provides statistical analysis, KPI computation, and performance scoring
for vendor analytics. Includes correlation analysis, statistical tests,
and comprehensive performance dashboards.

Example Usage:
    >>> from src.analysis.metrics import VendorMetrics
    >>> metrics = VendorMetrics(vendor_df)
    >>> kpis = metrics.calculate_kpis()
    >>> scores = metrics.calculate_performance_scores()
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class KPISummary:
    """Data class for KPI summary statistics."""
    total_vendors: int
    total_revenue: float
    total_profit: float
    avg_profit_margin: float
    avg_stock_turnover: float
    top_performer: str
    bottom_performer: str


class VendorMetrics:
    """
    Comprehensive vendor performance metrics and analysis.
    
    Provides:
    - KPI calculation and aggregation
    - Performance scoring with weighted metrics
    - Statistical analysis (correlation, hypothesis testing)
    - Vendor segmentation and ranking
    
    Attributes:
        df (pd.DataFrame): Vendor data
        config (dict): Analysis configuration
    """
    
    # Default weights for performance scoring
    DEFAULT_WEIGHTS = {
        'revenue': 0.30,
        'profit_margin': 0.25,
        'stock_turnover': 0.20,
        'sales_volume': 0.15,
        'price_stability': 0.10
    }
    
    def __init__(
        self,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize VendorMetrics with vendor data.
        
        Args:
            df: Vendor summary DataFrame
            weights: Optional custom weights for scoring
        """
        self.df = df.copy()
        self.weights = weights or self.DEFAULT_WEIGHTS
        logger.info(f"VendorMetrics initialized with {len(df)} vendors")
        
    def calculate_kpis(self) -> KPISummary:
        """
        Calculate key performance indicators.
        
        Returns:
            KPISummary dataclass with aggregated KPIs
        """
        logger.info("Calculating KPIs")
        
        kpi = KPISummary(
            total_vendors=len(self.df),
            total_revenue=self.df['TotalSalesDollars'].sum() if 'TotalSalesDollars' in self.df.columns else 0,
            total_profit=self.df['GrossProfit'].sum() if 'GrossProfit' in self.df.columns else 0,
            avg_profit_margin=self.df['ProfitMargin'].mean() if 'ProfitMargin' in self.df.columns else 0,
            avg_stock_turnover=self.df['StockTurnover'].mean() if 'StockTurnover' in self.df.columns else 0,
            top_performer=self._get_top_performer(),
            bottom_performer=self._get_bottom_performer()
        )
        
        logger.info(f"KPIs calculated: Revenue=${kpi.total_revenue:,.2f}, Profit=${kpi.total_profit:,.2f}")
        return kpi
        
    def _get_top_performer(self) -> str:
        """Get name of top performing vendor."""
        if 'GrossProfit' in self.df.columns and 'VendorName' in self.df.columns:
            idx = self.df['GrossProfit'].idxmax()
            return str(self.df.loc[idx, 'VendorName'])
        return "N/A"
        
    def _get_bottom_performer(self) -> str:
        """Get name of bottom performing vendor."""
        if 'GrossProfit' in self.df.columns and 'VendorName' in self.df.columns:
            idx = self.df['GrossProfit'].idxmin()
            return str(self.df.loc[idx, 'VendorName'])
        return "N/A"
        
    def calculate_performance_scores(self) -> pd.DataFrame:
        """
        Calculate weighted performance scores for each vendor.
        
        Score components:
        - Revenue Score (30%): Normalized total sales
        - Profit Margin Score (25%): Profit margin percentile
        - Stock Turnover Score (20%): Inventory efficiency
        - Sales Volume Score (15%): Total units sold
        - Price Stability Score (10%): Inverse of price spread
        
        Returns:
            DataFrame with performance scores and overall ranking
        """
        logger.info("Calculating performance scores")
        
        df = self.df.copy()
        
        # Calculate component scores (normalized 0-100)
        df['RevenueScore'] = self._normalize_score(df, 'TotalSalesDollars')
        df['ProfitMarginScore'] = self._normalize_score(df, 'ProfitMargin')
        df['TurnoverScore'] = self._normalize_score(df, 'StockTurnover')
        df['VolumeScore'] = self._normalize_score(df, 'TotalSalesQty')
        df['PriceStabilityScore'] = self._normalize_score(df, 'PriceSpread', inverse=True)
        
        # Calculate weighted overall score
        df['OverallScore'] = (
            df['RevenueScore'] * self.weights['revenue'] +
            df['ProfitMarginScore'] * self.weights['profit_margin'] +
            df['TurnoverScore'] * self.weights['stock_turnover'] +
            df['VolumeScore'] * self.weights['sales_volume'] +
            df['PriceStabilityScore'] * self.weights['price_stability']
        )
        
        # Add ranking
        df['Rank'] = df['OverallScore'].rank(ascending=False).astype(int)
        
        logger.info("Performance scores calculated")
        return df.sort_values('Rank')
        
    def _normalize_score(
        self,
        df: pd.DataFrame,
        column: str,
        inverse: bool = False
    ) -> pd.Series:
        """
        Normalize column to 0-100 scale.
        
        Args:
            df: Input DataFrame
            column: Column to normalize
            inverse: If True, lower values get higher scores
            
        Returns:
            Normalized score series
        """
        if column not in df.columns:
            return pd.Series([50] * len(df))  # Default neutral score
            
        col = df[column].fillna(0)
        min_val = col.min()
        max_val = col.max()
        
        if max_val == min_val:
            return pd.Series([50] * len(df))
            
        normalized = (col - min_val) / (max_val - min_val) * 100
        
        if inverse:
            normalized = 100 - normalized
            
        return normalized
        
    def segment_vendors(self, n_segments: int = 4) -> pd.DataFrame:
        """
        Segment vendors into performance tiers.
        
        Args:
            n_segments: Number of segments (default 4: A, B, C, D)
            
        Returns:
            DataFrame with segment labels
        """
        logger.info(f"Segmenting vendors into {n_segments} tiers")
        
        df = self.calculate_performance_scores()
        
        # Create segment labels
        labels = ['D', 'C', 'B', 'A'] if n_segments == 4 else [f'Tier_{i}' for i in range(n_segments, 0, -1)]
        
        df['Segment'] = pd.qcut(
            df['OverallScore'],
            q=n_segments,
            labels=labels
        )
        
        segment_counts = df['Segment'].value_counts()
        logger.info(f"Segmentation complete: {segment_counts.to_dict()}")
        
        return df
        
    def correlation_analysis(self) -> pd.DataFrame:
        """
        Perform correlation analysis on numeric columns.
        
        Returns:
            Correlation matrix as DataFrame
        """
        logger.info("Performing correlation analysis")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        return correlation_matrix
        
    def get_strong_correlations(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of variables with strong correlations.
        
        Args:
            threshold: Correlation threshold (absolute value)
            
        Returns:
            List of tuples (col1, col2, correlation)
        """
        corr_matrix = self.correlation_analysis()
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                
                if abs(corr) >= threshold:
                    strong_corrs.append((col1, col2, corr))
                    
        logger.info(f"Found {len(strong_corrs)} strong correlations (|r| >= {threshold})")
        return sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)
        
    def statistical_tests(self) -> Dict[str, Dict]:
        """
        Perform statistical tests on key metrics.
        
        Tests performed:
        - Shapiro-Wilk test for normality
        - Descriptive statistics
        
        Returns:
            Dictionary of test results
        """
        logger.info("Performing statistical tests")
        
        results = {}
        test_columns = ['TotalSalesDollars', 'ProfitMargin', 'StockTurnover']
        
        for col in test_columns:
            if col not in self.df.columns:
                continue
                
            data = self.df[col].dropna()
            
            # Shapiro-Wilk test for normality (sample if too large)
            sample = data.sample(min(5000, len(data))) if len(data) > 5000 else data
            shapiro_stat, shapiro_p = stats.shapiro(sample)
            
            results[col] = {
                'mean': data.mean(),
                'std': data.std(),
                'median': data.median(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
            
        return results
        
    def profit_analysis(self) -> Dict[str, any]:
        """
        Detailed profit analysis across vendors.
        
        Returns:
            Dictionary with profit analysis results
        """
        if 'GrossProfit' not in self.df.columns:
            return {'error': 'GrossProfit column not found'}
            
        df = self.df
        
        profitable = df[df['GrossProfit'] > 0]
        unprofitable = df[df['GrossProfit'] <= 0]
        
        analysis = {
            'profitable_vendors': len(profitable),
            'unprofitable_vendors': len(unprofitable),
            'profitable_pct': len(profitable) / len(df) * 100,
            'total_profit': df['GrossProfit'].sum(),
            'avg_profit': df['GrossProfit'].mean(),
            'max_profit': df['GrossProfit'].max(),
            'min_profit': df['GrossProfit'].min(),
            'profit_std': df['GrossProfit'].std()
        }
        
        logger.info(f"Profit analysis: {analysis['profitable_pct']:.1f}% vendors profitable")
        return analysis
        
    def get_summary_report(self) -> str:
        """
        Generate text summary report of all metrics.
        
        Returns:
            Formatted string report
        """
        kpis = self.calculate_kpis()
        profit = self.profit_analysis()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║               VENDOR PERFORMANCE SUMMARY REPORT              ║
╠══════════════════════════════════════════════════════════════╣
║ OVERVIEW                                                     ║
║   Total Vendors Analyzed: {kpis.total_vendors:>33,}║
║   Total Revenue: ${kpis.total_revenue:>38,.2f}║
║   Total Gross Profit: ${kpis.total_profit:>34,.2f}║
╠══════════════════════════════════════════════════════════════╣
║ KEY METRICS                                                  ║
║   Average Profit Margin: {kpis.avg_profit_margin:>32.2f}%║
║   Average Stock Turnover: {kpis.avg_stock_turnover:>32.2f}x║
╠══════════════════════════════════════════════════════════════╣
║ TOP PERFORMERS                                               ║
║   Best: {kpis.top_performer[:50]:>50}║
║   Needs Improvement: {kpis.bottom_performer[:36]:>36}║
╠══════════════════════════════════════════════════════════════╣
║ PROFITABILITY                                                ║
║   Profitable Vendors: {profit.get('profitable_vendors', 0):>35,}║
║   Unprofitable Vendors: {profit.get('unprofitable_vendors', 0):>33,}║
║   Profitability Rate: {profit.get('profitable_pct', 0):>34.1f}%║
╚══════════════════════════════════════════════════════════════╝
"""
        return report


def main():
    """Example usage of VendorMetrics."""
    # This would typically use real data
    from src.data.transformer import DataTransformer
    
    transformer = DataTransformer()
    df = transformer.create_vendor_summary()
    df = transformer.clean_data(df)
    
    metrics = VendorMetrics(df)
    
    # Print summary report
    print(metrics.get_summary_report())
    
    # Get performance scores
    scored = metrics.calculate_performance_scores()
    print("\n🏆 Top 5 Vendors by Performance Score:")
    print(scored[['VendorName', 'OverallScore', 'Rank']].head())
    

if __name__ == "__main__":
    main()
