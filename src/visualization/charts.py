"""
Vendor Charts Module
====================

Provides publication-ready visualizations for vendor analytics.
Includes bar charts, scatter plots, heatmaps, and executive dashboards.

Example Usage:
    >>> from src.visualization.charts import VendorCharts
    >>> charts = VendorCharts(vendor_df)
    >>> charts.plot_top_vendors()
    >>> charts.save_all_charts("reports/figures/")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VendorCharts:
    """
    Publication-ready vendor analytics visualizations.
    
    Provides:
    - Bar charts for vendor rankings
    - Scatter plots for correlation analysis
    - Heatmaps for correlation matrices
    - Pie charts for market share
    - Executive dashboards
    
    Attributes:
        df (pd.DataFrame): Vendor data
        style (str): Matplotlib style
        figsize (tuple): Default figure size
        dpi (int): Figure DPI for exports
    """
    
    # Color palettes
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#48A14D',
        'warning': '#F18F01',
        'danger': '#C73E1D',
        'gradient': ['#2E86AB', '#5BA3C6', '#88C0E0', '#B5DDF9']
    }
    
    def __init__(
        self,
        df: pd.DataFrame,
        style: str = 'seaborn-v0_8-whitegrid',
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 300
    ):
        """
        Initialize VendorCharts with data and styling.
        
        Args:
            df: Vendor summary DataFrame
            style: Matplotlib style name
            figsize: Default figure size (width, height)
            dpi: DPI for saved figures
        """
        self.df = df.copy()
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')
            
        # Set default parameters
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        logger.info(f"VendorCharts initialized with {len(df)} vendors")
        
    def plot_top_vendors(
        self,
        n: int = 10,
        metric: str = 'TotalSalesDollars',
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Create horizontal bar chart of top vendors by metric.
        
        Args:
            n: Number of vendors to show
            metric: Column to rank by
            title: Custom title (auto-generated if None)
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            Matplotlib Figure object
        """
        if metric not in self.df.columns:
            logger.warning(f"Metric {metric} not found")
            return plt.figure()
            
        # Get top N vendors
        top = self.df.nlargest(n, metric)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
            
        # Create bar chart
        colors = sns.color_palette('husl', n)
        bars = ax.barh(top['VendorName'], top[metric], color=colors)
        
        # Add value labels
        for bar, value in zip(bars, top[metric]):
            ax.text(
                value, bar.get_y() + bar.get_height()/2,
                f'  ${value:,.0f}' if 'Dollar' in metric else f'  {value:,.0f}',
                va='center', fontsize=9
            )
            
        # Styling
        ax.set_xlabel(self._format_metric_name(metric))
        ax.set_ylabel('')
        ax.set_title(title or f'Top {n} Vendors by {self._format_metric_name(metric)}')
        ax.invert_yaxis()  # Highest at top
        
        plt.tight_layout()
        logger.info(f"Created top vendors chart: {metric}")
        return fig
        
    def plot_profit_distribution(
        self,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Create histogram of profit margin distribution.
        
        Args:
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            Matplotlib Figure object
        """
        if 'ProfitMargin' not in self.df.columns:
            logger.warning("ProfitMargin column not found")
            return plt.figure()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
            
        # Create histogram with KDE
        data = self.df['ProfitMargin'].dropna()
        
        ax.hist(data, bins=30, color=self.COLORS['primary'], 
                alpha=0.7, edgecolor='white')
        
        # Add mean line
        mean_val = data.mean()
        ax.axvline(mean_val, color=self.COLORS['danger'], 
                   linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}%')
        
        # Styling
        ax.set_xlabel('Profit Margin (%)')
        ax.set_ylabel('Number of Vendors')
        ax.set_title('Distribution of Vendor Profit Margins')
        ax.legend()
        
        plt.tight_layout()
        logger.info("Created profit distribution chart")
        return fig
        
    def plot_correlation_heatmap(
        self,
        columns: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Create correlation heatmap for numeric columns.
        
        Args:
            columns: Specific columns to include (all numeric if None)
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            Matplotlib Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.get_figure()
            
        # Select columns
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:10]
        else:
            numeric_cols = [c for c in columns if c in self.df.columns]
            
        # Calculate correlation
        corr = self.df[numeric_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, ax=ax,
            square=True, linewidths=0.5
        )
        
        ax.set_title('Correlation Matrix')
        
        plt.tight_layout()
        logger.info("Created correlation heatmap")
        return fig
        
    def plot_sales_vs_profit(
        self,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Create scatter plot of sales vs profit with trend line.
        
        Args:
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            Matplotlib Figure object
        """
        required_cols = ['TotalSalesDollars', 'GrossProfit']
        if not all(c in self.df.columns for c in required_cols):
            logger.warning("Required columns not found")
            return plt.figure()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
            
        x = self.df['TotalSalesDollars']
        y = self.df['GrossProfit']
        
        # Scatter plot with size based on volume
        size = self.df.get('TotalSalesQty', pd.Series([100]*len(self.df)))
        size_normalized = (size / size.max() * 200) + 20
        
        scatter = ax.scatter(
            x, y, s=size_normalized, alpha=0.6,
            c=self.df.get('ProfitMargin', 'blue'),
            cmap='RdYlGn', edgecolors='white', linewidth=0.5
        )
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x.sort_values(), p(x.sort_values()), 
                color=self.COLORS['danger'], linestyle='--', 
                linewidth=2, label='Trend')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Profit Margin (%)')
        
        # Styling
        ax.set_xlabel('Total Sales ($)')
        ax.set_ylabel('Gross Profit ($)')
        ax.set_title('Sales vs Profit Analysis')
        ax.legend()
        
        # Format axes as currency
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        logger.info("Created sales vs profit scatter plot")
        return fig
        
    def plot_market_share(
        self,
        n: int = 8,
        metric: str = 'TotalSalesDollars',
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Create pie chart of market share by vendor.
        
        Args:
            n: Number of vendors to show individually
            metric: Column to use for share calculation
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            Matplotlib Figure object
        """
        if metric not in self.df.columns:
            logger.warning(f"Metric {metric} not found")
            return plt.figure()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.get_figure()
            
        # Get top N and group others
        df_sorted = self.df.nlargest(n, metric)
        top_values = df_sorted[metric].values
        top_labels = df_sorted['VendorName'].values
        
        other_value = self.df[~self.df['VendorName'].isin(top_labels)][metric].sum()
        
        values = list(top_values) + [other_value]
        labels = list(top_labels) + ['Others']
        
        # Create pie chart
        colors = sns.color_palette('husl', len(values))
        explode = [0.02] * len(values)
        
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=colors, explode=explode,
            pctdistance=0.75, startangle=90
        )
        
        # Styling
        plt.setp(autotexts, size=9, weight='bold')
        plt.setp(texts, size=10)
        ax.set_title(f'Market Share by Vendor ({self._format_metric_name(metric)})')
        
        plt.tight_layout()
        logger.info("Created market share pie chart")
        return fig
        
    def plot_performance_dashboard(self) -> plt.Figure:
        """
        Create comprehensive 4-panel dashboard.
        
        Returns:
            Matplotlib Figure with 4 subplots
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Vendor Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Panel 1: Top Vendors
        self.plot_top_vendors(n=8, ax=axes[0, 0])
        
        # Panel 2: Profit Distribution
        self.plot_profit_distribution(ax=axes[0, 1])
        
        # Panel 3: Sales vs Profit
        self.plot_sales_vs_profit(ax=axes[1, 0])
        
        # Panel 4: Market Share
        self.plot_market_share(n=6, ax=axes[1, 1])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        logger.info("Created performance dashboard")
        return fig
        
    def plot_segment_analysis(
        self,
        segment_col: str = 'Segment',
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Create bar chart comparing segments.
        
        Args:
            segment_col: Column containing segment labels
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            Matplotlib Figure object
        """
        if segment_col not in self.df.columns:
            logger.warning(f"Segment column {segment_col} not found")
            return plt.figure()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
            
        # Aggregate by segment
        segment_summary = self.df.groupby(segment_col).agg({
            'TotalSalesDollars': 'sum',
            'GrossProfit': 'sum',
            'VendorNumber': 'count'
        }).rename(columns={'VendorNumber': 'VendorCount'})
        
        # Create grouped bar chart
        x = np.arange(len(segment_summary))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, segment_summary['TotalSalesDollars'] / 1e6, 
                       width, label='Sales ($M)', color=self.COLORS['primary'])
        bars2 = ax.bar(x + width/2, segment_summary['GrossProfit'] / 1e6, 
                       width, label='Profit ($M)', color=self.COLORS['success'])
        
        # Styling
        ax.set_xlabel('Segment')
        ax.set_ylabel('Amount ($ Millions)')
        ax.set_title('Performance by Segment')
        ax.set_xticks(x)
        ax.set_xticklabels(segment_summary.index)
        ax.legend()
        
        # Add count labels
        for i, (idx, row) in enumerate(segment_summary.iterrows()):
            ax.annotate(f'n={row["VendorCount"]:.0f}', 
                       xy=(i, 0), xytext=(0, -20),
                       textcoords='offset points', ha='center',
                       fontsize=9, color='gray')
        
        plt.tight_layout()
        logger.info("Created segment analysis chart")
        return fig
        
    def save_chart(
        self,
        fig: plt.Figure,
        filename: str,
        output_dir: str = 'reports/figures'
    ) -> str:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib Figure to save
            filename: Output filename (without extension)
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{filename}.png"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        
        logger.info(f"Chart saved: {filepath}")
        return str(filepath)
        
    def save_all_charts(self, output_dir: str = 'reports/figures') -> Dict[str, str]:
        """
        Generate and save all charts.
        
        Args:
            output_dir: Output directory for all charts
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        saved = {}
        
        charts = [
            ('top_vendors', self.plot_top_vendors),
            ('profit_distribution', self.plot_profit_distribution),
            ('correlation_heatmap', self.plot_correlation_heatmap),
            ('sales_vs_profit', self.plot_sales_vs_profit),
            ('market_share', self.plot_market_share),
            ('dashboard', self.plot_performance_dashboard)
        ]
        
        for name, method in charts:
            try:
                fig = method()
                saved[name] = self.save_chart(fig, name, output_dir)
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating {name}: {e}")
                
        logger.info(f"Saved {len(saved)} charts to {output_dir}")
        return saved
        
    def _format_metric_name(self, metric: str) -> str:
        """Convert column name to display name."""
        replacements = {
            'TotalSalesDollars': 'Total Sales ($)',
            'TotalPurchaseDollars': 'Total Purchases ($)',
            'GrossProfit': 'Gross Profit ($)',
            'ProfitMargin': 'Profit Margin (%)',
            'StockTurnover': 'Stock Turnover (x)',
            'TotalSalesQty': 'Sales Volume (units)'
        }
        return replacements.get(metric, metric)


def main():
    """Example usage of VendorCharts."""
    from src.data.transformer import DataTransformer
    
    transformer = DataTransformer()
    df = transformer.create_vendor_summary()
    df = transformer.clean_data(df)
    
    charts = VendorCharts(df)
    saved = charts.save_all_charts()
    
    print("\n📊 Charts Generated:")
    for name, path in saved.items():
        print(f"  ✓ {name}: {path}")
        

if __name__ == "__main__":
    main()
