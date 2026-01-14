#!/usr/bin/env python
"""
Vendor Performance Analytics - Main Pipeline
=============================================

End-to-end data pipeline for vendor performance analysis.

This script orchestrates the complete analytics workflow:
1. Data ingestion from CSV files to SQLite
2. SQL-based transformations using CTEs
3. Data cleaning and metric calculation
4. Statistical analysis and scoring
5. Visualization generation
6. Report export

Usage:
    python main.py                    # Run full pipeline
    python main.py --ingest-only      # Only ingest data
    python main.py --analyze-only     # Only run analysis (skip ingestion)
    python main.py --export-charts    # Generate and save all charts

Author: Aman Saroha
Version: 2.0.0
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.transformer import DataTransformer
from src.analysis.metrics import VendorMetrics
from src.visualization.charts import VendorCharts
from src.utils.logger import setup_logger, PipelineLogger

logger = setup_logger(__name__, log_file="logs/pipeline.log")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vendor Performance Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run complete pipeline
  python main.py --ingest-only      Only load data into database
  python main.py --analyze-only     Skip ingestion, run analysis
  python main.py --export-charts    Generate all visualizations
  python main.py --top 20           Show top 20 vendors (default: 10)
        """
    )
    
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only run data ingestion step"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true", 
        help="Skip ingestion, run analysis on existing data"
    )
    parser.add_argument(
        "--export-charts",
        action="store_true",
        help="Generate and export all charts"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top vendors to display (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for reports and charts"
    )
    
    return parser.parse_args()


def run_ingestion(config_path: str) -> dict:
    """
    Run data ingestion pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with ingestion results
    """
    with PipelineLogger("Data Ingestion") as pl:
        loader = DataLoader(config_path)
        results = loader.ingest_all()
        
        pl.info(f"Ingested {len(results)} tables")
        for table, rows in results.items():
            pl.info(f"  {table}: {rows:,} rows")
            
        return results


def run_transformation(config_path: str):
    """
    Run data transformation pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Transformed and cleaned DataFrame
    """
    with PipelineLogger("Data Transformation") as pl:
        transformer = DataTransformer(config_path)
        
        # Create vendor summary with CTEs
        pl.info("Creating vendor summary using SQL CTEs...")
        df = transformer.create_vendor_summary()
        pl.info(f"Raw summary: {len(df)} vendors")
        
        # Clean data and calculate derived metrics
        pl.info("Cleaning data and calculating derived metrics...")
        df = transformer.clean_data(df)
        pl.info(f"Cleaned data: {len(df)} vendors")
        
        # Save to database
        transformer.save_to_db(df, "vendor_summary")
        pl.info("Saved vendor_summary to database")
        
        return df


def run_analysis(df, n_top: int = 10):
    """
    Run statistical analysis and scoring.
    
    Args:
        df: Vendor DataFrame
        n_top: Number of top vendors to highlight
        
    Returns:
        Tuple of (metrics object, scored DataFrame)
    """
    with PipelineLogger("Statistical Analysis") as pl:
        metrics = VendorMetrics(df)
        
        # Calculate KPIs
        kpis = metrics.calculate_kpis()
        pl.info(f"Total Revenue: ${kpis.total_revenue:,.2f}")
        pl.info(f"Total Profit: ${kpis.total_profit:,.2f}")
        pl.info(f"Avg Profit Margin: {kpis.avg_profit_margin:.2f}%")
        
        # Calculate performance scores
        scored_df = metrics.calculate_performance_scores()
        pl.info("Performance scores calculated")
        
        # Segment vendors
        segmented_df = metrics.segment_vendors()
        segment_counts = segmented_df['Segment'].value_counts()
        pl.info(f"Segmentation: {segment_counts.to_dict()}")
        
        # Correlation analysis
        strong_corr = metrics.get_strong_correlations(threshold=0.7)
        pl.info(f"Found {len(strong_corr)} strong correlations")
        
        return metrics, scored_df


def run_visualization(df, output_dir: str = "reports/figures"):
    """
    Generate all visualizations.
    
    Args:
        df: Vendor DataFrame
        output_dir: Output directory for charts
        
    Returns:
        Dictionary of saved chart paths
    """
    with PipelineLogger("Visualization") as pl:
        charts = VendorCharts(df)
        saved = charts.save_all_charts(output_dir)
        
        pl.info(f"Generated {len(saved)} charts")
        for name, path in saved.items():
            pl.info(f"  {name}: {path}")
            
        return saved


def print_summary(df, metrics, n_top: int = 10):
    """Print executive summary to console."""
    print("\n" + "=" * 70)
    print("                 VENDOR PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Print summary report
    print(metrics.get_summary_report())
    
    # Print top vendors
    print(f"\n🏆 TOP {n_top} VENDORS BY PERFORMANCE SCORE")
    print("-" * 70)
    
    scored = metrics.calculate_performance_scores().head(n_top)
    for i, (_, row) in enumerate(scored.iterrows(), 1):
        print(f"  {i:2}. {row['VendorName'][:40]:<40} "
              f"Score: {row['OverallScore']:.1f}")
    
    print("-" * 70)
    print(f"\n✅ Analysis complete. Check 'reports/' for detailed outputs.\n")


def main():
    """Main entry point for the pipeline."""
    args = parse_args()
    
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("VENDOR PERFORMANCE ANALYTICS PIPELINE")
    logger.info(f"Started: {start_time.isoformat()}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Data Ingestion
        if not args.analyze_only:
            ingestion_results = run_ingestion(args.config)
            
            if args.ingest_only:
                logger.info("Ingestion complete (--ingest-only flag set)")
                return 0
        
        # Step 2: Data Transformation
        df = run_transformation(args.config)
        
        # Step 3: Analysis
        metrics, scored_df = run_analysis(df, args.top)
        
        # Step 4: Visualization
        if args.export_charts:
            chart_paths = run_visualization(df, f"{args.output_dir}/figures")
        
        # Step 5: Export results
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scored data
        scored_df.to_csv(output_path / "vendor_scores.csv", index=False)
        logger.info(f"Saved vendor scores to {output_path / 'vendor_scores.csv'}")
        
        # Print summary
        print_summary(df, metrics, args.top)
        
        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETE")
        logger.info(f"Duration: {duration}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
