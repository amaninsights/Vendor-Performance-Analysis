# Vendor Performance Analysis Project

A comprehensive data analytics solution for evaluating vendor performance in retail operations. This project provides end-to-end data processing, analysis, and visualization capabilities to help businesses make informed decisions about their vendor relationships.

## 🎯 Project Overview

This system analyzes vendor performance across multiple dimensions including sales volume, profitability, inventory turnover, and operational costs. It processes large datasets from various sources to generate actionable insights through automated reporting and interactive dashboards.

## ✨ Key Features

- **Automated Data Ingestion**: Seamlessly loads CSV files into a centralized SQLite database
- **Advanced Analytics**: Calculates key performance metrics including profit margins, stock turnover, and sales ratios
- **Comprehensive Reporting**: Generates detailed vendor summaries with financial and operational insights
- **Interactive Visualizations**: Jupyter notebook with statistical analysis and data visualizations
- **Business Intelligence Dashboard**: Power BI dashboard for executive-level reporting
- **Robust Logging**: Complete audit trail of all data processing operations

## 📊 Key Metrics Analyzed

- **Gross Profit & Profit Margins**: Revenue vs. cost analysis
- **Stock Turnover Rates**: Inventory efficiency metrics
- **Sales-to-Purchase Ratios**: Vendor relationship efficiency
- **Freight Cost Analysis**: Logistics cost optimization
- **Volume & Price Analysis**: Purchase price vs. actual price comparisons

## 🏗️ Project Structure

```
Vendor Performance Analysis Project/
├── data/                          # Raw data files (CSV format)
│   ├── purchases.csv             # Purchase transactions (361MB)
│   ├── sales.csv                 # Sales transactions (1.5GB)
│   ├── begin_inventory.csv       # Starting inventory levels
│   ├── end_inventory.csv         # Ending inventory levels
│   ├── purchase_prices.csv       # Price reference data
│   └── vendor_invoice.csv        # Vendor invoice details
├── Dashboard/                     # Power BI dashboard files
│   ├── Dashboard.pbix            # Main Power BI dashboard
│   └── background image.jpg      # Dashboard assets
├── logs/                         # Application logs
│   ├── ingestion_db.log          # Data ingestion logs
│   └── get_vendor_summary.log    # Summary generation logs
├── ingestion_db.py               # Data ingestion pipeline
├── get_vendor_summary.py         # Vendor summary generation
├── Vendor Performance Analysis.ipynb  # Analysis notebook
├── inventory.db                  # SQLite database
├── vendor_sales_summary_export.csv   # Generated summary export
└── README.md                     # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **pandas**: Data manipulation and analysis
- **SQLAlchemy**: Database ORM and connection management
- **SQLite**: Lightweight database for data storage
- **Jupyter Notebook**: Interactive analysis environment
- **Power BI**: Business intelligence and visualization
- **matplotlib/seaborn**: Statistical visualizations
- **scipy**: Statistical analysis and testing

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Power BI Desktop (for dashboard viewing)
- Jupyter Notebook

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Vendor Performance Analysis Project"
   ```

2. **Install required packages**
   ```bash
   pip install pandas sqlalchemy matplotlib seaborn scipy jupyter
   ```

3. **Download data files**
   Due to the large size of the data files (>2GB total), they are hosted separately:
   
   📁 **[Download Data Files from OneDrive](https://1drv.ms/f/c/6df8bb1ee8929b58/EpTN3ndZbLxHipIYZHhMSdAB5fvqIf-ORfZTvdrUqYaQwQ?e=SQ9RN3)**
   
   Download and extract all CSV files to the `data/` directory before proceeding.

### Usage

#### 1. Data Ingestion
Load raw CSV data into the SQLite database:
```bash
python ingestion_db.py
```

#### 2. Generate Vendor Summary
Create comprehensive vendor performance metrics:
```bash
python get_vendor_summary.py
```

#### 3. Interactive Analysis
Launch the Jupyter notebook for detailed analysis:
```bash
jupyter notebook "Vendor Performance Analysis.ipynb"
```

#### 4. View Dashboard
Open `Dashboard/Dashboard.pbix` in Power BI Desktop for executive reporting.

## 📈 Data Processing Pipeline

1. **Ingestion**: Raw CSV files are loaded into SQLite database tables
2. **Transformation**: Complex SQL joins merge purchase, sales, and inventory data
3. **Calculation**: Key performance metrics are computed and derived
4. **Cleaning**: Data quality checks and missing value handling
5. **Export**: Final summary exported for further analysis and reporting

## 🔍 Key SQL Operations

The system performs sophisticated data joins across multiple tables:
- **Purchase Summary**: Aggregates vendor purchase data with pricing information
- **Sales Summary**: Consolidates sales performance by vendor and brand
- **Freight Analysis**: Calculates total freight costs per vendor
- **Performance Metrics**: Derives profit margins, turnover rates, and efficiency ratios

## 📋 Output Files

- `inventory.db`: Complete SQLite database with all processed data
- `vendor_sales_summary_export.csv`: Comprehensive vendor performance summary
- `logs/`: Detailed processing logs for troubleshooting and auditing

## 🔧 Configuration

The system uses configurable logging with different levels:
- **INFO**: General processing information
- **DEBUG**: Detailed execution steps
- **ERROR**: Exception handling and error reporting

Log files are automatically created in the `logs/` directory with timestamps.

## 📊 Sample Insights

The analysis provides insights such as:
- Top-performing vendors by profit margin
- Inventory turnover efficiency by brand
- Freight cost optimization opportunities
- Price variance analysis between purchase and actual prices
- Sales trend analysis and forecasting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or support, please:
- Check the log files in the `logs/` directory for troubleshooting
- Review the Jupyter notebook for detailed analysis examples
- Examine the SQL queries in `get_vendor_summary.py` for data logic

---

**Note**: This project handles large datasets (>2GB total). Ensure adequate system resources for processing.