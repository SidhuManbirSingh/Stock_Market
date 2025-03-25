# Financial Data Analysis Tool

## Overview
This tool collects, processes, and standardizes financial data from multiple sources. It is designed to help you compare historical financial metrics across companies by aggregating stock prices, dividend data, debt-to-equity ratios, and various key financial indicators.

## Features
- **Data Collection:**  
  - **Yahoo Finance:** Retrieves historical stock prices, dividend payments, and financial statements.
  - **Quandl:** Fetches RBI interest rate data (requires a valid API key).
- **Data Processing:**  
  - Processes up to 12 years of daily stock prices.
  - Computes percentage changes in closing prices.
  - Filters dividend data to only include days with actual payments.
  - Calculates debt-to-equity ratios from balance sheet information.
- **Data Standardization:**  
  - Converts raw financial figures into comparable units (e.g., billions).
  - Computes normalized metrics and financial ratios for cross-company analysis.
- **Output:**  
  - Generates consolidated CSV files for Prices, Dividends, and Other financial data.
  - Produces standardized output for easier comparison and further analysis.

## Prerequisites
- Python 3.6+
- Required packages:  
  - yfinance  
  - pandas  
  - numpy  
  - quandl

## Installation
Clone the repository (if applicable), navigate to the project directory, and install the required packages:
```bash
git clone <REPOSITORY_URL>
cd financial-data-analysis
pip install yfinance pandas numpy quandl
```

## Directory Structure
```
financial-data-analysis/
├── Data/
│   ├── Prices/         # Stock price data
│   ├── Dividends/      # Dividend information
│   └── Others/         # Financial metrics, ratios, and interest rates
├── Ticker/
│   └── ticker.csv      # List of stock tickers to analyze
├── main.py             # Main data collection script
├── standardize.py      # Data standardization and comparison script
└── README.md           # This documentation file
```

## Usage

### Setting Up Your Ticker List
Create a CSV file at `Ticker/ticker.csv` with a header named `Ticker` and list the stock tickers you wish to analyze:
```csv
Ticker
TATAMOTORS.NS
MARUTI.NS
M&M.NS
```

### Running the Data Collection Script
Run the main script to fetch and export data:
```bash
python main.py
```
This will create the following output files:
- `Data/Prices/all_prices.csv`
- `Data/Dividends/all_dividends.csv`
- `Data/Others/all_others.csv`

### Standardizing Data for Comparative Analysis
Use the standardization script to process the consolidated data:
```bash
python Stan1.py
```
This will output a standardized file (e.g., `all_others_standardized.csv`) that normalizes financial metrics across companies for easy comparison.

## Data Collection Process
1. **Stock Prices:** Daily closing prices and percentage changes.
2. **Dividend Data:** Records only the days when dividends are paid.
3. **Financial Metrics:**  
   - Debt-to-equity ratios computed from annual/quarterly balance sheets.
   - Key indicators such as Total Assets, Total Liabilities, EBITDA, and Operating Margin.
4. **Market Data:** RBI interest rate and NIFTY index volatility.

## Standardization Approach
The standardization process includes:
- Converting financial figures to consistent units (e.g., billions).
- Creating comparative ratios such as ROA (Return on Assets) and Asset-Liability Ratio.
- Removing unnecessary duplicate date columns and normalizing data formats.
- Filling in missing values appropriately.

## Troubleshooting
- **Ticker Issues:** If you see "No data found" errors, verify that your ticker symbols are correct.
- **Quandl API Errors:** Ensure that your API key is valid and correctly configured.
- **Data Limitations:** Some companies might have incomplete historical financial statements.
- **Regional Adjustments:** The tool is optimized for companies based in India; modifications may be needed for international data.

## Extending the Analysis
- Integrate additional financial ratios such as P/E Ratio and ROE.
- Incorporate data visualization tools.
- Add sector-specific metrics.
- Develop advanced time-series analysis features.

## Notes on Data Sources
- **Yahoo Finance:** The primary source for market and financial statement data.
- **Quandl:** Provides RBI interest rate data (requires an API key).
- The tool is designed to handle missing data gracefully by using alternative data points when needed.

Happy Analyzing!
