import yfinance as yf   
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import quandl
import os
import logging
import warnings
from typing import Tuple, List, Dict, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set Quandl API key
quandl.ApiConfig.api_key = "DqymmKW4ZeSB6B3kACmj"


class StockDataFetcher:
    """Class responsible for fetching stock data from various sources."""
    
    def __init__(self, ticker_symbol: str, start_date: datetime, end_date: datetime):
        """
        Initialize StockDataFetcher with ticker symbol and date range.
        
        Args:
            ticker_symbol: The stock ticker symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
        """
        self.ticker_symbol = ticker_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.stock = None
        self.price_data = None
        self.dividend_data = None
        self.market_index = None
        
    def fetch_stock_data(self) -> bool:
        """
        Fetch basic stock data including price and dividend information.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Fetching data for {self.ticker_symbol}...")
            self.stock = yf.Ticker(self.ticker_symbol)
            self.price_data = self.stock.history(start=self.start_date, end=self.end_date, actions=True)
            
            if self.price_data.empty:
                logger.error(f"No price data available for {self.ticker_symbol}")
                return False
                
            logger.info(f"Retrieved {len(self.price_data)} rows of price data")
            
            self.dividend_data = self.stock.actions
            if self.dividend_data is None or self.dividend_data.empty:
                logger.warning(f"No dividend data available for {self.ticker_symbol}")
            else:
                logger.info(f"Retrieved dividend data with {len(self.dividend_data)} rows")
                
            return True
            
        except Exception as e:
            logger.error(f"Error fetching basic stock data for {self.ticker_symbol}: {str(e)}")
            return False
            
    def fetch_market_index(self) -> bool:
        """
        Fetch market index data (NIFTY 50).
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Fetching market index data (NIFTY 50)...")
            self.market_index = yf.download('^NSEI', start=self.start_date, end=self.end_date)
            
            if self.market_index is None or self.market_index.empty:
                logger.warning("Failed to fetch NIFTY 50 data")
                return False
                
            logger.info(f"Retrieved {len(self.market_index)} rows of market index data")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching market index data: {str(e)}")
            return False
            
    def fetch_eps_data_primary(self) -> pd.DataFrame:
        """
        Fetch EPS data from primary source (Yahoo Finance).
        
        Returns:
            DataFrame: EPS data or empty DataFrame if failed
        """
        try:
            logger.info(f"Fetching EPS data for {self.ticker_symbol} from Yahoo Finance...")
            earnings = self.stock.earnings
            quarterly_earnings = self.stock.quarterly_earnings
            
            if quarterly_earnings is not None and not quarterly_earnings.empty:
                logger.info(f"Retrieved {len(quarterly_earnings)} quarters of EPS data")
                return quarterly_earnings
            elif earnings is not None and not earnings.empty:
                logger.info(f"Retrieved {len(earnings)} years of EPS data")
                return earnings
            else:
                logger.warning(f"No EPS data available from Yahoo Finance for {self.ticker_symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching EPS data from Yahoo Finance: {str(e)}")
            return pd.DataFrame()
            
    def fetch_eps_data_alternative(self) -> pd.DataFrame:
        """
        Fetch EPS data from alternative source (Quandl).
        
        Returns:
            DataFrame: EPS data or empty DataFrame if failed
        """
        try:
            logger.info(f"Attempting to fetch EPS data for {self.ticker_symbol} from Quandl...")
            
            # Convert ticker symbol for Quandl format if needed
            # Example: TATA.NS to TATA_NSE
            quandl_ticker = self.ticker_symbol.replace('.NS', '_NSE')
            
            # Try to get EPS data from Quandl (Example dataset path)
            try:
                eps_data = quandl.get(f"NSE/EPS/{quandl_ticker}", start_date=self.start_date, end_date=self.end_date)
                if not eps_data.empty:
                    logger.info(f"Retrieved {len(eps_data)} rows of EPS data from Quandl")
                    return eps_data
            except:
                logger.warning(f"No EPS data available from Quandl for {self.ticker_symbol}")
            
            # If we're here, Quandl didn't work either
            logger.warning(f"Failed to retrieve EPS data from alternative sources for {self.ticker_symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching EPS data from alternative sources: {str(e)}")
            return pd.DataFrame()
            
    def fetch_balance_sheet_data(self) -> pd.DataFrame:
        """
        Fetch balance sheet data from Yahoo Finance.
        
        Returns:
            DataFrame: Balance sheet data or empty DataFrame if failed
        """
        try:
            logger.info(f"Fetching balance sheet data for {self.ticker_symbol}...")
            
            quarterly_balance_sheet = self.stock.quarterly_balance_sheet
            annual_balance_sheet = self.stock.balance_sheet
            
            if quarterly_balance_sheet is not None and not quarterly_balance_sheet.empty:
                logger.info(f"Retrieved {len(quarterly_balance_sheet.columns)} quarters of balance sheet data")
                return quarterly_balance_sheet
            elif annual_balance_sheet is not None and not annual_balance_sheet.empty:
                logger.info(f"Retrieved {len(annual_balance_sheet.columns)} years of balance sheet data")
                return annual_balance_sheet
            else:
                logger.warning(f"No balance sheet data available for {self.ticker_symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching balance sheet data: {str(e)}")
            return pd.DataFrame()
            
    def fetch_all_data(self) -> bool:
        """
        Fetch all required data for the stock.
        
        Returns:
            bool: True if basic data retrieval was successful, False otherwise
        """
        if not self.fetch_stock_data():
            return False
            
        self.fetch_market_index()
        
        # These will be processed by StockProcessor class later
        # They may be empty DataFrames if retrieval fails
        self.eps_data_primary = self.fetch_eps_data_primary()
        self.eps_data_alternative = self.fetch_eps_data_alternative()
        self.balance_sheet_data = self.fetch_balance_sheet_data()
        
        return True


class StockProcessor:
    """Class responsible for processing and analyzing stock data."""
    
    def __init__(self, data_fetcher: StockDataFetcher):
        """
        Initialize StockProcessor with data from StockDataFetcher.
        
        Args:
            data_fetcher: StockDataFetcher instance with fetched data
        """
        self.data_fetcher = data_fetcher
        self.ticker_symbol = data_fetcher.ticker_symbol
        self.price_df = None
        self.dividend_df = None
        self.eps_df = None
        self.metrics_df = None
        
    def process_price_data(self) -> pd.DataFrame:
        """
        Process daily stock price data with volatility and market comparison.
        
        Returns:
            DataFrame: Processed price data
        """
        logger.info(f"Processing price data for {self.ticker_symbol}...")
        
        # Extract price data
        price_data = self.data_fetcher.price_data
        
        # Create base price DataFrame
        price_df = pd.DataFrame(price_data['Close'])
        price_df.columns = ['Close_Price']
        price_df['Close_Price'] = price_df['Close_Price'].round(2)
        
        # Calculate daily percentage change
        price_df['Change_%'] = price_df['Close_Price'].pct_change() * 100
        price_df['Change_%'] = price_df['Change_%'].round(2)
        
        # Calculate volatility (20-day rolling standard deviation annualized)
        try:
            price_df['Volatility_%'] = price_df['Change_%'].rolling(window=20).std() * np.sqrt(252)
            price_df['Volatility_%'] = price_df['Volatility_%'].round(2)
            logger.info("Volatility calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
        
        # Add market index data if available
        if self.data_fetcher.market_index is not None and not self.data_fetcher.market_index.empty:
            try:
                market_df = self.data_fetcher.market_index
                price_df['Market_Index'] = market_df['Close']
                price_df['Market_Change_%'] = price_df['Market_Index'].pct_change() * 100
                price_df['Market_Change_%'] = price_df['Market_Change_%'].round(2)
                
                # Calculate beta (relationship to market)
                returns = price_df['Change_%'].fillna(0)
                market_returns = price_df['Market_Change_%'].fillna(0)
                if len(returns) > 30:  # Need sufficient data points
                    covariance = returns.cov(market_returns)
                    market_variance = market_returns.var()
                    if market_variance != 0:
                        beta = covariance / market_variance
                        price_df['Beta_60D'] = [None] * len(price_df)
                        price_df.iloc[-1, price_df.columns.get_loc('Beta_60D')] = round(beta, 2)
                        logger.info(f"Beta calculated: {beta:.2f}")
                
                logger.info("Market data integrated successfully")
            except Exception as e:
                logger.error(f"Error adding market index data: {str(e)}")
        
        # Add date columns for reference
        price_df['Date_YYYYMMDD'] = price_df.index.strftime('%Y%m%d')
        
        # Store the original datetime index as string for CSV
        price_df.index = price_df.index.strftime('%Y-%m-%d')
        
        self.price_df = price_df
        logger.info(f"Price data processing complete with {len(price_df)} rows")
        
        return price_df
        
    def process_dividend_data(self) -> pd.DataFrame:
        """
        Process dividend data.
        
        Returns:
            DataFrame: Processed dividend data
        """
        logger.info(f"Processing dividend data for {self.ticker_symbol}...")
        
        # Create DataFrame with same index as price data
        price_data = self.data_fetcher.price_data
        div_df = pd.DataFrame(index=price_data.index)
        div_df['Dividend'] = 0.0
        
        # Add dividend data if available
        dividend_data = self.data_fetcher.dividend_data
        if dividend_data is not None and 'Dividends' in dividend_data.columns:
            temp_div_df = pd.DataFrame(dividend_data['Dividends'])
            temp_div_df.columns = ['Dividend']
            
            # Find overlapping indices and update
            div_df.loc[temp_div_df.index.intersection(div_df.index), 'Dividend'] = \
                temp_div_df.loc[temp_div_df.index.intersection(div_df.index), 'Dividend']
            
            div_df['Dividend'] = div_df['Dividend'].round(2)
            
            # Calculate year-over-year dividend growth
            div_df['YoY_Growth_%'] = div_df['Dividend'].pct_change(periods=252) * 100
            div_df['YoY_Growth_%'] = div_df['YoY_Growth_%'].round(2)
            
            dividend_count = len(temp_div_df[temp_div_df['Dividend'] > 0])
            logger.info(f"Processed {dividend_count} dividend payments")
        else:
            logger.info("No dividend data available")
        
        # Add date columns for reference
        div_df['Date_YYYYMMDD'] = div_df.index.strftime('%Y%m%d')
        div_df['Year'] = div_df.index.year
        div_df['Quarter'] = div_df.index.quarter
        
        # Convert index to string for CSV
        div_df.index = div_df.index.strftime('%Y-%m-%d')
        
        self.dividend_df = div_df
        logger.info(f"Dividend data processing complete with {len(div_df)} rows")
        
        return div_df
        
    def process_eps_data(self) -> pd.DataFrame:
        """
        Process EPS data from primary or alternative sources.
        
        Returns:
            DataFrame: Processed EPS data
        """
        logger.info(f"Processing EPS data for {self.ticker_symbol}...")
        
        # Create DataFrame with same index as price data
        price_data = self.data_fetcher.price_data
        eps_df = pd.DataFrame(index=price_data.index)
        eps_df['EPS'] = None
        
        # Check primary source (Yahoo Finance)
        has_eps_data = False
        
        # Process from primary source
        primary_eps = self.data_fetcher.eps_data_primary
        if not primary_eps.empty:
            try:
                for date, row in primary_eps.iterrows():
                    # Get a date string and convert to datetime
                    date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
                    
                    # Find the corresponding date or closest future date in our index
                    matching_dates = eps_df.index[eps_df.index >= date_str]
                    if len(matching_dates) > 0:
                        closest_date = matching_dates[0]
                        # Forward fill EPS value
                        eps_df.loc[closest_date:, 'EPS'] = row['Earnings']
                
                # Check if we got valid data
                if eps_df['EPS'].notna().any():
                    has_eps_data = True
                    logger.info("Successfully processed EPS data from Yahoo Finance")
            except Exception as e:
                logger.error(f"Error processing primary EPS data: {str(e)}")
        
        # If primary source failed, try alternative source
        if not has_eps_data:
            logger.info("Attempting to process EPS data from alternative source...")
            alt_eps = self.data_fetcher.eps_data_alternative
            
            if not alt_eps.empty:
                try:
                    # The format of alternative data might differ; adjust as needed
                    for date, value in alt_eps.iterrows():
                        date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
                        
                        matching_dates = eps_df.index[eps_df.index >= date_str]
                        if len(matching_dates) > 0:
                            closest_date = matching_dates[0]
                            # Get the EPS value (adjust column name based on Quandl data structure)
                            eps_value = value[0] if isinstance(value, pd.Series) else value
                            # Forward fill EPS value
                            eps_df.loc[closest_date:, 'EPS'] = eps_value
                    
                    # Check if we got valid data
                    if eps_df['EPS'].notna().any():
                        has_eps_data = True
                        logger.info("Successfully processed EPS data from alternative source")
                except Exception as e:
                    logger.error(f"Error processing alternative EPS data: {str(e)}")
        
        if not has_eps_data:
            logger.warning("Failed to retrieve valid EPS data from all sources")
            
        # Add date columns for reference
        eps_df['Date_YYYYMMDD'] = eps_df.index
        
        # Forward fill the EPS values
        eps_df['EPS'] = eps_df['EPS'].fillna(method='ffill')
        
        # Calculate trailing 12-month EPS if possible
        if has_eps_data:
            try:
                # This is simplified and may need adjustment based on actual data format
                eps_df['TTM_EPS'] = eps_df['EPS'].rolling(window=4).sum()
                logger.info("Calculated trailing 12-month EPS")
            except Exception as e:
                logger.error(f"Error calculating TTM EPS: {str(e)}")
        
        self.eps_df = eps_df
        logger.info(f"EPS data processing complete with {len(eps_df)} rows")
        
        return eps_df
        
    def process_financial_metrics(self) -> pd.DataFrame:
        """
        Process financial metrics including Debt-Equity ratio.
        
        Returns:
            DataFrame: Processed financial metrics
        """
        logger.info(f"Processing financial metrics for {self.ticker_symbol}...")
        
        # Create DataFrame with same index as price data
        price_data = self.data_fetcher.price_data
        metrics_df = pd.DataFrame(index=price_data.index)
        metrics_df['Debt_Equity_Ratio'] = None
        
        # Process balance sheet data if available
        balance_sheet = self.data_fetcher.balance_sheet_data
        if not balance_sheet.empty:
            try:
                # Check for required fields
                debt_field = None
                for field in ['Total Debt', 'Long Term Debt']:
                    if field in balance_sheet.index:
                        debt_field = field
                        break
                
                equity_field = None
                for field in ['Total Stockholder Equity', 'Total Equity']:
                    if field in balance_sheet.index:
                        equity_field = field
                        break
                
                if debt_field and equity_field:
                    debt = balance_sheet.loc[debt_field]
                    equity = balance_sheet.loc[equity_field]
                    
                    # Calculate debt-equity ratio
                    de_ratio = pd.Series(index=debt.index)
                    for date in debt.index:
                        if equity[date] != 0:
                            de_ratio[date] = round(debt[date] / equity[date], 2)
                    
                    # Populate metrics DataFrame
                    for date in de_ratio.index:
                        # Convert date to string format if it's a timestamp
                        date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
                        
                        # Find closest future date in our index
                        matching_dates = metrics_df.index[metrics_df.index >= date_str]
                        if len(matching_dates) > 0:
                            closest_date = matching_dates[0]
                            # Forward fill D/E ratio
                            metrics_df.loc[closest_date:, 'Debt_Equity_Ratio'] = de_ratio[date]
                    
                    logger.info(f"Processed Debt-Equity ratio for {len(de_ratio)} periods")
                else:
                    logger.warning("Missing required fields in balance sheet data")
            except Exception as e:
                logger.error(f"Error processing balance sheet data: {str(e)}")
        else:
            logger.warning("No balance sheet data available")
        
        # Add P/E ratio if both price and EPS data are available
        if self.price_df is not None and self.eps_df is not None:
            try:
                # Make sure indices match
                price_values = self.price_df['Close_Price']
                eps_values = self.eps_df['EPS']
                
                # Calculate P/E ratio
                metrics_df['PE_Ratio'] = None
                for date in metrics_df.index:
                    if date in price_values.index and date in eps_values.index:
                        price = price_values[date]
                        eps = eps_values[date]
                        if eps and eps != 0:
                            metrics_df.loc[date, 'PE_Ratio'] = round(price / eps, 2)
                
                logger.info("Calculated P/E ratio")
            except Exception as e:
                logger.error(f"Error calculating P/E ratio: {str(e)}")
        
        # Add date columns for reference
        metrics_df['Date_YYYYMMDD'] = metrics_df.index
        
        # Forward fill missing values
        metrics_df = metrics_df.fillna(method='ffill')
        
        self.metrics_df = metrics_df
        logger.info(f"Financial metrics processing complete with {len(metrics_df)} rows")
        
        return metrics_df
        
    def process_all_data(self):
        """Process all stock data."""
        self.process_price_data()
        self.process_dividend_data()
        self.process_eps_data()
        self.process_financial_metrics()
        
        # Display summary
        self.display_summary()
        
    def display_summary(self):
        """Display summary statistics for the processed data."""
        logger.info(f"\n===== Summary for {self.ticker_symbol} =====")
        
        if self.price_df is not None:
            latest_price = self.price_df['Close_Price'].iloc[-1] if not self.price_df.empty else "N/A"
            latest_vol = self.price_df['Volatility_%'].iloc[-1] if 'Volatility_%' in self.price_df.columns and not self.price_df.empty else "N/A"
            logger.info(f"Price data rows: {len(self.price_df)}")
            logger.info(f"Latest price: {latest_price}")
            logger.info(f"Latest volatility: {latest_vol}")
        
        if self.dividend_df is not None:
            total_div = self.dividend_df['Dividend'].sum() if not self.dividend_df.empty else 0
            dividend_count = len(self.dividend_df[self.dividend_df['Dividend'] > 0])
            logger.info(f"Total dividends: {total_div:.2f}")
            logger.info(f"Dividend occurrences: {dividend_count}")
        
        if self.eps_df is not None:
            has_eps = self.eps_df['EPS'].notna().any() if not self.eps_df.empty else False
            latest_eps = self.eps_df['EPS'].iloc[-1] if has_eps and not self.eps_df.empty else "N/A"
            logger.info(f"EPS data available: {has_eps}")
            logger.info(f"Latest EPS: {latest_eps}")
        
        if self.metrics_df is not None:
            has_de = self.metrics_df['Debt_Equity_Ratio'].notna().any() if not self.metrics_df.empty else False
            latest_de = self.metrics_df['Debt_Equity_Ratio'].iloc[-1] if has_de and not self.metrics_df.empty else "N/A"
            logger.info(f"D/E Ratio available: {has_de}")
            logger.info(f"Latest D/E Ratio: {latest_de}")
            
            if 'PE_Ratio' in self.metrics_df.columns:
                has_pe = self.metrics_df['PE_Ratio'].notna().any() if not self.metrics_df.empty else False
                latest_pe = self.metrics_df['PE_Ratio'].iloc[-1] if has_pe and not self.metrics_df.empty else "N/A"
                logger.info(f"P/E Ratio available: {has_pe}")
                logger.info(f"Latest P/E Ratio: {latest_pe}")
        
        logger.info("="*40)


class FileHandler:
    """Class responsible for managing file operations."""
    
    def __init__(self):
        """Initialize FileHandler."""
        self.base_dir = "Data"
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories for storing data."""
        directories = [
            f"{self.base_dir}/Stock_Price",
            f"{self.base_dir}/Dividend",
            f"{self.base_dir}/EPS",
            f"{self.base_dir}/Other_metrics"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            
    def save_data_to_csv(self, df: pd.DataFrame, filename: str) -> bool:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Destination filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df.to_csv(filename)
            logger.info(f"Data saved successfully to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving to {filename}: {str(e)}")
            return False
            
    def save_processed_data(self, ticker_symbol: str, processor: StockProcessor) -> bool:
        """
        Save all processed data for a ticker symbol.
        
        Args:
            ticker_symbol: Stock ticker symbol
            processor: StockProcessor instance with processed data
            
        Returns:
            bool: True if all saves were successful, False otherwise
        """
        success = True
        
        # Save price data
        if processor.price_df is not None and not processor.price_df.empty:
            price_filename = f"{self.base_dir}/Stock_Price/{ticker_symbol}_prices.csv"
            if not self.save_data_to_csv(processor.price_df, price_filename):
                success = False
        
        # Save dividend data
        if processor.dividend_df is not None and not processor.dividend_df.empty:
            div_filename = f"{self.base_dir}/Dividend/{ticker_symbol}_dividends.csv"
            if not self.save_data_to_csv(processor.dividend_df, div_filename):
                success = False
        
        # Save EPS data
        if processor.eps_df is not None and not processor.eps_df.empty:
            eps_filename = f"{self.base_dir}/EPS/{ticker_symbol}_eps.csv"
            if not self.save_data_to_csv(processor.eps_df, eps_filename):
                success = False
        
        # Save financial metrics
        if processor.metrics_df is not None and not processor.metrics_df.empty:
            metrics_filename = f"{self.base_dir}/Other_metrics/{ticker_symbol}_metrics.csv"
            if not self.save_data_to_csv(processor.metrics_df, metrics_filename):
                success = False
        
        return success


class StockDataManager:
    """Main class for managing the stock data collection and processing workflow."""
    
    def __init__(self, test_mode: bool = True, test_ticker: str = "TATAMOTORS.NS"):
        """
        Initialize StockDataManager.
        
        Args:
            test_mode: Whether to run in test mode (single ticker)
            test_ticker: Ticker to use in test mode
        """
        self.test_mode = test_mode
        self.test_ticker = test_ticker
        self.file_handler = FileHandler()
        
        # Date range for data fetching (12 years)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=12*365)
        
    def process_ticker(self, ticker_symbol: str) -> bool:
        """
        Process data for a single ticker.
        
        Args:
            ticker_symbol: Stock ticker symbol
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"\n{'='*20} Processing {ticker_symbol} {'='*20}")
        
        # Initialize data fetcher
        data_fetcher = StockDataFetcher(ticker_symbol, self.start_date, self.end_date)
        
        # Fetch all data
        if not data_fetcher.fetch_all_data():
            logger.error(f"Failed to fetch basic data for {ticker_symbol}")
            return False
        
        # Process data
        processor = StockProcessor(data_fetcher)
        processor.process_all_data()
        
        # Save processed data
        result = self.file_handler.save_processed_data(ticker_symbol, processor)
        
        logger.info(f"{'='*20} Completed {ticker_symbol} {'='*20}\n")
        
        return result
        
    def run(self):
        """Run the data collection and processing workflow."""
        if self.test_mode:
            logger.info(f"Running in TEST MODE with ticker: {self.test_ticker}")
            self.process_ticker(self.test_ticker)
        else:
            logger.info("Running in BATCH MODE")
            try:
                ticker_df = pd.read_csv('ticker.csv')
                ticker_symbols = ticker_df['Ticker'].tolist()
                
                successful_tickers = 0
                failed_tickers = 0
                
                for ticker_symbol in ticker_symbols:
                    if self.process_ticker(ticker_symbol):
                        successful_tickers += 1
                    else:
                        failed_tickers += 1
                
                logger.info(f"\nProcessing complete. Successful: {successful_tickers}, Failed: {failed_tickers}")
                
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")


def main():
    """Main function to run the stock data collection and processing."""
    
    # Run in test mode with a single ticker
    test_mode = True
    test_ticker = "TATAMOTORS.NS"
    
    # Create and run the manager
    manager = StockDataManager(test_mode=test_mode, test_ticker=test_ticker)
    manager.run()
    
    logger.info("Program execution complete.")


if __name__ == "__main__":
    main()
    
    # Uncomment to run in batch mode
    # manager = StockDataManager(test_mode=False)
    # manager.run()