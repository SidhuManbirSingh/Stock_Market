import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import quandl

# Set Quandl API key
quandl.ApiConfig.api_key = "DqymmKW4ZeSB6B3kACmj"

def create_dirs():
    """Creates directories for saving CSV files."""
    dirs = ["Data/Prices", "Data/Dividends", "Data/Debt", "Data/Others"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_stock_data(ticker, start_date, end_date):
    """
    Fetches stock data (price and dividends) using yfinance.
    Returns the yf.Ticker instance, price history, and dividend actions.
    """
    stock = yf.Ticker(ticker)
    price_data = stock.history(start=start_date, end=end_date, actions=True)
    dividend_data = stock.actions
    return stock, price_data, dividend_data

def process_price_data(price_data):
    """
    Processes daily closing price data.
    Adds percent change and formatted date columns.
    """
    df = pd.DataFrame(price_data['Close'])
    df.columns = ["Close_Price"]
    df["Close_Price"] = df["Close_Price"].round(2)
    df["Change_%"] = (df["Close_Price"].pct_change() * 100).round(2)
    df["Date"] = df.index.strftime('%Y-%m-%d')
    df["Date_YYYYMMDD"] = df.index.strftime('%Y%m%d')
    df.index = df["Date"]
    return df

def process_dividend_data(dividend_data, price_dates):
    """
    Aligns dividend data with price dates.
    Missing dividend entries are set to 0.
    """
    df = pd.DataFrame(index=price_dates)
    df["Dividend"] = 0.0
    if "Dividends" in dividend_data.columns:
        temp = pd.DataFrame(dividend_data["Dividends"])
        temp.columns = ["Dividend"]
        df.loc[temp.index, "Dividend"] = temp["Dividend"]
    df["Dividend"] = df["Dividend"].round(2)
    df["Date"] = df.index.strftime('%Y-%m-%d')
    df["Date_YYYYMMDD"] = df.index.strftime('%Y%m%d')
    df.index = df["Date"]
    return df

def process_debt_data(stock):
    """
    Extracts debt-to-equity ratio data using the annual balance sheet.
    We attempt to pull extensive historical (up to 10 years) data.
    If annual data isn’t sufficient, quarterly values are used.
    """
    try:
        annual_bs = stock.balance_sheet
        if annual_bs is not None and not annual_bs.empty:
            # Get debt: prefer "Total Debt" else "Long Term Debt"
            if "Total Debt" in annual_bs.index:
                debt = annual_bs.loc["Total Debt"]
            elif "Long Term Debt" in annual_bs.index:
                debt = annual_bs.loc["Long Term Debt"]
            else:
                print("No debt field in annual balance sheet.")
                return pd.DataFrame()
            # Get equity:
            if "Total Stockholder Equity" in annual_bs.index:
                equity = annual_bs.loc["Total Stockholder Equity"]
            elif "Stockholders Equity" in annual_bs.index:
                equity = annual_bs.loc["Stockholders Equity"]
            else:
                print("No equity field in annual balance sheet.")
                return pd.DataFrame()
            ratio = (debt / equity).round(2)
            df = pd.DataFrame({"Debt_Equity_Ratio": ratio})
            df = df.sort_index()
            df["Date"] = df.index.strftime("%Y-%m-%d")
            df["Date_YYYYMMDD"] = df.index.strftime("%Y%m%d")
            df.index = df["Date"]
            if len(df.index) < 4:
                print("WARNING: Annual debt data spans less than ~10 years.")
            return df
        else:
            print("No annual balance sheet data available; attempting quarterly data.")
            quarterly_bs = stock.quarterly_balance_sheet
            if quarterly_bs is not None and not quarterly_bs.empty:
                if "Total Debt" in quarterly_bs.index:
                    debt = quarterly_bs.loc["Total Debt"]
                elif "Long Term Debt" in quarterly_bs.index:
                    debt = quarterly_bs.loc["Long Term Debt"]
                else:
                    return pd.DataFrame()
                if "Total Stockholder Equity" in quarterly_bs.index:
                    equity = quarterly_bs.loc["Total Stockholder Equity"]
                elif "Stockholders Equity" in quarterly_bs.index:
                    equity = quarterly_bs.loc["Stockholders Equity"]
                else:
                    return pd.DataFrame()
                ratio = (debt / equity).round(2)
                df = pd.DataFrame({"Debt_Equity_Ratio": ratio})
                df = df.sort_index()
                df["Date"] = df.index.strftime("%Y-%m-%d")
                df["Date_YYYYMMDD"] = df.index.strftime("%Y%m%d")
                df.index = df["Date"]
                return df
        return pd.DataFrame()
    except Exception as e:
        print("Error processing debt data:", e)
        return pd.DataFrame()

def get_interest_rates(start_date, end_date):
    """
    Fetches RBI repo rate data from Quandl (Indian government interest rate) 
    for the given 10-year period.
    """
    try:
        s_date = start_date.strftime("%Y-%m-%d")
        e_date = end_date.strftime("%Y-%m-%d")
        rates = quandl.get("RBI/RRATE", start_date=s_date, end_date=e_date)
        rates.columns = ["Interest_Rate"]
        rates["Change_%"] = (rates["Interest_Rate"].pct_change() * 100).round(2)
        rates["Date"] = rates.index.strftime('%Y-%m-%d')
        rates["Date_YYYYMMDD"] = rates.index.strftime('%Y%m%d')
        rates.index = rates["Date"]
        return rates
    except Exception as e:
        print("Error fetching interest rate data:", e)
        return pd.DataFrame()


def process_other_data(stock):
    """
    Pulls additional financial data from the annual financials:
    • Total Assets and Total Liabilities 
    • EBITDA 
    • Operating Margin (computed as Operating Income/Total Revenue * 100)
    These data are based on annual reports (if available).
    """
    df = pd.DataFrame()
    annual_bs = stock.balance_sheet
    if annual_bs is not None and not annual_bs.empty:
        print("Balance Sheet keys:", list(annual_bs.index))  # Debug statement
        # Attempt to get Total Assets
        if "Total Assets" in annual_bs.index:
            df["Total_Assets"] = annual_bs.loc["Total Assets"]
        elif "Assets" in annual_bs.index:  # alternate key
            df["Total_Assets"] = annual_bs.loc["Assets"]
        else:
            print("Total Assets not found in balance sheet.")
        
        # Attempt to get Total Liabilities from various keys
        if "Total Liab" in annual_bs.index:
            df["Total_Liabilities"] = annual_bs.loc["Total Liab"]
        elif "Total Liabilities" in annual_bs.index:
            df["Total_Liabilities"] = annual_bs.loc["Total Liabilities"]
        elif "Liab" in annual_bs.index:
            df["Total_Liabilities"] = annual_bs.loc["Liab"]
        elif "Total Liabilities Net Minority Interest" in annual_bs.index:
            df["Total_Liabilities"] = annual_bs.loc["Total Liabilities Net Minority Interest"]
        else:
            print("Total Liabilities not found in balance sheet.")
    
    # Annual financials for EBITDA and operating margin
    annual_fin = stock.financials
    if annual_fin is not None and not annual_fin.empty:
        if "EBITDA" in annual_fin.index:
            df["EBITDA"] = annual_fin.loc["EBITDA"]
        if "Operating Income" in annual_fin.index and "Total Revenue" in annual_fin.index:
            op_margin = (annual_fin.loc["Operating Income"] / annual_fin.loc["Total Revenue"] * 100).round(2)
            df["Operating_Margin_%"] = op_margin
    if not df.empty:
        df = df.sort_index()
        df["Date"] = pd.to_datetime(df.index).strftime("%Y-%m-%d")
        df["Date_YYYYMMDD"] = pd.to_datetime(df.index).strftime("%Y%m%d")
        df.index = df["Date"]
    return df


def get_market_volatility(index_ticker, start_date, end_date):
    """
    Fetches market index data (e.g., NIFTY) and computes a 30-day rolling annualized volatility.
    """
    try:
        idx_data = yf.download(index_ticker, start=start_date, end=end_date)
        volatility = idx_data['Close'].pct_change().rolling(window=30).std() * np.sqrt(252) * 100
        vol_df = pd.DataFrame({"Market_Volatility_%": volatility.round(2)}, index=idx_data.index)
        vol_df["Date"] = vol_df.index.strftime("%Y-%m-%d")
        vol_df["Date_YYYYMMDD"] = vol_df.index.strftime("%Y%m%d")
        vol_df.index = vol_df["Date"]
        return vol_df
    except Exception as e:
        print("Error fetching market volatility data:", e)
        return pd.DataFrame()

def filter_nonzero_dividends(dividend_df):
    """
    Filters dividend DataFrame to include only rows where Dividend > 0.
    """
    return dividend_df[dividend_df["Dividend"] != 0]

def save_csv(df, filename):
    """Saves DataFrame to CSV."""
    try:
        df.to_csv(filename, index=True)
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {str(e)}")



def main():
    ticker = "TATAMOTORS.NS"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=12*365)  # Approximately 12 years
    create_dirs()
    
    stock, price_data, dividend_data = get_stock_data(ticker, start_date, end_date)
    if price_data is None or dividend_data is None:
        print("Failed to retrieve data for", ticker)
        return
    
    # Process primary data
    price_df = process_price_data(price_data)
    dividend_df = process_dividend_data(dividend_data, price_data.index)
    debt_df = process_debt_data(stock)
    
    # Get additional financial data from Quandl and annual reports
    interest_df = get_interest_rates(start_date, end_date)
    other_df = process_other_data(stock)
    market_vol_df = get_market_volatility("^NSEI", start_date, end_date)
    
    # Save complete datasets
    save_csv(price_df, f"Data/Prices/{ticker}_prices.csv")
    save_csv(dividend_df, f"Data/Dividends/{ticker}_dividends.csv")
    if not debt_df.empty:
        save_csv(debt_df, f"Data/Debt/{ticker}_debt.csv")
    else:
        print("No debt data available for", ticker)
    if not interest_df.empty:
        save_csv(interest_df, f"Data/Others/{ticker}_interest.csv")
    else:
        print("No interest rate data available.")
    
    # Save market volatility separately in Others directory
    if not market_vol_df.empty:
        save_csv(market_vol_df, f"Data/Others/{ticker}_market_volatility.csv")
    else:
        print("No market volatility data available.")
    
    # Merge market volatility with other financial data to create combined file including Total Liabilities
    if not market_vol_df.empty:
        if other_df.empty:
            combined_others = market_vol_df.copy()
        else:
            combined_others = pd.merge(other_df, market_vol_df, left_index=True, right_index=True, how='outer')
    else:
        combined_others = other_df.copy()
    
    if not combined_others.empty:
        save_csv(combined_others, f"Data/Others/{ticker}_others.csv")
    else:
        print("No other financial or market volatility data available.")
    
    # Additionally, save only non-zero dividend rows.
    nonzero_div_df = filter_nonzero_dividends(dividend_df)
    if not nonzero_div_df.empty:
        save_csv(nonzero_div_df, f"Data/Dividends/{ticker}_dividends_nonzero.csv")
    else:
        print("No non-zero dividend data available for", ticker)

if __name__ == "__main__":
    main()