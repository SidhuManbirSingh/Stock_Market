import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging (overwriting previous logs)
logging.basicConfig(
    filename='visuals_generation.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set the style for the plots
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')

# --- Data Loading ---
def load_financial_data(filepath="Data/financial_analysis.xlsx"):
    """Loads data from the specified Excel file."""
    if not os.path.exists(filepath):
        logging.error(f"Input file not found: {filepath}")
        print(f"Error: Input file '{filepath}' does not exist.")
        return None
    try:
        logging.info(f"Reading data from {filepath}...")
        df = pd.read_excel(filepath)
        logging.info(f"Successfully read {len(df)} rows from {filepath}.")
        # Basic data cleaning: Ensure 'Year' is integer and remove rows with NaN year
        if 'Year' in df.columns:
            df['Year'] = df['Year'].fillna(-1).astype(int)
            df = df[df['Year'] != -1]
        
        # Convert potential ratio columns to numeric
        ratio_cols = ['ROA', 'Operating_Margin', 'Debt_to_Equity_Ratio', 'Asset_Liability_Ratio']
        for col in ratio_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logging.info(f"Converted column '{col}' to numeric, coercing errors.")
        
        # Convert price/dividend columns to numeric if needed
        price_div_cols = ['Stock_Price_Mean', 'Stock_Price_Lower_CI', 'Stock_Price_Upper_CI',
                          'Dividend_Mean', 'Dividend_Lower_CI', 'Dividend_Upper_CI']
        for col in price_div_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logging.info(f"Ensured column '{col}' is numeric, coercing errors.")
        
        return df
    except Exception as e:
        logging.error(f"Error reading or processing Excel file {filepath}: {e}")
        print(f"Error: Could not read or process the Excel file '{filepath}'. Check format and permissions.")
        return None


# --- Create Visualizations ---
def create_visualizations(df, output_dir="visuals"):
    """Generates visualizations from the financial DataFrame."""
    if df is None or df.empty:
        logging.warning("DataFrame is empty or None. No visualizations will be created.")
        print("Warning: No data available to create visualizations.")
        return

    # Create directory for saving plots
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory '{output_dir}' ensured.")
    except OSError as e:
        logging.error(f"Could not create output directory '{output_dir}': {e}")
        print(f"Error: Failed to create output directory '{output_dir}'. Check permissions.")
        return

    # 1. Correlation Heatmap for Numeric Columns
    try:
        logging.info("Attempting to generate Correlation Heatmap...")
        numeric_cols = ['Stock_Price_Mean', 'Dividend_Mean', 'ROA', 'Operating_Margin',
                        'Debt_to_Equity_Ratio', 'Asset_Liability_Ratio']
        available_cols = [col for col in numeric_cols if col in df.columns and df[col].notna().any()]
        if len(available_cols) > 1:
            # Aggregate data by ticker (mean across years)
            df_aggregated = df.groupby('Ticker')[available_cols].mean().reset_index()
            corr_matrix = df_aggregated[available_cols].corr()
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm',
                        vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
            plt.title('Correlation Heatmap of Key Metrics (Averaged by Ticker)', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            save_path = os.path.join(output_dir, 'correlation_heatmap.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved Correlation Heatmap to {save_path}")
        else:
            logging.warning("Skipping Correlation Heatmap: Not enough numeric columns with data available.")
    except Exception as e:
        logging.error(f"Error generating Correlation Heatmap: {e}")
        plt.close()

    # 2. Price vs Dividend Scatter Plot
    try:
        logging.info("Attempting to generate Price vs Dividend Scatter Plot...")
        required_cols = ['Stock_Price_Mean', 'Dividend_Mean', 'Ticker']
        if all(col in df.columns for col in required_cols) and \
           df['Stock_Price_Mean'].notna().any() and df['Dividend_Mean'].notna().any():
            plt.figure(figsize=(12, 8))
            # Use average per ticker for less clutter
            df_avg = df.groupby('Ticker')[['Stock_Price_Mean', 'Dividend_Mean']].mean().reset_index()
            sns.scatterplot(data=df_avg, x='Stock_Price_Mean', y='Dividend_Mean', hue='Ticker', s=100, alpha=0.7)
            plt.title('Average Stock Price vs Average Dividend by Company', fontsize=16)
            plt.xlabel('Average Stock Price', fontsize=14)
            plt.ylabel('Average Dividend', fontsize=14)
            plt.grid(True, alpha=0.3)
            # Adjust legend: hide if too many tickers; otherwise place it outside plot
            if len(df['Ticker'].unique()) > 30:
                plt.legend().set_visible(False)
                plt.title('Average Stock Price vs Average Dividend (Tickers numerous)', fontsize=16)
            else:
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Ticker')
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            save_path = os.path.join(output_dir, 'price_vs_dividend_scatter.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved Price vs Dividend Scatter Plot to {save_path}")
        else:
            logging.warning(f"Skipping Price vs Dividend plot. Missing one or more required columns: {required_cols} or columns have no data.")
    except Exception as e:
        logging.error(f"Error generating Price vs Dividend Scatter Plot: {e}")
        plt.close()

    # 3. Price Trends Over Time (Using Stock_Price_Mean)
    try:
        logging.info("Attempting to generate Price Trends Over Time plot...")
        required_cols = ['Year', 'Stock_Price_Mean', 'Ticker']
        if all(col in df.columns for col in required_cols) and df['Stock_Price_Mean'].notna().any():
            plt.figure(figsize=(14, 10))
            tickers = df['Ticker'].unique()
            for ticker in tickers:
                ticker_data = df[df['Ticker'] == ticker].sort_values('Year')
                if not ticker_data.empty:
                    plt.plot(ticker_data['Year'], ticker_data['Stock_Price_Mean'], marker='.', linewidth=1.5,
                             label=ticker if len(tickers) <= 20 else None)
            plt.title(f'Stock Price Trends ({df["Year"].min()}-{df["Year"].max()})', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Mean Stock Price', fontsize=14)
            plt.grid(True, alpha=0.3)
            if len(tickers) <= 20:
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Ticker')
            else:
                plt.title(f'Stock Price Trends ({df["Year"].min()}-{df["Year"].max()}) - Individual Tickers Numerous', fontsize=16)
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            save_path = os.path.join(output_dir, 'price_trends.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved Price Trends plot to {save_path}")
        else:
            logging.warning(f"Skipping Price Trends plot. Missing one or more required columns: {required_cols} or column has no data.")
    except Exception as e:
        logging.error(f"Error generating Price Trends plot: {e}")
        plt.close()

    # 4. Dividend Yield Calculation and Visualization
    try:
        logging.info("Attempting to generate Dividend Yield plot...")
        required_cols = ['Dividend_Mean', 'Stock_Price_Mean', 'Year', 'Ticker']
        if all(col in df.columns for col in required_cols) and \
           df['Dividend_Mean'].notna().any() and df['Stock_Price_Mean'].notna().any():
            df_yield = df.copy()
            df_yield['Dividend_Yield'] = np.where(
                df_yield['Stock_Price_Mean'] > 0.01,
                (df_yield['Dividend_Mean'] / df_yield['Stock_Price_Mean']) * 100,
                0
            )
            df_yield = df_yield.dropna(subset=['Dividend_Yield'])
            if not df_yield.empty:
                plt.figure(figsize=(16, 8))
                avg_yield = df_yield.groupby('Ticker')['Dividend_Yield'].mean().reset_index().sort_values('Dividend_Yield', ascending=False)
                num_tickers = len(avg_yield['Ticker'].unique())
                top_n = 30
                if num_tickers > top_n:
                    avg_yield = avg_yield.head(top_n)
                    plot_title = f'Average Dividend Yield (%) - Top {top_n} Tickers'
                else:
                    plot_title = 'Average Dividend Yield (%) by Company'
                sns.barplot(data=avg_yield, x='Ticker', y='Dividend_Yield', palette='viridis')
                plt.title(plot_title, fontsize=16)
                plt.xlabel('Company', fontsize=14)
                plt.ylabel('Average Dividend Yield (%)', fontsize=14)
                plt.xticks(rotation=75)
                plt.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                save_path = os.path.join(output_dir, 'average_dividend_yield.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved Average Dividend Yield plot to {save_path}")
            else:
                logging.warning("Skipping Dividend Yield plot: No valid yield data after calculation.")
        else:
            logging.warning(f"Skipping Dividend Yield plot. Missing one or more required columns: {required_cols} or columns have no data.")
    except Exception as e:
        logging.error(f"Error generating Dividend Yield plot: {e}")
        plt.close()

    # 5. Financial Ratios Distribution (Box Plots)
    ratio_cols_to_plot = ['ROA', 'Operating_Margin', 'Debt_to_Equity_Ratio', 'Asset_Liability_Ratio']
    for ratio_col in ratio_cols_to_plot:
        try:
            logging.info(f"Attempting to generate {ratio_col} Distribution plot...")
            required_cols = [ratio_col, 'Ticker']
            if all(col in df.columns for col in required_cols) and df[ratio_col].notna().any():
                order = df.groupby('Ticker')[ratio_col].median().sort_values().index
                plt.figure(figsize=(14, 8))
                sns.boxplot(data=df, x='Ticker', y=ratio_col, order=order, palette='coolwarm')
                plt.title(f'{ratio_col} Distribution by Company (All Years)', fontsize=16)
                plt.xlabel('Company', fontsize=14)
                plt.ylabel(ratio_col, fontsize=14)
                plt.xticks(rotation=75)
                plt.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                save_path = os.path.join(output_dir, f'{ratio_col}_distribution.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved {ratio_col} Distribution plot to {save_path}")
            else:
                logging.warning(f"Skipping {ratio_col} plot. Column not found or has no data.")
        except Exception as e:
            logging.error(f"Error generating {ratio_col} Distribution plot: {e}")
            plt.close()

    logging.info("Completed generating visualizations.")


# --- Main Execution Block ---
if __name__ == "__main__":
    logging.info("--- Visualization Script Start ---")
    input_excel_file = "Data/financial_analysis.xlsx"
    visuals_output_dir = "visuals"

    # Load data
    stock_data = load_financial_data(input_excel_file)

    # Create visualizations if data was loaded successfully
    if stock_data is not None:
        create_visualizations(stock_data, visuals_output_dir)
        print(f"Visualizations generation process finished. Check the '{visuals_output_dir}' directory and 'visuals_generation.log'.")
    else:
        print("Visualization script could not proceed due to data loading errors.")

    logging.info("--- Visualization Script End ---")