import pandas as pd
import numpy as np

def standardize_financial_data(input_file, output_file=None):
    """
    Standardize financial data for comparative analysis
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str, optional): Path to save standardized data
    
    Returns:
    pandas.DataFrame: Standardized financial data
    """
    # Read the input file
    df = pd.read_csv(input_file)
    
    # Clean up the data
    # 1. Remove duplicate date columns
    date_columns = [col for col in df.columns if 'Date' in col]
    if len(date_columns) > 1:
        main_date_col = date_columns[0]
        df = df.drop([col for col in date_columns if col != main_date_col], axis=1)
        df = df.rename(columns={main_date_col: 'Date'})
    
    # 2. Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 3. Set Date as index
    df = df.set_index('Date')
    
    # Exclude data from 2020-03-31
    df = df[df.index != pd.Timestamp("2020-03-31")]
    
    # 4. Create standardized metrics (suitable for cross-company comparison)
    # Convert values to billions for readability (if needed)
    scale_columns = ['Total_Assets', 'Total_Liabilities', 'EBITDA']
    for col in scale_columns:
        if col in df.columns:
            # Convert to billions (adjust scale if needed)
            df[f'{col}_Billions'] = df[col] / 1e9
    
    # Calculate financial ratios for comparison
    if 'Total_Assets' in df.columns and 'Total_Liabilities' in df.columns:
        # Asset to Liability ratio
        df['Asset_Liability_Ratio'] = df['Total_Assets'] / df['Total_Liabilities']
        
        # Equity calculation
        df['Total_Equity'] = df['Total_Assets'] - df['Total_Liabilities']
        
        # Return on Assets (if EBITDA available)
        if 'EBITDA' in df.columns:
            df['ROA'] = df['EBITDA'] / df['Total_Assets']
    
    # Convert percentages to decimal form
    percent_columns = [col for col in df.columns if '%' in col]
    for col in percent_columns:
        new_col = col.replace('%', 'Ratio')
        df[new_col] = df[col] / 100
        df = df.drop(col, axis=1)
    
    # Fill NaN values with 0 or other appropriate method
    df = df.fillna(0)
    
    # Save to output file if specified
    if output_file:
        df.to_csv(output_file)
        print(f"Standardized data saved to {output_file}")
    
    return df

def create_company_comparison(company_files, output_file=None):
    """
    Create a standardized dataset for comparing multiple companies
    
    Parameters:
    company_files (dict): Dictionary with company names as keys and file paths as values
    output_file (str, optional): Path to save combined data
    
    Returns:
    dict: Dictionary of standardized DataFrames for each company
    """
    standardized_data = {}
    combined_metrics = {}
    
    # Process each company's data
    for company, file_path in company_files.items():
        df = standardize_financial_data(file_path)
        standardized_data[company] = df
        
        # Extract key metrics for comparison
        key_metrics = ['Total_Assets_Billions', 'Total_Liabilities_Billions', 
                      'EBITDA_Billions', 'Debt_Equity_Ratio', 'ROA', 
                      'Asset_Liability_Ratio', 'Operating_MarginRatio']
        
        # Create a metrics DataFrame for this company
        company_metrics = pd.DataFrame()
        for metric in key_metrics:
            if metric in df.columns:
                company_metrics[f"{company}_{metric}"] = df[metric]
        
        # Add to combined metrics
        if combined_metrics:
            combined_metrics = pd.concat([combined_metrics, company_metrics], axis=1)
        else:
            combined_metrics = company_metrics
    
    # Save combined metrics if specified
    if output_file and combined_metrics is not None:
        combined_metrics.to_csv(output_file)
        print(f"Combined comparison data saved to {output_file}")
    
    return standardized_data, combined_metrics

# Example usage
if __name__ == "__main__":
    # Standardize all companies' consolidated others data from all_others.csv
    all_others_std = standardize_financial_data("Data/Others/all_others.csv", "all_others_standardized.csv")
    print("Standardized data from all_others.csv:")
    print(all_others_std.head())
    
    # Example for multiple companies comparison (uncomment and update file paths as needed)
    # company_files = {
    #     "TATAMOTORS": "TATAMOTORS.NS_others.csv",
    #     "MARUTI": "MARUTI.NS_others.csv",
    #     "MAHINDRA": "MAHINDRA.NS_others.csv"
    # }
    # standardized_companies, comparison = create_company_comparison(company_files, "auto_companies_comparison.csv")