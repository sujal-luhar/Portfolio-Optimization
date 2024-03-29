import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException

import yfinance as yf
# Define function to read data for a single stock from Yahoo Finance
def read_stock_data(stock_symbol, start_date, end_date):
    """
    Read data for a single stock from Yahoo Finance
    """
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    df = stock_data[['Close']].rename(columns={'Close': stock_symbol})
    logging.info(f"Data for stock: {stock_symbol} has been successfully retrieved")
    return df

# Define function to create DataFrame with closing prices for multiple stocks
def create_portfolio_dataframe(stock_symbols, start_date, end_date):
    """
    Create a DataFrame with closing prices for multiple stocks
    """
    # Initialize an empty DataFrame
    portfolio_df = pd.DataFrame(index=pd.date_range(start_date, end_date))

    # Loop through each stock symbol
    for stock_symbol in stock_symbols:
        try:
            # Read data for the current stock from Yahoo Finance
            stock_df = read_stock_data(stock_symbol, start_date, end_date)
            
            # Merge the current stock data with the portfolio DataFrame
            portfolio_df = pd.merge(portfolio_df, stock_df, how='left', left_index=True, right_index=True)
        except Exception as e:
            logging.info(CustomException(e,sys))
            print(f"Error retrieving data for stock: {stock_symbol}")
            raise CustomException(e,sys)

        
    return portfolio_df.dropna()