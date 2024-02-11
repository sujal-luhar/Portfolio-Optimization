import pandas as pd
import os
import sys

def list_symboles_in_directory():
    """
    List filenames available in the specified directory.
    
    Args:
    - directory (str): Path to the directory
    
    Returns:
    - files (list): List of filenames in the directory
    """
    directory = r'C:\Users\luhar\Downloads\archive\Stocks'
    files = []
    symbols = []

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)

    for i in files:
        i = i.split('.')[0]
        symbols.append(i)

    return symbols

def get_stock_price(ticker):
    """
    Load the stock Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    price = pd.read_csv(f"././data/raw/Stocks/{ticker}.us.txt",
                       header=0, index_col=0, parse_dates=True)["Close"]
    return price

def get_stock_returns(ticker):
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    data = pd.read_csv(f"././data/raw/Stocks/{ticker}.us.txt",
                       header=0, index_col=0, parse_dates=True)
    rets = data["Close"].pct_change().dropna()
    return rets