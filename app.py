import sys
from src.logger import logging
from src.exception import CustomException
import streamlit as st
from src.data.make_dataset import get_stock_returns
from src.data.make_dataset import get_stock_price
from src.data.make_dataset import list_symboles_in_directory
import pandas as pd
import matplotlib.pyplot as plt


ticker = st.multiselect(
    "Choose Company by Symbol", list_symboles_in_directory(), ['aapl', 'amzn', 'googl']
    )
if not ticker:
    st.error("Please select at least one stock.")

st.write(f"Selected Tickers: {ticker}")

start_date = st.date_input('Start Date', )
end_date = st.date_input('End Date')
# for t in ticker:
#     df = get_stock_returns(t)
#     st.line_chart(df)
df = pd.DataFrame()

for t in ticker:
    price = get_stock_price(t)
    df[t] = price

st.line_chart(df)



if __name__ == "__main__":
    logging.info("Hello World")
    print("Hello World")

    try:
        a = 1/0
    except Exception as e:
        logging.info(CustomException(e,sys))
        print(CustomException(e,sys))
        raise CustomException(e,sys)