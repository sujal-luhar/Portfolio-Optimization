import streamlit as st
from datetime import datetime

from src.data.symbols import symbols
from src.data.stockdata import create_portfolio_dataframe

from src.data.summary_stats import summary_stats

from src.data.portfolio import plot_ef
from src.data.portfolio import gmv
from src.data.portfolio import msr

from src.data.cppi import run_cppi

from src.data.montecarlo import show_cppi

import pandas as pd
import numpy as np 

st.header("Portfolio Optimization and CPPI Strategy")
st.write("Sorry for the time it took to load the page. It's because matplotlib does not plot huge data very quickly. Please be patient.")
st.subheader("Choose Company by Symbol")
ticker = st.multiselect(
    "Choose Company by Symbol", symbols(), ['aapl', 'amzn', 'googl', 'wmt', 'xyl'], label_visibility="hidden"
    )
if not ticker:
    st.error("Please select at least one stock.")

col1, col2 = st.columns([1, 1])

start_date_default = datetime.strptime('2012-11-10', '%Y-%m-%d').date()
end_date_default = datetime.today().date()

start_date = col1.date_input('**Start Date**', value=start_date_default)
end_date = col2.date_input('**End Date**', value=end_date_default)

portfolio_df = create_portfolio_dataframe(ticker, start_date, end_date)
st.line_chart(portfolio_df)
returns = portfolio_df.pct_change().dropna()

st.write("#### Company Details")
st.dataframe(summary_stats(returns))




############################################################################################




st.write("#### Efficient Frontire")
checkbox_elements = ["CML","GMV","Equally Weighted"]

col1, col2, col3 = st.columns([1, 1, 5])
show_cml = col1.checkbox(checkbox_elements[0], value=True)
show_gmv = col2.checkbox(checkbox_elements[1])
show_ew = col3.checkbox(checkbox_elements[2])

er = summary_stats(returns)["Annualized Return"]
cov = returns.cov()

# Initialize session state variables if not defined
if "previous_show_cml" not in st.session_state:
    st.session_state.previous_show_cml = False
if "previous_show_gmv" not in st.session_state:
    st.session_state.previous_show_gmv = False
if "previous_show_ew" not in st.session_state:
    st.session_state.previous_show_ew = False

# Check if any checkbox value changes or if the initial run occurs
if st.session_state.previous_show_cml != show_cml or st.session_state.previous_show_gmv != show_gmv or st.session_state.previous_show_ew != show_ew:

    # Update the session state with current checkbox values
    st.session_state.previous_show_cml = show_cml
    st.session_state.previous_show_gmv = show_gmv
    st.session_state.previous_show_ew = show_ew

    fig = plot_ef(50, er, cov, style='.-', legend=False, show_cml=st.session_state.previous_show_cml, show_ew=st.session_state.previous_show_ew, show_gmv=st.session_state.previous_show_gmv, riskfree_rate=0.07)
    st.pyplot(fig)  
    
st.write("#### Weight of Companies in Portfolio")
company_names = list(portfolio_df.columns)

weights_gmv = gmv(returns.cov())
weights_msr = msr(0.07, er, cov)
weights_eq = np.repeat(1/len(company_names), len(company_names))

# Create the DataFrame with company names and weights
col1, col2, col3 = st.columns([1, 1, 1])
weights_df_gmv = pd.DataFrame({
    'Companies': company_names,
    'Weights': weights_gmv
})
col1.write("**GMV Diversification**")
col1.dataframe(weights_df_gmv)

weights_df_msr = pd.DataFrame({
    'Companies': company_names,
    'Weights': weights_msr
})
col2.write("**MSR Diversification**")
col2.dataframe(weights_df_msr)

weights_df_eq = pd.DataFrame({
    'Companies': company_names,
    'Weights': weights_eq
})
col3.write("**Equally Weighted Diversification**")
col3.dataframe(weights_df_eq)

st.write("#### GMV Portfolio Performance")
Optimized_portfolio = pd.Series(np.dot(portfolio_df.dropna(), weights_gmv), index=portfolio_df.index, name="Portfolio Return")
Opt_rets = Optimized_portfolio.pct_change().dropna()
risky_rets = 100*(1+Opt_rets).cumprod()
risky_rets.index.name = "Date"
st.line_chart(risky_rets)

st.write("#### GMV Portfolio Details")
st.dataframe(summary_stats(Opt_rets.to_frame()))




############################################################################################




st.write("#### Dynamic Allocation in Safe Assets and Risky Assets with CPPI")
cppi = run_cppi(Opt_rets, drawdown=0.20)
merge1 = pd.merge(cppi["Wealth"],cppi["Risky Wealth"], left_index=True, right_index=True)
merge2 = pd.merge(merge1,cppi["floorval"], left_index=True, right_index=True)
merge2.columns = ["Wealth", "Risky Wealth", "Floor"]
st.line_chart(merge2)

Risky_Allocation = pd.merge(cppi["Risk Budget"],cppi["Risky Allocation"], left_index=True, right_index=True)
Risky_Allocation.columns = ["Risk Budget", "Risky Allocation"]
st.line_chart(Risky_Allocation[["Risk Budget", "Risky Allocation"]])




############################################################################################




st.write("### Interactive Testing of CPPI strategy with Monte-Carlo Simulation")
st.write("Use CPPI parameters in sidebar to test the strategy with different scenarios")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Streamlit Sidebar UI components
st.sidebar.title('CPPI Parameters')
n_scenarios = st.sidebar.slider('Number of Scenarios', min_value=1, max_value=1000, step=5, value=50)
mu = st.sidebar.slider('Mean Return', min_value=0.0, max_value=1.0, step=0.01, value=0.07)
sigma = st.sidebar.slider('Standard Deviation', min_value=0.0, max_value=1.0, step=0.01, value=0.15)
floor = st.sidebar.slider('Floor', min_value=0.0, max_value=1.0, step=0.05, value=0.0)
m = st.sidebar.slider('Multiplier (m)', min_value=1.0, max_value=5.0, step=0.5, value=3.0)
riskfree_rate = st.sidebar.slider('Risk-Free Rate', min_value=0.0, max_value=0.10, step=0.01, value=0.03)
y_max = st.sidebar.number_input('Max of Y Axis', min_value=100, step=1, value=None)

# Display the interactive plot
fig = show_cppi(n_scenarios, mu, sigma, m, floor, riskfree_rate, y_max)
st.pyplot(fig)