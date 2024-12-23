import pandas as pd
import streamlit as st
from financial_data import FinancialData, Config  # Import Config correctly
from dashboard_visualizer import DashboardVisualizer, debug_log

def main():
    st.set_page_config(page_title="Enhanced Financial Dashboard", layout="wide")
    st.title("Enhanced Financial Dashboard")

    with st.sidebar:
        st.header("Configuration")
        primary_ticker = st.text_input("Primary Ticker:", value="AAPL")
        st.write("DEBUG: Config.DEFAULT_START_DATE", Config.DEFAULT_START_DATE)  # Debugging line
        start_date = st.date_input("Start Date", value=pd.to_datetime(Config.DEFAULT_START_DATE))
        end_date = st.date_input("End Date", value=pd.to_datetime("today"))

    try:
        primary_data = FinancialData(primary_ticker, start_date, end_date)
        primary_data.add_momentum_indicators()
        primary_data.add_bollinger_bands()

        debug_log("Primary Data after processing", primary_data.df.head())

        viz = DashboardVisualizer(primary_data)

        st.plotly_chart(viz.create_bollinger_chart(), use_container_width=True)
        st.plotly_chart(viz.create_momentum_chart(), use_container_width=True)
        st.plotly_chart(viz.create_price_ma_difference_chart(), use_container_width=True)
        st.plotly_chart(viz.create_yearly_min_max_chart(), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        debug_log("Application Error", e)

if __name__ == "__main__":
    main()
