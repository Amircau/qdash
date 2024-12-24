# qdash.py
import streamlit as st
import pandas as pd
from datetime import datetime
from data_manager import FinancialData, Config
from charts import (
    momentum_chart,
    price_ma_chart,
    seasonality_chart,
    yearly_minmax_chart,
    comparison_charts,
    bollinger_chart
)

# Set page config
st.set_page_config(page_title="Enhanced Financial Dashboard", layout="wide")
st.title("Enhanced Financial Dashboard")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    primary_ticker = st.text_input("Primary Ticker:", value="AAPL")
    secondary_ticker = st.text_input("Secondary Ticker (optional):", value="")
    start_date = st.date_input("Start Date", value=pd.to_datetime(Config.DEFAULT_START_DATE))
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

try:
    # Initialize data with error handling
    with st.spinner('Fetching data...'):
        # Primary data initialization
        primary_data = FinancialData(primary_ticker, start_date, end_date)
        primary_data.add_momentum_indicators()
        primary_data.compute_rolling_return()
        primary_data.add_bollinger_bands(window=20, num_std=2)

        # Secondary data initialization (if provided)
        secondary_data = None
        if secondary_ticker:
            secondary_data = FinancialData(secondary_ticker, start_date, end_date)
            secondary_data.add_momentum_indicators()
            secondary_data.compute_rolling_return()

        # Display charts in columns
        col1, col2 = st.columns(2)
        with col1:
            fig_momentum = momentum_chart.create_momentum_chart(
                primary_data.df, 
                Config.MOVING_AVERAGE_PERIODS
            )
            st.plotly_chart(fig_momentum, use_container_width=True)
        
        with col2:
            fig_price_ma = price_ma_chart.create_price_ma_difference_chart(
                primary_data.df
            )
            st.plotly_chart(fig_price_ma, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            seasonality_data = primary_data.compute_seasonality()
            fig_seasonality = seasonality_chart.create_seasonality_chart(
                seasonality_data
            )
            st.plotly_chart(fig_seasonality, use_container_width=True)
        
        with col4:
            yearly_data = primary_data.compute_yearly_min_max()
            fig_yearly = yearly_minmax_chart.create_yearly_min_max_chart(
                yearly_data
            )
            st.plotly_chart(fig_yearly, use_container_width=True)

        # Comparison charts (if secondary ticker provided)
        if secondary_data:
            fig_gap, fig_ratio = comparison_charts.create_comparison_charts(
                primary_data.df,
                secondary_data.df
            )
            if fig_gap is not None:
                st.plotly_chart(fig_gap, use_container_width=True)
            if fig_ratio is not None:
                st.plotly_chart(fig_ratio, use_container_width=True)

        # Bollinger Bands chart
        fig_bollinger = bollinger_chart.create_bollinger_chart(
            primary_data.df
        )
        st.plotly_chart(fig_bollinger, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
