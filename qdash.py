# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from dataclasses import dataclass

@dataclass
class Config:
    MOVING_AVERAGE_PERIODS = [10, 21, 63]
    DEFAULT_START_DATE = "1960-01-01"

class FinancialData:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.df = self._fetch_data(ticker, start_date, end_date)

    def _fetch_data(self, ticker, start_date, end_date):
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        return df.rename(columns={
            'Close': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        })

    def add_bollinger_bands(self, window=20, num_std=2):
        self.df['BB_MA'] = self.df['close'].rolling(window=window).mean()
        self.df['BB_STD'] = self.df['close'].rolling(window=window).std()
        self.df['BB_Upper'] = self.df['BB_MA'] + num_std * self.df['BB_STD']
        self.df['BB_Lower'] = self.df['BB_MA'] - num_std * self.df['BB_STD']

class DashboardVisualizer:
    def __init__(self, data):
        self.data = data

    def create_bollinger_chart(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.df.index, y=self.data.df['close'], name='Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.data.df.index, y=self.data.df['BB_Upper'], name='Bollinger Upper', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=self.data.df.index, y=self.data.df['BB_Lower'], name='Bollinger Lower', line=dict(color='green', dash='dot')))
        fig.update_layout(title="Bollinger Bands")
        return fig

def main():
    st.set_page_config(page_title="Financial Dashboard", layout="wide")
    st.title("Financial Dashboard")

    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Ticker Symbol", value="AAPL")
        start_date = st.date_input("Start Date", pd.to_datetime(Config.DEFAULT_START_DATE))
        end_date = st.date_input("End Date", pd.to_datetime("today"))

    try:
        data = FinancialData(ticker, start_date, end_date)
        data.add_bollinger_bands()

        viz = DashboardVisualizer(data)
        st.plotly_chart(viz.create_bollinger_chart(), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
