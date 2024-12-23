# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import sem, t
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass

# Set page config at the very beginning
st.set_page_config(page_title="Enhanced Financial Dashboard", layout="wide")

@dataclass
class Config:
    """Configuration settings for the dashboard"""
    MOVING_AVERAGE_PERIODS = [10, 21, 63]
    ROC_PERIODS = [10, 21, 63, 252]
    ROLLING_WINDOW_WEEKS = 4
    STD_DEV_BANDS = [2, 3]
    DEFAULT_START_DATE = "1960-01-01"

class FinancialData:
    """Class to handle all financial data operations"""

    def __init__(_self, ticker: str, start_date: datetime, end_date: datetime):
        _self.ticker = ticker
        _self.df = _self._fetch_data(ticker, start_date, end_date)

    @st.cache_data  # Add caching to prevent repeated data fetching
    def _fetch_data(__self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch and clean data from yfinance"""
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")

        column_mapping = {
            'Close': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        }
        return df.rename(columns=column_mapping)

    def add_momentum_indicators(self) -> None:
        """Add momentum indicators to the dataframe"""
        for period in Config.MOVING_AVERAGE_PERIODS:
            _self.df[f'MA{period}'] = _self.df['close'].rolling(window=period).mean()

        for period in Config.ROC_PERIODS:
            _self.df[f'ROC{period}'] = (
                (_self.df['close'] - _self.df['close'].shift(period)) /
                _self.df['close'].shift(period)
            ) * 100

        roc_cols = [f'ROC{period}' for period in Config.ROC_PERIODS]
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        _self.df['MOMO_SCORE'] = np.average(_self.df[roc_cols], axis=1, weights=weights)

        for period in Config.MOVING_AVERAGE_PERIODS:
            _self.df[f'MOMO_MA{period}'] = _self.df['MOMO_SCORE'].rolling(window=period).mean()

    def compute_rolling_return(self) -> None:
        """Compute 4-week rolling return"""
        weeks = Config.ROLLING_WINDOW_WEEKS
        shift_days = weeks * 5
        _self.df['4W_RETURN'] = (_self.df['close'] / _self.df['close'].shift(shift_days)) - 1

    def compute_seasonality(self) -> pd.DataFrame:
        """Group by week-of-year and compute average 4W return"""
        _self.df['week_of_year'] = _self.df.index.isocalendar().week
        _self.df['week_of_year'] = _self.df['week_of_year'].apply(lambda x: 52 if x > 52 else x)
        return _self.df.groupby('week_of_year')['4W_RETURN'].mean().reset_index(name='avg_4w_return')

    def compute_yearly_min_max(self) -> pd.DataFrame:
        """Compute yearly min and max prices"""
        _self.df['Year'] = _self.df.index.year
        return _self.df.groupby('Year')['close'].agg(['min', 'max']).reset_index()

    def add_bollinger_bands(_self, window: int = 20, num_std: int = 2) -> None:
        """Calculate Bollinger Bands"""
        _self.df['BB_MA'] = _self.df['close'].rolling(window=window).mean()
        _self.df['BB_STD'] = _self.df['close'].rolling(window=window).std()
        _self.df['BB_Upper'] = _self.df['BB_MA'] + (num_std * _self.df['BB_STD'])
        _self.df['BB_Lower'] = _self.df['BB_MA'] - (num_std * _self.df['BB_STD'])

class DashboardVisualizer:
    """Class to handle all visualization logic"""

    def __init__(_self, primary_data: FinancialData, secondary_data: Optional[FinancialData] = None):
        _self.primary = primary_data
        _self.secondary = secondary_data

    @st.cache_data  # Add caching for charts
    def create_momentum_chart(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=_self.primary.df.index,
            y=_self.primary.df['close'],
            name='Price',
            line=dict(color='blue')
        ))
        for period in Config.MOVING_AVERAGE_PERIODS:
            fig.add_trace(go.Scatter(
                x=_self.primary.df.index,
                y=_self.primary.df[f'MA{period}'],
                name=f'{period}MA',
                line=dict(dash='dot')
            ))
        fig.add_trace(go.Scatter(
            x=_self.primary.df.index,
            y=_self.primary.df['MOMO_SCORE'],
            name='Momentum Score',
            yaxis='y2',
            line=dict(color='red')
        ))
        for period in Config.MOVING_AVERAGE_PERIODS:
            fig.add_trace(go.Scatter(
                x=_self.primary.df.index,
                y=_self.primary.df[f'MOMO_MA{period}'],
                name=f'MOMO MA{period}',
                line=dict(dash='dot', color='green')
            ))
        fig.update_layout(
            title="Momentum Score and Moving Averages",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Momentum Score", overlaying="y", side="right")
        )
        return fig

    @st.cache_data
    def create_price_ma_difference_chart(self) -> go.Figure:
        _self.primary.df['Price_MA_Diff'] = _self.primary.df['close'] - _self.primary.df['MA21']
        fig = px.line(
            _self.primary.df,
            x=_self.primary.df.index,
            y='Price_MA_Diff',
            title="Price - MA Difference"
        )
        return fig

    @st.cache_data
    def create_seasonality_chart(self) -> go.Figure:
        seasonality = _self.primary.compute_seasonality()
        fig = px.bar(
            seasonality,
            x='week_of_year',
            y='avg_4w_return',
            title="Seasonality (4W Return)"
        )
        fig.update_yaxes(tickformat=".2%")
        return fig

    @st.cache_data
    def create_yearly_min_max_chart(self) -> go.Figure:
        yearly = _self.primary.compute_yearly_min_max()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly['Year'],
            y=yearly['min'],
            name='Yearly Min',
            marker_color='red'
        ))
        fig.add_trace(go.Bar(
            x=yearly['Year'],
            y=yearly['max'],
            name='Yearly Max',
            marker_color='blue'
        ))
        fig.update_layout(
            title="Yearly Min and Max Prices",
            barmode='group'
        )
        return fig

    @st.cache_data
    def create_comparison_charts(self) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
        if not _self.secondary:
            return None, None

        df_perf = pd.DataFrame(index=_self.primary.df.index)
        df_perf['Primary'] = _self.primary.df['close'] / _self.primary.df['close'].iloc[0] - 1
        
        _self.secondary.df = _self.secondary.df.reindex(_self.primary.df.index, method='ffill')
        df_perf['Secondary'] = _self.secondary.df['close'] / _self.secondary.df['close'].iloc[0] - 1
        df_perf['Gap'] = df_perf['Primary'] - df_perf['Secondary']

        fig_gap = px.line(df_perf, x=df_perf.index, y='Gap', title="Performance Gap")
        fig_ratio = px.line(
            x=_self.primary.df.index,
            y=_self.primary.df['close'] / _self.secondary.df['close'],
            title="Ratio Spread"
        )
        return fig_gap, fig_ratio

    @st.cache_data
    def create_bollinger_chart(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=_self.primary.df.index,
            y=_self.primary.df['close'],
            name='Price',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=_self.primary.df.index,
            y=_self.primary.df['BB_Upper'],
            name='Bollinger Upper',
            line=dict(color='red', dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=_self.primary.df.index,
            y=_self.primary.df['BB_MA'],
            name='Bollinger MA',
            line=dict(color='orange', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=_self.primary.df.index,
            y=_self.primary.df['BB_Lower'],
            name='Bollinger Lower',
            line=dict(color='green', dash='dot')
        ))
        fig.update_layout(title="Bollinger Bands")
        return fig

# Sidebar and main app
st.title("Enhanced Financial Dashboard")

with st.sidebar:
    st.header("Configuration")
    primary_ticker = st.text_input("Primary Ticker:", value="AAPL")
    secondary_ticker = st.text_input("Secondary Ticker (optional):", value="")
    start_date = st.date_input("Start Date", value=pd.to_datetime(Config.DEFAULT_START_DATE))
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

try:
    # Initialize data and visualizations with error handling
    with st.spinner('Fetching data...'):
        primary_data = FinancialData(primary_ticker, start_date, end_date)
        primary_data.add_momentum_indicators()
        primary_data.compute_rolling_return()
        primary_data.add_bollinger_bands(window=20, num_std=2)

        secondary_data = None
        if secondary_ticker:
            secondary_data = FinancialData(secondary_ticker, start_date, end_date)
            secondary_data.add_momentum_indicators()
            secondary_data.compute_rolling_return()

        viz = DashboardVisualizer(primary_data, secondary_data)

        # Display charts in columns
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(viz.create_momentum_chart(), use_container_width=True)
        with col2:
            st.plotly_chart(viz.create_price_ma_difference_chart(), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(viz.create_seasonality_chart(), use_container_width=True)
        with col4:
            st.plotly_chart(viz.create_yearly_min_max_chart(), use_container_width=True)

        if secondary_data:
            fig_gap, fig_ratio = viz.create_comparison_charts()
            if fig_gap is not None:
                st.plotly_chart(fig_gap, use_container_width=True)
            if fig_ratio is not None:
                st.plotly_chart(fig_ratio, use_container_width=True)

        st.plotly_chart(viz.create_bollinger_chart(), use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
