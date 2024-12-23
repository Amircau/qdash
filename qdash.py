# --------------------------------------------------------------------------------
# Configuration and Constants
# --------------------------------------------------------------------------------
import streamlit as st
import polars as pl
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

@dataclass
class Config:
    """Configuration settings for the dashboard"""
    MOVING_AVERAGE_PERIODS = [10, 21, 63]
    ROC_PERIODS = [10, 21, 63, 252]
    ROLLING_WINDOW_WEEKS = 4
    STD_DEV_BANDS = [2, 3]
    DEFAULT_START_DATE = "1960-01-01"

# --------------------------------------------------------------------------------
# Data Models
# --------------------------------------------------------------------------------
class FinancialData:
    """Class to handle all financial data operations"""

    def __init__(self, ticker: str, start_date: datetime, end_date: datetime):
        self.ticker = ticker
        self.df = self._fetch_data(ticker, start_date, end_date)

    def _fetch_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pl.DataFrame:
        """Fetch and clean data from yfinance"""
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
    
            # Reset index to include the date
            data.reset_index(inplace=True)
    
            # Rename columns to match the expected format
            column_mapping = {
                'Date': 'Date',
                'Close': 'close',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Volume': 'volume'
            }
            data.rename(columns=column_mapping, inplace=True)
    
            # Convert to Polars DataFrame
            pl_df = pl.DataFrame(data)
    
            # Ensure 'Date' is parsed as datetime
            pl_df = pl_df.with_columns(
                pl.col('Date').str.strptime(pl.Datetime, fmt="%Y-%m-%d").alias('Date')
            )
            return pl_df
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            raise


    def add_momentum_indicators(self) -> None:
        """Add momentum indicators to the dataframe"""
        try:
            # Add Moving Averages
            for period in Config.MOVING_AVERAGE_PERIODS:
                self.df = self.df.with_columns(
                    self.df['close'].rolling_mean(window_size=period).alias(f'MA{period}')
                )

            # Add Rate of Change (ROC)
            for period in Config.ROC_PERIODS:
                self.df = self.df.with_columns(
                    ((self.df['close'] - self.df['close'].shift(period)) /
                     self.df['close'].shift(period) * 100).alias(f'ROC{period}')
                )

            # Calculate MOMO_SCORE
            roc_cols = [f'ROC{period}' for period in Config.ROC_PERIODS]
            self.df = self.df.with_columns(
                pl.concat_list([self.df[col] for col in roc_cols]).mean(axis=1).alias('MOMO_SCORE')
            )
        except Exception as e:
            st.error(f"Error adding momentum indicators: {e}")
            raise

    def add_bollinger_bands(self, window: int = 20, num_std: int = 2) -> None:
        """Calculate Bollinger Bands"""
        try:
            self.df = self.df.with_columns([
                self.df['close'].rolling_mean(window_size=window).alias('BB_MA'),
                self.df['close'].rolling_std(window_size=window).alias('BB_STD'),
                (self.df['BB_MA'] + num_std * self.df['BB_STD']).alias('BB_Upper'),
                (self.df['BB_MA'] - num_std * self.df['BB_STD']).alias('BB_Lower')
            ])
        except Exception as e:
            st.error(f"Error calculating Bollinger Bands: {e}")
            raise

    def compute_rolling_return(self) -> None:
        """Compute 4-week rolling return"""
        try:
            shift_days = Config.ROLLING_WINDOW_WEEKS * 5
            self.df = self.df.with_columns(
                ((self.df['close'] / self.df['close'].shift(shift_days)) - 1).alias('4W_RETURN')
            )
        except Exception as e:
            st.error(f"Error calculating rolling return: {e}")
            raise

    def compute_seasonality(self) -> pl.DataFrame:
        """Group by week-of-year and compute average 4W return"""
        try:
            self.df = self.df.with_columns(
                self.df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[1]).alias('week_of_year')
            )
            return self.df.groupby('week_of_year').agg(
                pl.col('4W_RETURN').mean().alias('avg_4w_return')
            )
        except Exception as e:
            st.error(f"Error calculating seasonality: {e}")
            raise

    def compute_yearly_min_max(self) -> pl.DataFrame:
        """Compute yearly min and max prices"""
        try:
            self.df = self.df.with_columns(
                self.df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').year).alias('Year')
            )
            return self.df.groupby('Year').agg([
                pl.col('close').min().alias('min'),
                pl.col('close').max().alias('max')
            ])
        except Exception as e:
            st.error(f"Error calculating yearly min and max: {e}")
            raise

# --------------------------------------------------------------------------------
# Visualization Components
# --------------------------------------------------------------------------------
class DashboardVisualizer:
    """Class to handle all visualization logic"""

    def __init__(self, primary_data: FinancialData, secondary_data: Optional[FinancialData] = None):
        self.primary = primary_data
        self.secondary = secondary_data

    def create_momentum_chart(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.primary.df['Date'], y=self.primary.df['close'], name='Price', line=dict(color='blue')))
        for period in Config.MOVING_AVERAGE_PERIODS:
            fig.add_trace(go.Scatter(x=self.primary.df['Date'], y=self.primary.df[f'MA{period}'], name=f'MA{period}', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=self.primary.df['Date'], y=self.primary.df['MOMO_SCORE'], name='Momentum Score', line=dict(color='red')))
        return fig

    def create_price_ma_difference_chart(self) -> go.Figure:
        self.primary.df = self.primary.df.with_columns((self.primary.df['close'] - self.primary.df['MA21']).alias('Price_MA_Diff'))
        fig = px.line(self.primary.df, x='Date', y='Price_MA_Diff', title="Price - MA Difference")
        return fig

# --------------------------------------------------------------------------------
# Main Application
# --------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Enhanced Financial Dashboard", layout="wide")

    # Sidebar for controls
    with st.sidebar:
        st.header("Configuration")
        primary_ticker = st.text_input("Primary Ticker:", value="AAPL")
        secondary_ticker = st.text_input("Secondary Ticker (optional):", value="")
        start_date = st.date_input("Start Date", value=datetime.strptime(Config.DEFAULT_START_DATE, '%Y-%m-%d'))
        end_date = st.date_input("End Date", value=datetime.today())

    try:
        # Primary Data
        primary_data = FinancialData(primary_ticker, start_date, end_date)
        primary_data.add_momentum_indicators()
        primary_data.add_bollinger_bands()

        viz = DashboardVisualizer(primary_data)
        st.title("Enhanced Financial Dashboard")
        st.plotly_chart(viz.create_momentum_chart(), use_container_width=True)
        st.plotly_chart(viz.create_price_ma_difference_chart(), use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
