# data_manager.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
import streamlit as st

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

    def __init__(self, ticker: str, start_date: datetime, end_date: datetime):
        self.ticker = ticker
        self.df = self._fetch_data(ticker, start_date, end_date)

    @staticmethod
    @st.cache_data  
    def _fetch_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
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
            self.df[f'MA{period}'] = self.df['close'].rolling(window=period).mean()

        for period in Config.ROC_PERIODS:
            self.df[f'ROC{period}'] = (
                (self.df['close'] - self.df['close'].shift(period)) /
                self.df['close'].shift(period)
            ) * 100

        roc_cols = [f'ROC{period}' for period in Config.ROC_PERIODS]
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        self.df['MOMO_SCORE'] = np.average(self.df[roc_cols], axis=1, weights=weights)

        for period in Config.MOVING_AVERAGE_PERIODS:
            self.df[f'MOMO_MA{period}'] = self.df['MOMO_SCORE'].rolling(window=period).mean()

    def compute_rolling_return(self) -> None:
        """Compute 4-week rolling return"""
        weeks = Config.ROLLING_WINDOW_WEEKS
        shift_days = weeks * 5
        self.df['4W_RETURN'] = (self.df['close'] / self.df['close'].shift(shift_days)) - 1

    def compute_seasonality(self) -> pd.DataFrame:
        """Group by week-of-year and compute average 4W return"""
        self.df['week_of_year'] = self.df.index.isocalendar().week
        self.df['week_of_year'] = self.df['week_of_year'].apply(lambda x: 52 if x > 52 else x)
        return self.df.groupby('week_of_year')['4W_RETURN'].mean().reset_index(name='avg_4w_return')

    def compute_yearly_min_max(self) -> pd.DataFrame:
        """Compute yearly min and max prices"""
        self.df['Year'] = self.df.index.year
        return self.df.groupby('Year')['close'].agg(['min', 'max']).reset_index()

    def add_bollinger_bands(self, window: int = 20, num_std: int = 2) -> None:
        """Calculate Bollinger Bands"""
        self.df['BB_MA'] = self.df['close'].rolling(window=window).mean()
        self.df['BB_STD'] = self.df['close'].rolling(window=window).std()
        self.df['BB_Upper'] = self.df['BB_MA'] + (num_std * self.df['BB_STD'])
        self.df['BB_Lower'] = self.df['BB_MA'] - (num_std * self.df['BB_STD'])
