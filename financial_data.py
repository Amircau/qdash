# Directory Structure
# ├── my-financial-dashboard/
# │   ├── app.py                  # Main Streamlit application
# │   ├── financial_data.py       # Data fetching and manipulation logic
# │   ├── dashboard_visualizer.py # Visualization components
# │   ├── config.py              # Configuration and constants
# │   ├── requirements.txt        # Python dependencies
# │   ├── README.md               # Project documentation
# │   ├── .gitignore              # Ignored files for Git

# financial_data.py
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from datetime import datetime

def debug_log(message, value=None):
    """Log debug information."""
    print(f"DEBUG: {message}")
    if value is not None:
        print(value)

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

    def _fetch_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch and clean data from yfinance"""
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            debug_log("Fetched data", df.head())

            column_mapping = {
                'Close': 'close',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Volume': 'volume'
            }
            return df.rename(columns=column_mapping)
        except Exception as e:
            debug_log("Error fetching data", e)
            raise

    def add_momentum_indicators(self) -> None:
        """Add momentum indicators to the dataframe"""
        for period in Config.MOVING_AVERAGE_PERIODS:
            self.df[f'MA{period}'] = self.df['close'].rolling(window=period).mean()
        debug_log("Added Moving Averages", self.df.head())

    def add_bollinger_bands(self, window: int = 20, num_std: int = 2) -> None:
        """
        Calculate Bollinger Bands using a rolling window and specified # of standard deviations.
        """
        self.df['BB_MA'] = self.df['close'].rolling(window=window).mean()
        self.df['BB_STD'] = self.df['close'].rolling(window=window).std()
        self.df['BB_Upper'] = self.df['BB_MA'] + (num_std * self.df['BB_STD'])
        self.df['BB_Lower'] = self.df['BB_MA'] - (num_std * self.df['BB_STD'])
        debug_log("Added Bollinger Bands", self.df.head())
