# --------------------------------------------------------------------------------
# Configuration and Imports
# --------------------------------------------------------------------------------
import streamlit as st
import polars as pl
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Config:
    """Configuration settings for the dashboard"""
    MOVING_AVERAGE_PERIODS = [10, 21, 63]
    ROC_PERIODS = [10, 21, 63, 252]
    ROLLING_WINDOW_WEEKS = 4
    STD_DEV_BANDS = [2, 3]
    DEFAULT_START_DATE = "1960-01-01"

# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------
def debug_log(message, value=None):
    """Log debug information to the console."""
    print(f"DEBUG: {message}")
    if value is not None:
        print(value)

def flatten_yfinance_columns(df):
    """
    Flattens the multi-index columns from yfinance into a single level,
    e.g., ('Close', 'AAPL') becomes 'Close_AAPL'.
    """
    if df.columns.nlevels > 1:
        # Flatten multi-level columns
        df.columns = [
            "_".join([str(level).strip() for level in col if level])
            for col in df.columns.values
        ]
    return df

def standardize_columns(df):
    """
    Renames flattened yfinance columns into standardized names:
    Date -> Date
    Close or Close_<TICKER> -> close
    Open  or Open_<TICKER>  -> open
    High  or High_<TICKER>  -> high
    Low   or Low_<TICKER>   -> low
    Volume or Volume_<TICKER> -> volume
    (Ignores or keeps any other columns that might appear, e.g. Adj Close)
    """
    rename_dict = {}
    for col in df.columns:
        col_lower = col.lower()
        if "date" in col_lower:
            rename_dict[col] = "Date"
        elif "close" in col_lower and "adj" not in col_lower:
            rename_dict[col] = "close"
        elif "open" in col_lower:
            rename_dict[col] = "open"
        elif "high" in col_lower:
            rename_dict[col] = "high"
        elif "low" in col_lower:
            rename_dict[col] = "low"
        elif "volume" in col_lower:
            rename_dict[col] = "volume"
        else:
            # keep the column name as is if it doesn't match
            rename_dict[col] = col
    df.rename(columns=rename_dict, inplace=True)
    return df

# --------------------------------------------------------------------------------
# Data Models
# --------------------------------------------------------------------------------
class FinancialData:
    """Class to handle all financial data operations"""

    def __init__(self, ticker: str, start_date: datetime, end_date: datetime):
        self.ticker = ticker
        self.df = self._fetch_data(ticker, start_date, end_date)

    def _fetch_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pl.DataFrame:
        """Fetch and clean data from yfinance, flatten columns, rename, convert to Polars."""
        try:
            # 1. Download data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                raise ValueError(f"No data found for ticker: {ticker}")

            # 2. Flatten multi-level columns if needed
            data = flatten_yfinance_columns(data)

            # 3. Standardize column names: close, open, high, low, volume, Date
            data = standardize_columns(data)

            # 4. Ensure 'Date' becomes a column (reset index)
            data.reset_index(inplace=True)
            if 'Date' not in data.columns:
                raise KeyError("'Date' column is missing after flattening and resetting index.")

            # 5. Convert to Polars DataFrame
            pl_df = pl.DataFrame(data)
            debug_log("Polars DataFrame columns", pl_df.columns)

            # 6. Convert Date column to Polars datetime if needed
            if "Date" in pl_df.columns and pl_df["Date"].dtype != pl.Datetime:
                pl_df = pl_df.with_columns(
                    pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d").alias("Date")
                )

            debug_log("Final Polars DataFrame head", pl_df.head())
            return pl_df

        except Exception as e:
            debug_log("Error fetching or processing data", e)
            raise

    def add_momentum_indicators(self) -> None:
        """Add momentum indicators: MAs, ROC, Momentum Score."""
        try:
            # Moving Averages
            for period in Config.MOVING_AVERAGE_PERIODS:
                ma_name = f"MA{period}"
                self.df = self.df.with_columns(
                    self.df["close"].rolling_mean(window_size=period).alias(ma_name)
                )

            # Rate of Change (ROC)
            for period in Config.ROC_PERIODS:
                roc_col = f"ROC{period}"
                self.df = self.df.with_columns(
                    (
                        (self.df["close"] - self.df["close"].shift(period))
                        / self.df["close"].shift(period)
                        * 100
                    ).alias(roc_col)
                )

            # Momentum Score (avg of all ROC columns)
            roc_cols = [f"ROC{period}" for period in Config.ROC_PERIODS]
            self.df = self.df.with_columns(
                pl.concat_list([self.df[col] for col in roc_cols]).mean(axis=1).alias("MOMO_SCORE")
            )

        except Exception as e:
            st.error(f"Error adding momentum indicators: {e}")
            raise

    def add_bollinger_bands(self, window: int = 20, num_std: int = 2) -> None:
        """Calculate Bollinger Bands."""
        try:
            self.df = self.df.with_columns([
                self.df["close"].rolling_mean(window_size=window).alias("BB_MA"),
                self.df["close"].rolling_std(window_size=window).alias("BB_STD"),
                (pl.col("BB_MA") + num_std * pl.col("BB_STD")).alias("BB_Upper"),
                (pl.col("BB_MA") - num_std * pl.col("BB_STD")).alias("BB_Lower"),
            ])
        except Exception as e:
            st.error(f"Error calculating Bollinger Bands: {e}")
            raise

    def compute_rolling_return(self) -> None:
        """Compute 4-week rolling return."""
        try:
            shift_days = Config.ROLLING_WINDOW_WEEKS * 5  # approximate 5 trading days/week
            self.df = self.df.with_columns(
                ((self.df["close"] / self.df["close"].shift(shift_days)) - 1).alias("4W_RETURN")
            )
        except Exception as e:
            st.error(f"Error calculating rolling return: {e}")
            raise

    def compute_seasonality(self) -> pl.DataFrame:
        """Group by week-of-year and compute average 4W return."""
        try:
            # Polars datetime means we can directly call x.isocalendar()[1]
            self.df = self.df.with_columns(
                pl.col("Date").apply(lambda x: x.isocalendar()[1] if x else None).alias("week_of_year")
            )
            return self.df.groupby("week_of_year").agg(
                pl.col("4W_RETURN").mean().alias("avg_4w_return")
            )
        except Exception as e:
            st.error(f"Error calculating seasonality: {e}")
            raise

    def compute_yearly_min_max(self) -> pl.DataFrame:
        """Compute yearly min and max close."""
        try:
            self.df = self.df.with_columns(
                pl.col("Date").apply(lambda x: x.year if x else None).alias("Year")
            )
            return self.df.groupby("Year").agg([
                pl.col("close").min().alias("min"),
                pl.col("close").max().alias("max"),
            ])
        except Exception as e:
            st.error(f"Error calculating yearly min and max: {e}")
            raise

# --------------------------------------------------------------------------------
# Visualization Components
# --------------------------------------------------------------------------------
class DashboardVisualizer:
    """Handles all Plotly visualizations for the dashboard."""

    def __init__(self, primary_data: FinancialData, secondary_data: Optional[FinancialData] = None):
        self.primary = primary_data
        self.secondary = secondary_data

    def create_momentum_chart(self) -> go.Figure:
        """Price chart with MAs and Momentum Score."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.primary.df["Date"], y=self.primary.df["close"],
            name="Price", line=dict(color="blue")
        ))

        # Add each moving average as a separate line
        for period in Config.MOVING_AVERAGE_PERIODS:
            fig.add_trace(go.Scatter(
                x=self.primary.df["Date"], y=self.primary.df[f"MA{period}"],
                name=f"MA{period}", line=dict(dash="dot")
            ))

        # Plot Momentum Score
        fig.add_trace(go.Scatter(
            x=self.primary.df["Date"], y=self.primary.df["MOMO_SCORE"],
            name="Momentum Score", line=dict(color="red")
        ))
        fig.update_layout(title="Price & Momentum Indicators")
        return fig

    def create_price_ma_difference_chart(self) -> go.Figure:
        """
        Creates a line chart of (Price - MA21).
        """
        if "Price_MA_Diff" not in self.primary.df.columns:
            self.primary.df = self.primary.df.with_columns(
                (self.primary.df["close"] - self.primary.df["MA21"]).alias("Price_MA_Diff")
            )

        # Convert to pandas for px.line
        df_pd = self.primary.df.to_pandas()
        fig = px.line(df_pd, x="Date", y="Price_MA_Diff", title="Price - MA21 Difference")
        return fig

    def create_bollinger_chart(self) -> go.Figure:
        """Bollinger Bands + Price."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.primary.df["Date"], y=self.primary.df["close"],
            name="Price", line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=self.primary.df["Date"], y=self.primary.df["BB_Upper"],
            name="BB Upper", line=dict(color="red", dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=self.primary.df["Date"], y=self.primary.df["BB_MA"],
            name="BB MA", line=dict(color="green", dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=self.primary.df["Date"], y=self.primary.df["BB_Lower"],
            name="BB Lower", line=dict(color="red", dash="dot")
        ))
        fig.update_layout(title="Bollinger Bands")
        return fig

    def create_4w_return_chart(self) -> go.Figure:
        """4-Week Rolling Return line chart."""
        df_pd = self.primary.df.to_pandas()
        fig = px.line(df_pd, x="Date", y="4W_RETURN", title="4-Week Rolling Return")
        return fig

    def create_seasonality_chart(self, seasonality_df: pl.DataFrame) -> go.Figure:
        """Average 4W return by week of year."""
        df_pd = seasonality_df.to_pandas()
        fig = px.line(df_pd, x="week_of_year", y="avg_4w_return",
                      title="Average 4W Return by Week of Year")
        return fig

    def create_yearly_min_max_chart(self, yearly_df: pl.DataFrame) -> go.Figure:
        """Yearly min & max close prices."""
        df_pd = yearly_df.to_pandas()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_pd["Year"], y=df_pd["min"],
            name="Yearly Min", mode="lines+markers"
        ))
        fig.add_trace(go.Scatter(
            x=df_pd["Year"], y=df_pd["max"],
            name="Yearly Max", mode="lines+markers"
        ))
        fig.update_layout(title="Yearly Min and Max Close")
        return fig

# --------------------------------------------------------------------------------
# Main Streamlit Application
# --------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Enhanced Financial Dashboard", layout="wide")

    # Sidebar for controls
    with st.sidebar:
        st.header("Configuration")
        primary_ticker = st.text_input("Primary Ticker:", value="AAPL")
        secondary_ticker = st.text_input("Secondary Ticker (optional):", value="")
        start_date = st.date_input("Start Date", value=datetime.strptime(Config.DEFAULT_START_DATE, "%Y-%m-%d"))
        end_date = st.date_input("End Date", value=datetime.today())

    try:
        # Prepare primary data
        primary_data = FinancialData(primary_ticker, start_date, end_date)
        primary_data.add_momentum_indicators()
        primary_data.add_bollinger_bands()
        primary_data.compute_rolling_return()

        # Compute seasonality & yearly min/max
        seasonality_df = primary_data.compute_seasonality()
        yearly_df = primary_data.compute_yearly_min_max()

        # Build visualizations
        viz = DashboardVisualizer(primary_data)

        st.title("Enhanced Financial Dashboard")
        st.plotly_chart(viz.create_momentum_chart(), use_container_width=True)
        st.plotly_chart(viz.create_price_ma_difference_chart(), use_container_width=True)
        st.plotly_chart(viz.create_bollinger_chart(), use_container_width=True)
        st.plotly_chart(viz.create_4w_return_chart(), use_container_width=True)
        st.plotly_chart(viz.create_seasonality_chart(seasonality_df), use_container_width=True)
        st.plotly_chart(viz.create_yearly_min_max_chart(yearly_df), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
