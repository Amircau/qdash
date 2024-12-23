import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import sem, t
from datetime import datetime

# --------------------------------------------------------------------------------
# Control Flags
# --------------------------------------------------------------------------------
ENABLE_MA_PRICING = False
ENABLE_MOMENTUM_INDICATORS = True
ENABLE_EXTREME_MARKERS = True
ENABLE_ROLLING_RETURN = True
ENABLE_SEASONALITY = True
ENABLE_STATS = True

# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------
def fetch_data(ticker, start_date, end_date):
    """Fetch data from yfinance and rename columns for convenience."""
    df = yf.download(ticker, start=start_date, end=end_date)
    df.rename(
        columns={
            'Close': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        },
        inplace=True
    )
    return df

def add_momentum_indicators(df):
    """
    Add various technical and momentum indicators:
    - 10 / 21 / 63-day moving averages
    - Rate of Change (ROC) for multiple periods
    - MOMO_SCORE (average of the ROC values)
    """
    if ENABLE_MOMENTUM_INDICATORS:
        df['MA10'] = df['close'].rolling(window=10).mean()
        if ENABLE_MA_PRICING:
            df['MA21'] = df['close'].rolling(window=21).mean()
        df['MA63'] = df['close'].rolling(window=63).mean()

        # Rate of Change
        for period in [10, 21, 63, 252]:
            df[f'ROC{period}'] = (
                (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
            ) * 100

        # MOMO Score = average(ROC10, ROC21, ROC63, ROC252)
        df['MOMO_SCORE'] = df[['ROC10', 'ROC21', 'ROC63', 'ROC252']].mean(axis=1)

    return df

def add_extreme_markers(df):
    """
    Calculate daily returns and add flags:
    - XU1, XU2: Extreme Ups
    - XD1, XD2: Extreme Downs
    """
    if ENABLE_EXTREME_MARKERS:
        df['daily_return'] = df['close'].pct_change()
        mean_ret = df['daily_return'].mean()
        std_ret = df['daily_return'].std()

        # XU1 = daily_return > mean + 2*std
        df['XU1'] = df['daily_return'] > (mean_ret + 2 * std_ret)
        # XU2 = daily_return > mean + 3*std
        df['XU2'] = df['daily_return'] > (mean_ret + 3 * std_ret)
        # XD1 = daily_return < mean - 2*std
        df['XD1'] = df['daily_return'] < (mean_ret - 2 * std_ret)
        # XD2 = daily_return < (mean - 3*std)
        df['XD2'] = df['daily_return'] < (mean_ret - 3 * std_ret)

    return df

def compute_rolling_return(df, weeks=4):
    """
    Creates a 4W_RETURN column representing the total return over ~20 trading days (4 weeks).
    Also computes 'ROLLING_RETURN_4W' as the rolling average of daily returns.
    """
    if ENABLE_ROLLING_RETURN:
        shift_days = weeks * 5  # ~20 trading days for 4 weeks
        df['4W_RETURN'] = np.nan

        prev_close = df['close'].shift(shift_days)
        df['4W_RETURN'] = np.where(
            (pd.notna(df['close']) & pd.notna(prev_close) & (prev_close != 0)),
            (df['close'] / prev_close) - 1,
            np.nan
        )

        # 2) "ROLLING_RETURN_4W": a rolling average of daily returns
        df['daily_return'] = df['close'].pct_change()
        df['ROLLING_RETURN_4W'] = df['daily_return'].rolling(weeks * 5).mean()

    return df

def compute_yearly_min_max(df):
    """
    Return a DataFrame with columns 'Year', 'MIN', 'MAX' for the close price.
    """
    df['Year'] = df.index.year
    grouped = df.groupby('Year')['close'].agg(['min', 'max']).reset_index()
    grouped.rename(columns={'min': 'MIN', 'max': 'MAX'}, inplace=True)
    return grouped

def create_price_ma_difference(df):
    """
    Compute the difference between the 'close' price and the 21-day moving average (MA21).
    """
    if ENABLE_MA_PRICING:
        if 'close' in df.columns and 'MA21' in df.columns:
            df['Price_MA_Diff'] = df['close'] - df['MA21']
        else:
            df['Price_MA_Diff'] = np.nan
    return df

def compute_stats(df):
    """
    Compute statistical metrics (mean, std, variance, etc.) for reference.
    """
    if ENABLE_STATS:
        close_prices = df['close'].dropna()
        mean_val = close_prices.mean()
        std_val = close_prices.std()
        var_val = close_prices.var()
        skew_val = close_prices.skew()
        kurt_val = close_prices.kurt()
        min_val = close_prices.min()
        max_val = close_prices.max()
        sum_val = close_prices.sum()
        rng_val = max_val - min_val

        n = len(close_prices)
        if n > 1:
            std_err = sem(close_prices)
            margin = std_err * t.ppf((1 + 0.95) / 2., n - 1)
            ci_lower = mean_val - margin
            ci_upper = mean_val + margin
        else:
            ci_lower = mean_val
            ci_upper = mean_val

        stats = {
            'mean': mean_val,
            'std_dev': std_val,
            'variance': var_val,
            'skewness': skew_val,
            'kurtosis': kurt_val,
            'min': min_val,
            'max': max_val,
            'sum': sum_val,
            'range': rng_val,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper
        }
        return stats
    return {}

def compute_seasonality_4w_return(df):
    """
    Group by week-of-year (1..52) using the '4W_RETURN' column,
    compute the average 4W return across all years for that week.
    """
    if ENABLE_SEASONALITY:
        df['week_of_year'] = df.index.isocalendar().week
        df['week_of_year'] = df['week_of_year'].apply(lambda w: 52 if w > 52 else w)

        grouped = df.groupby('week_of_year')['4W_RETURN'].mean(numeric_only=True)
        seasonality = grouped.reset_index()
        seasonality.rename(columns={'4W_RETURN': 'avg_4w_return'}, inplace=True)
        return seasonality
    return pd.DataFrame()

# --------------------------------------------------------------------------------
# Streamlit App Layout
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Chart Financial Dashboard", layout="wide")

st.title("Multi-Chart Financial Dashboard")

primary_ticker = st.text_input("Stock:", value="TSLA")
secondary_ticker = st.text_input("Compare:", value="")
start_date = st.date_input("Start Date", value=pd.to_datetime("1960-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))

if st.button("Compare"):
    # --------------------------------------------------------------------------
    # 1) Fetch & Process Primary Ticker
    # --------------------------------------------------------------------------
    if not primary_ticker:
        st.error("Please enter a primary ticker.")
        st.stop()

    df_primary = fetch_data(primary_ticker, start_date, end_date)
    if df_primary.empty:
        st.error(f"No data found for {primary_ticker} in the given date range.")
        st.stop()

    df_primary = add_momentum_indicators(df_primary)
    df_primary = add_extreme_markers(df_primary)
    df_primary = compute_rolling_return(df_primary, weeks=4)
    df_primary = create_price_ma_difference(df_primary)
    stats_primary = compute_stats(df_primary)

    st.write("Primary DataFrame:")
    st.write(df_primary.head())

    # Show stats
    if ENABLE_STATS:
        st.subheader("Statistics")
        st.write(stats_primary)
