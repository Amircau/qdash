import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import sem, t
from datetime import datetime

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
    # Moving Averages
    df['MA10'] = df['close'].rolling(window=10).mean()
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
    Based on daily_return relative to mean ± N*std.
    """
    df['daily_return'] = df['close'].pct_change()
    mean_ret = df['daily_return'].mean()
    std_ret = df['daily_return'].std()

    # XU1 = daily_return > mean + 2*std
    df['XU1'] = df['daily_return'] > (mean_ret + 2 * std_ret)
    # XU2 = daily_return > mean + 3*std
    df['XU2'] = df['daily_return'] > (mean_ret + 3 * std_ret)
    # XD1 = daily_return < mean - 2*std
    df['XD1'] = df['daily_return'] < (mean_ret - 2 * std_ret)
    # XD2 = daily_return < (mean_ret - 3 * std_ret)
    df['XD2'] = df['daily_return'] < (mean_ret - 3 * std_ret)

    return df

def compute_rolling_return(df, weeks=4):
    """
    Creates a 4W_RETURN column = (close[t]/close[t-20] - 1).
    Also sets 'ROLLING_RETURN_4W' as a bar chart metric if you'd like to keep the older approach
    of 'rolling average of daily returns' for visual reference.
    """
    # 1) 4W_RETURN: total return over ~20 trading days
    df['4W_RETURN'] = np.nan
    shift_days = weeks * 5  # typically 4 weeks ~ 20 days
    for i in range(shift_days, len(df)):
        now_close = df['close'].iloc[i]
        prev_close = df['close'].iloc[i - shift_days]
        if pd.notna(now_close) and pd.notna(prev_close) and prev_close != 0:
            df.at[df.index[i], '4W_RETURN'] = (now_close / prev_close) - 1

    # 2) "ROLLING_RETURN_4W": a rolling average of daily returns, as originally coded
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
    Compute difference between Price and 21MA.
    """
    df['Price_MA_Diff'] = df['close'] - df['MA21']
    return df

def compute_stats(df):
    """
    Compute statistical metrics (mean, std, variance, etc.) for reference.
    """
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

def compute_seasonality_4w_return(df):
    """
    Group by week-of-year (1..52) using the '4W_RETURN' column,
    compute the average 4W return across all years for that week.
    Returns a DataFrame: ["week_of_year", "avg_4w_return"].
    """
    df['week_of_year'] = df.index.isocalendar().week
    df['week_of_year'] = df['week_of_year'].apply(lambda w: 52 if w > 52 else w)

    grouped = df.groupby('week_of_year')['4W_RETURN'].mean(numeric_only=True)
    seasonality = grouped.reset_index()
    seasonality.rename(columns={'4W_RETURN': 'avg_4w_return'}, inplace=True)
    return seasonality

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

    # Build yearly min/max for the primary
    df_yearly_primary = compute_yearly_min_max(df_primary)

    # 4W seasonality for primary
    seasonality_primary = compute_seasonality_4w_return(df_primary)

    # --------------------------------------------------------------------------
    # 2) Optionally Fetch & Process Secondary Ticker
    # --------------------------------------------------------------------------
    df_secondary = None
    stats_secondary = {}
    if secondary_ticker.strip():
        df_secondary = fetch_data(secondary_ticker.strip(), start_date, end_date)
        if df_secondary.empty:
            st.warning(f"No data found for {secondary_ticker} in the given date range.")
            df_secondary = None
        else:
            df_secondary = add_momentum_indicators(df_secondary)
            df_secondary = add_extreme_markers(df_secondary)
            df_secondary = compute_rolling_return(df_secondary, weeks=4)
            df_secondary = create_price_ma_difference(df_secondary)
            stats_secondary = compute_stats(df_secondary)

    # --------------------------------------------------------------------------
    # 1) Momentum Score
    # --------------------------------------------------------------------------
    fig_momentum = px.line()
    fig_momentum.add_scatter(
        x=df_primary.index,
        y=df_primary['close'],
        mode='lines',
        name='Close'
    )
    fig_momentum.add_scatter(
        x=df_primary.index,
        y=df_primary['MOMO_SCORE'],
        mode='lines',
        name='MOMO SCORE'
    )
    fig_momentum.add_scatter(
        x=df_primary.index,
        y=df_primary['MA10'],
        mode='lines',
        name='MMS 10MA'
    )
    fig_momentum.add_scatter(
        x=df_primary.index,
        y=df_primary['MA21'],
        mode='lines',
        name='MMS 21MA'
    )
    fig_momentum.update_layout(
        title="Momentum Score",
        xaxis_title="Date",
        yaxis_title="Value"
    )

    # --------------------------------------------------------------------------
    # 2) 4w Rolling Return (Avg)
    # --------------------------------------------------------------------------
    fig_rolling_return = px.bar(
        df_primary,
        x=df_primary.index,
        y='ROLLING_RETURN_4W',
        title="4w Rolling Return (Avg)"
    )
    fig_rolling_return.update_layout(
        xaxis_title="Date",
        yaxis_title="4W Return (Avg of daily returns)"
    )

    # --------------------------------------------------------------------------
    # 3) Performance Gap (only if secondary is available)
    # --------------------------------------------------------------------------
    fig_perf_gap = None
    if df_secondary is not None:
        df_perf_gap = pd.DataFrame(index=df_primary.index)
        df_perf_gap['PRIMARY_RET'] = df_primary['close'] / df_primary['close'].iloc[0] - 1
        # align secondary by reindexing
        df_secondary = df_secondary.reindex(df_primary.index, method="ffill")
        df_perf_gap['SECONDARY_RET'] = df_secondary['close'] / df_secondary['close'].iloc[0] - 1
        df_perf_gap['Performance_Gap'] = df_perf_gap['PRIMARY_RET'] - df_perf_gap['SECONDARY_RET']

        fig_perf_gap = px.line(
            df_perf_gap,
            x=df_perf_gap.index,
            y='Performance_Gap',
            title="Performance Gap"
        )
        fig_perf_gap.update_layout(
            xaxis_title="Date",
            yaxis_title="Performance Gap"
        )

    # --------------------------------------------------------------------------
    # 4) Price - MA Difference
    # --------------------------------------------------------------------------
    mean_diff = df_primary['Price_MA_Diff'].mean()
    std_diff = df_primary['Price_MA_Diff'].std()

    fig_price_ma_diff = go.Figure()
    fig_price_ma_diff.add_trace(
        go.Scatter(
            x=df_primary.index,
            y=df_primary['Price_MA_Diff'],
            mode='lines',
            name='Price - 21MA'
        )
    )
    # Horizontal bands for extreme zones
    fig_price_ma_diff.add_hrect(
        y0=mean_diff + 2*std_diff,
        y1=mean_diff + 3*std_diff,
        fillcolor="green",
        opacity=0.2,
        line_width=0
    )
    fig_price_ma_diff.add_hrect(
        y0=mean_diff - 3*std_diff,
        y1=mean_diff - 2*std_diff,
        fillcolor="red",
        opacity=0.2,
        line_width=0
    )
    fig_price_ma_diff.add_hline(y=mean_diff+2*std_diff,
                                line=dict(color='orange', dash='dash'),
                                name='XU1')
    fig_price_ma_diff.add_hline(y=mean_diff+3*std_diff,
                                line=dict(color='red', dash='dash'),
                                name='XU2')
    fig_price_ma_diff.add_hline(y=mean_diff-2*std_diff,
                                line=dict(color='orange', dash='dash'),
                                name='XD1')
    fig_price_ma_diff.add_hline(y=mean_diff-3*std_diff,
                                line=dict(color='red', dash='dash'),
                                name='XD2')
    fig_price_ma_diff.update_layout(
        title="Price - MA Difference",
        xaxis_title="Date",
        yaxis_title="Price - 21MA"
    )

    # --------------------------------------------------------------------------
    # 5) Yearly Min\Max Price
    # --------------------------------------------------------------------------
    fig_yearly_min_max = go.Figure()
    fig_yearly_min_max.add_trace(
        go.Bar(
            x=df_yearly_primary['Year'],
            y=df_yearly_primary['MIN'],
            name='MIN',
            marker_color='red'
        )
    )
    fig_yearly_min_max.add_trace(
        go.Bar(
            x=df_yearly_primary['Year'],
            y=df_yearly_primary['MAX'],
            name='MAX',
            marker_color='blue'
        )
    )
    fig_yearly_min_max.update_layout(
        title="Yearly Min \\ Max Price",
        xaxis_title="Year",
        yaxis_title="Price",
        barmode='group'
    )

    # --------------------------------------------------------------------------
    # 6) Ratio Spread (only if secondary is available)
    # --------------------------------------------------------------------------
    fig_ratio_spread = None
    if df_secondary is not None:
        df_ratio = pd.DataFrame(index=df_primary.index)
        df_ratio['RATIO'] = df_primary['close'] / df_secondary['close']
        fig_ratio_spread = px.line(
            df_ratio,
            x=df_ratio.index,
            y='RATIO',
            title="Ratio Spread"
        )
        fig_ratio_spread.update_layout(
            xaxis_title=f"{secondary_ticker}/Date",
            yaxis_title=f"{primary_ticker}/{secondary_ticker} Ratio"
        )

    # --------------------------------------------------------------------------
    # 7) 4W Return Seasonality (New Chart)
    # --------------------------------------------------------------------------
    fig_seasonality = px.bar(
        seasonality_primary,
        x="week_of_year",
        y="avg_4w_return",
        title="Seasonality: 4W Return",
        labels={
            "week_of_year": "Week of Year (1..52)",
            "avg_4w_return": "Avg 4W Return"
        }
    )
    # Format as percentages
    fig_seasonality.update_yaxes(tickformat=".2%")

    # --------------------------------------------------------------------------------
    # Layout: We'll place the first 6 charts in a 2x3 grid, then put
    # the new 4W Return Seasonality chart in a third row spanning full width.
    # --------------------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig_momentum, use_container_width=True)
    with col2:
        st.plotly_chart(fig_rolling_return, use_container_width=True)
    with col3:
        if fig_perf_gap is not None:
            st.plotly_chart(fig_perf_gap, use_container_width=True)
        else:
            st.markdown("**No Secondary Ticker**\nPerformance Gap skipped.")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.plotly_chart(fig_price_ma_diff, use_container_width=True)
    with col5:
        st.plotly_chart(fig_yearly_min_max, use_container_width=True)
    with col6:
        if fig_ratio_spread is not None:
            st.plotly_chart(fig_ratio_spread, use_container_width=True)
        else:
            st.markdown("**No Secondary Ticker**\nRatio Spread skipped.")

    st.subheader("Seasonality (4W Return)")
    st.plotly_chart(fig_seasonality, use_container_width=True)

    # --------------------------------------------------------------------------------
    # Show Stats
    # --------------------------------------------------------------------------------
    st.subheader("Additional Statistics")
    st.write(f"**{primary_ticker} Stats**:")
    st.json(stats_primary)

    if df_secondary is not None:
        st.write(f"**{secondary_ticker} Stats**:")
        st.json(stats_secondary)

    st.info("Done! Explore the charts above.")
