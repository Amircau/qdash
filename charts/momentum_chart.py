import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from typing import List

@st.cache_data
def create_momentum_chart(df: pd.DataFrame, ma_periods: List[int]) -> go.Figure:
    """
    Create a chart showing price and momentum indicators, such as moving averages and momentum score.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        name='Price',
        line=dict(color='light_blue')
    ))
    for period in ma_periods:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f'MA{period}'],
            name=f'{period}MA',
            line=dict(dash='dot')
        ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MOMO_SCORE'],
        name='Momentum Score',
        yaxis='y2',
        line=dict(color='pink')
    ))
    fig.update_layout(
        title="Momentum Score and Moving Averages",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Momentum Score", overlaying="y", side="right")
    )
    return fig

@st.cache_data
def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create an RSI chart with thresholds for overbought (70) and oversold (30) levels.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ))
    fig.update_layout(
        title="Relative Strength Index (RSI)",
        yaxis=dict(title="RSI Value", range=[0, 100]),
        shapes=[
            dict(type="line", x0=df.index.min(), x1=df.index.max(), y0=70, y1=70, line=dict(color="red", dash="dot")),
            dict(type="line", x0=df.index.min(), x1=df.index.max(), y0=30, y1=30, line=dict(color="green", dash="dot"))
        ]
    )
    return fig

@st.cache_data
def create_macd_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a MACD chart including MACD line, signal line, and histogram.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal_Line'],
        name='Signal Line',
        line=dict(color='orange', dash='dot')
    ))
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_Histogram'],
        name='MACD Histogram',
        marker_color=['green' if x >= 0 else 'red' for x in df['MACD_Histogram']]
    ))
    fig.update_layout(
        title="MACD (Moving Average Convergence Divergence)",
        yaxis=dict(title="MACD Value")
    )
    return fig
