# Enhancements to momentum_chart.py
# Add new methods for RSI and MACD

import plotly.graph_objects as go
from typing import List

@st.cache_data
def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
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
