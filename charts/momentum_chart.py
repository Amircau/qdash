# charts/momentum_chart.py
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from typing import List

@st.cache_data
def create_momentum_chart(df: pd.DataFrame, ma_periods: List[int]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        name='Price',
        line=dict(color='blue')
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
        line=dict(color='red')
    ))
    for period in ma_periods:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f'MOMO_MA{period}'],
            name=f'MOMO MA{period}',
            line=dict(dash='dot', color='green')
        ))
    fig.update_layout(
        title="Momentum Score and Moving Averages",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Momentum Score", overlaying="y", side="right")
    )
    return fig