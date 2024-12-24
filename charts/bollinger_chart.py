# charts/bollinger_chart.py
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

@st.cache_data
def create_bollinger_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        name='Price',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_Upper'],
        name='Bollinger Upper',
        line=dict(color='red', dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_MA'],
        name='Bollinger MA',
        line=dict(color='orange', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_Lower'],
        name='Bollinger Lower',
        line=dict(color='green', dash='dot')
    ))
    fig.update_layout(title="Bollinger Bands")
    return fig
