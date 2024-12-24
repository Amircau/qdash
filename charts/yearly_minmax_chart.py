# charts/yearly_minmax_chart.py
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

@st.cache_data
def create_yearly_min_max_chart(yearly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly['Year'],
        y=yearly['min'],
        name='Yearly Min',
        marker_color='red'
    ))
    fig.add_trace(go.Bar(
        x=yearly['Year'],
        y=yearly['max'],
        name='Yearly Max',
        marker_color='blue'
    ))
    fig.update_layout(
        title="Yearly Min and Max Prices",
        barmode='group'
    )
    return fig