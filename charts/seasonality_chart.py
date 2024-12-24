# charts/seasonality_chart.py
import plotly.express as px
import streamlit as st
import pandas as pd

@st.cache_data
def create_seasonality_chart(seasonality: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        seasonality,
        x='week_of_year',
        y='avg_4w_return',
        title="Seasonality (4W Return)"
    )
    fig.update_yaxes(tickformat=".2%")
    return fig