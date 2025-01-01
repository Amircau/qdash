# charts/yearly_minmax_chart.py
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

@st.cache_data
def create_yearly_min_max_chart(df: pd.DataFrame) -> go.Figure:
    """
    Expects a DataFrame with columns: ['WeekOfYear', 'AvgMinPercent', 'AvgMaxPercent'].
    Creates a 2-line chart showing the average Min% and Max% for each WeekOfYear.
    """
    df = df.copy()

    # Use Plotly Express line chart for simplicity
    fig = px.line(
        df,
        x='WeekOfYear',
        y=['AvgMinPercent', 'AvgMaxPercent'],
        title="Avg Weekly Min/Max % from Year Open"
    )

    fig.update_layout(
        xaxis_title="Week of Year (1..52)",
        yaxis_title="Percent (%)",
        legend_title="",
    )
    return fig
