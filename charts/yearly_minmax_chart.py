# charts/yearly_minmax_chart.py
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

@st.cache_data
def create_yearly_min_max_chart(df: pd.DataFrame) -> go.Figure:
    """
    Expects a DataFrame with columns: ['Week', 'Min', 'Max']
    """
    df = df.copy()

    fig = go.Figure()

    # Min bars (red)
    fig.add_trace(go.Bar(
        x=df['Week'],
        y=df['Min'],
        name='MIN',
        marker_color='red'
    ))

    # Max bars (blue)
    fig.add_trace(go.Bar(
        x=df['Week'],
        y=df['Max'],
        name='MAX',
        marker_color='blue'
    ))

    fig.update_layout(
        title="Yearly Min \\ Max Price",
        xaxis_title="DATE",
        yaxis_title="Price / Return",
        barmode='group',  # side-by-side
        legend=dict(
            x=0.9, 
            y=1.1,
            orientation='h'
        ),
    )

    return fig
