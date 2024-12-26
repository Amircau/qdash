# charts/yearly_minmax_chart.py
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

@st.cache_data
def create_yearly_min_max_chart(df: pd.DataFrame) -> go.Figure:
    """
    Expects a DataFrame with columns: ['Year', 'min', 'max'] 
    as returned by financial_data.compute_yearly_min_max().
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    fig = go.Figure()

    # Add a Bar trace for MIN values (red)
    fig.add_trace(go.Bar(
        x=df['Year'],         # X-axis: each calendar year
        y=df['min'],          # Y-axis: the min price
        name='MIN',
        marker_color='red'
    ))

    # Add a Bar trace for MAX values (blue)
    fig.add_trace(go.Bar(
        x=df['Year'],
        y=df['max'],
        name='MAX',
        marker_color='blue'
    ))

    # Update layout
    fig.update_layout(
        title="Yearly Min / Max Price",
        xaxis_title="Year",
        yaxis_title="Price",
        barmode='group',  # Show min and max side by side
        legend=dict(
            x=0.8,
            y=1.1,
            orientation='h'
        ),
    )

    return fig
