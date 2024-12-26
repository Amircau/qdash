# charts/yearly_minmax_chart.py
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

@st.cache_data
def create_yearly_min_max_chart(df: pd.DataFrame) -> go.Figure:
    """
    df is expected to have the following columns (rename as needed):
      - 'WeekNumber': e.g., W1, W2, W3, ...
      - 'MinValue':   numeric values (negative or positive)
      - 'MaxValue':   numeric values (negative or positive)
    """

    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Create the figure
    fig = go.Figure()

    # Add MIN bars (red)
    fig.add_trace(go.Bar(
        x=df['WeekNumber'],     # e.g., W1, W2, ...
        y=df['MinValue'],
        name='MIN',
        marker_color='red'
    ))

    # Add MAX bars (blue)
    fig.add_trace(go.Bar(
        x=df['WeekNumber'],
        y=df['MaxValue'],
        name='MAX',
        marker_color='blue'
    ))

    # Customize the layout
    fig.update_layout(
        title="Yearly Min \\ Max Price",
        xaxis_title="DATE (Week)",
        yaxis_title="Price / Return",
        barmode='group',   # shows MIN and MAX side by side for each week
        legend=dict(
            x=0.9,
            y=1.1,
            orientation='h'
        )
    )

    # Optionally adjust y-axis range if desired
    # fig.update_yaxes(range=[-1, 1])

    return fig
