import plotly.graph_objects as go
import streamlit as st
import pandas as pd

@st.cache_data
def create_yearly_min_max_chart(df: pd.DataFrame) -> go.Figure:
    # Process yearly returns for week 1
    df['Year'] = df.index.year
    df['Week'] = df.index.isocalendar().week
    
    # Get first week data for each year
    week1_data = df[df['Week'] == 1].copy()
    week1_data['4W_RETURN'] = week1_data['4W_RETURN'] * 100  # Convert to percentage
    
    max_return = week1_data['4W_RETURN'].max()
    min_return = week1_data['4W_RETURN'].min()
    
    fig = go.Figure()
    
    # Add bars for returns
    fig.add_trace(go.Bar(
        x=week1_data['Year'],
        y=week1_data['4W_RETURN'],
        name='4W Returns',
        marker_color=week1_data['4W_RETURN'].apply(
            lambda x: 'red' if x < 0 else 'green'
        )
    ))
    
    # Add reference lines
    fig.add_hline(y=max_return, line_dash="dash", line_color="green", 
                  annotation_text=f"Max: {max_return:.1f}%")
    fig.add_hline(y=min_return, line_dash="dash", line_color="red", 
                  annotation_text=f"Min: {min_return:.1f}%")
    
    fig.update_layout(
        title="Week 1 Returns by Year",
        yaxis_title="4-Week Return (%)",
        xaxis_title="Year",
        showlegend=False,
        yaxis_tickformat='.1f'
    )
    
    return fig
