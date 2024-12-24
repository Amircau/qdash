# charts/comparison_charts.py
import plotly.express as px
import streamlit as st
import pandas as pd
from typing import Tuple, Optional

@st.cache_data
def create_comparison_charts(df_primary: pd.DataFrame, df_secondary: pd.DataFrame) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    df_perf = pd.DataFrame(index=df_primary.index)
    df_perf['Primary'] = df_primary['close'] / df_primary['close'].iloc[0] - 1
    
    df_secondary = df_secondary.reindex(df_primary.index, method='ffill')
    df_perf['Secondary'] = df_secondary['close'] / df_secondary['close'].iloc[0] - 1
    df_perf['Gap'] = df_perf['Primary'] - df_perf['Secondary']
    
    ratio = df_primary['close'] / df_secondary['close']
    
    fig_gap = px.line(df_perf, x=df_perf.index, y='Gap', title="Performance Gap")
    fig_ratio = px.line(x=ratio.index, y=ratio, title="Ratio Spread")
    return fig_gap, fig_ratio