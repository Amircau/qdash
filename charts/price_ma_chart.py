# charts/price_ma_chart.py
import plotly.express as px
import streamlit as st
import pandas as pd

@st.cache_data
def create_price_ma_difference_chart(df: pd.DataFrame) -> go.Figure:
    df_temp = df.copy()
    df_temp['Price_MA_Diff'] = df_temp['close'] - df_temp['MA21']
    return px.line(
        df_temp,
        x=df_temp.index,
        y='Price_MA_Diff',
        title="Price - MA Difference"
    )