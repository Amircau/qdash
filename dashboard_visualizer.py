# dashboard_visualizer.py
import plotly.graph_objects as go
import plotly.express as px
from financial_data import FinancialData, Config, debug_log

class DashboardVisualizer:
    """Class to handle all visualization logic"""

    def __init__(self, primary_data: FinancialData):
        self.primary = primary_data

    def create_bollinger_chart(self) -> go.Figure:
        """Create a Bollinger Bands chart for the primary data."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.primary.df.index,
            y=self.primary.df['close'],
            name='Price',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=self.primary.df.index,
            y=self.primary.df['BB_Upper'],
            name='Bollinger Upper',
            line=dict(color='red', dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=self.primary.df.index,
            y=self.primary.df['BB_Lower'],
            name='Bollinger Lower',
            line=dict(color='green', dash='dot')
        ))
        fig.update_layout(title="Bollinger Bands")
        debug_log("Created Bollinger Chart")
        return fig

    def create_momentum_chart(self) -> go.Figure:
        """Create a Momentum Chart including Moving Averages."""
        if 'MOMO_SCORE' not in self.primary.df.columns:
            debug_log("Error: 'MOMO_SCORE' not found in DataFrame")
            raise KeyError("'MOMO_SCORE' column is missing. Ensure add_momentum_indicators() is called.")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.primary.df.index,
            y=self.primary.df['close'],
            name='Price',
            line=dict(color='blue')
        ))
        for period in Config.MOVING_AVERAGE_PERIODS:
            fig.add_trace(go.Scatter(
                x=self.primary.df.index,
                y=self.primary.df[f'MA{period}'],
                name=f'{period}MA',
                line=dict(dash='dot')
            ))
        fig.add_trace(go.Scatter(
            x=self.primary.df.index,
            y=self.primary.df['MOMO_SCORE'],
            name='Momentum Score',
            yaxis='y2',
            line=dict(color='red')
        ))
        fig.update_layout(
            title="Momentum Score and Moving Averages",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Momentum Score", overlaying="y", side="right")
        )
        debug_log("Created Momentum Chart")
        return fig

    def create_price_ma_difference_chart(self) -> go.Figure:
        """Create a chart showing the difference between Price and Moving Average."""
        self.primary.df['Price_MA_Diff'] = self.primary.df['close'] - self.primary.df['MA21']
        fig = px.line(
            self.primary.df,
            x=self.primary.df.index,
            y='Price_MA_Diff',
            title="Price - MA Difference"
        )
        debug_log("Created Price - MA Difference Chart")
        return fig

    def create_yearly_min_max_chart(self) -> go.Figure:
        """Create a chart showing yearly minimum and maximum prices."""
        yearly = self.primary.df.groupby(self.primary.df.index.year)['close'].agg(['min', 'max']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly['index'],
            y=yearly['min'],
            name='Yearly Min',
            marker_color='red'
        ))
        fig.add_trace(go.Bar(
            x=yearly['index'],
            y=yearly['max'],
            name='Yearly Max',
            marker_color='blue'
        ))
        fig.update_layout(
            title="Yearly Min and Max Prices",
            barmode='group'
        )
        debug_log("Created Yearly Min-Max Chart")
        return fig

    def create_seasonality_chart(self) -> go.Figure:
        """Create a seasonality chart showing 4-week returns grouped by week of the year."""
        self.primary.df['week_of_year'] = self.primary.df.index.isocalendar().week
        self.primary.df['week_of_year'] = self.primary.df['week_of_year'].apply(lambda x: 52 if x > 52 else x)
        seasonality = self.primary.df.groupby('week_of_year')['4W_RETURN'].mean().reset_index(name='avg_4w_return')
        fig = px.bar(
            seasonality,
            x='week_of_year',
            y='avg_4w_return',
            title="Seasonality (4W Return)"
        )
        fig.update_yaxes(tickformat=".2%")
        debug_log("Created Seasonality Chart")
        return fig
