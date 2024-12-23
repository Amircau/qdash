import plotly.graph_objects as go
import plotly.express as px
from financial_data import FinancialData, debug_log

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
