"""
Dashboard visualization components using Plotly for SupplyChainOptix.

This module provides reusable chart components for rendering supply chain
metrics, predictions, and anomaly detection visualizations.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta


class ChartComponents:
    """Collection of reusable Plotly chart components for supply chain visualization."""
    
    # Color scheme for consistency across charts
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff9800',
        'info': '#17a2b8',
        'neutral': '#7f7f7f',
        'prediction': '#9467bd',
        'anomaly': '#e377c2'
    }
    
    @staticmethod
    def create_disruption_timeline(
        df: pd.DataFrame,
        date_col: str = 'date',
        disruption_col: str = 'disruption_probability',
        title: str = "Supply Chain Disruption Forecast"
    ) -> go.Figure:
        """
        Create a timeline chart showing disruption probabilities over time.
        
        Args:
            df: DataFrame with date and disruption probability columns
            date_col: Name of the date column
            disruption_col: Name of the disruption probability column
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return ChartComponents._create_empty_chart(title)
        
        fig = go.Figure()
        
        # Add probability line
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[disruption_col],
            mode='lines+markers',
            name='Disruption Probability',
            line=dict(color=ChartComponents.COLORS['danger'], width=2),
            marker=dict(size=6),
            hovertemplate='%{x|%Y-%m-%d}<br>Probability: %{y:.2%}<extra></extra>'
        ))
        
        # Add threshold zones
        fig.add_hrect(
            y0=0, y1=0.3,
            fillcolor=ChartComponents.COLORS['success'],
            opacity=0.1,
            line_width=0,
            annotation_text="Low Risk",
            annotation_position="right"
        )
        fig.add_hrect(
            y0=0.3, y1=0.6,
            fillcolor=ChartComponents.COLORS['warning'],
            opacity=0.1,
            line_width=0,
            annotation_text="Medium Risk",
            annotation_position="right"
        )
        fig.add_hrect(
            y0=0.6, y1=1.0,
            fillcolor=ChartComponents.COLORS['danger'],
            opacity=0.1,
            line_width=0,
            annotation_text="High Risk",
            annotation_position="right"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Disruption Probability",
            yaxis=dict(tickformat='.0%', range=[0, 1]),
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_inventory_heatmap(
        df: pd.DataFrame,
        product_col: str = 'product',
        location_col: str = 'location',
        value_col: str = 'stock_level',
        title: str = "Inventory Levels by Location"
    ) -> go.Figure:
        """
        Create a heatmap showing inventory levels across products and locations.
        
        Args:
            df: DataFrame with product, location, and stock level data
            product_col: Name of the product column
            location_col: Name of the location column
            value_col: Name of the value column to display
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return ChartComponents._create_empty_chart(title)
        
        # Pivot data for heatmap
        pivot_df = df.pivot_table(
            index=product_col,
            columns=location_col,
            values=value_col,
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlGn',
            hovertemplate='Product: %{y}<br>Location: %{x}<br>Stock: %{z:.0f}<extra></extra>',
            colorbar=dict(title="Stock Level")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Location",
            yaxis_title="Product",
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_anomaly_scatter(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        anomaly_col: str = 'is_anomaly',
        title: str = "Anomaly Detection",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None
    ) -> go.Figure:
        """
        Create a scatter plot highlighting anomalies in the data.
        
        Args:
            df: DataFrame with feature and anomaly columns
            x_col: Name of the x-axis column
            y_col: Name of the y-axis column
            anomaly_col: Name of the boolean anomaly indicator column
            title: Chart title
            x_label: Custom x-axis label
            y_label: Custom y-axis label
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return ChartComponents._create_empty_chart(title)
        
        fig = go.Figure()
        
        # Normal points
        normal_df = df[df[anomaly_col] == False]
        fig.add_trace(go.Scatter(
            x=normal_df[x_col],
            y=normal_df[y_col],
            mode='markers',
            name='Normal',
            marker=dict(
                color=ChartComponents.COLORS['primary'],
                size=8,
                opacity=0.6
            ),
            hovertemplate=f'{x_label or x_col}: %{{x}}<br>{y_label or y_col}: %{{y}}<extra></extra>'
        ))
        
        # Anomaly points
        anomaly_df = df[df[anomaly_col] == True]
        if not anomaly_df.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_df[x_col],
                y=anomaly_df[y_col],
                mode='markers',
                name='Anomaly',
                marker=dict(
                    color=ChartComponents.COLORS['anomaly'],
                    size=12,
                    symbol='x',
                    line=dict(width=2)
                ),
                hovertemplate=f'{x_label or x_col}: %{{x}}<br>{y_label or y_col}: %{{y}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label or x_col,
            yaxis_title=y_label or y_col,
            template='plotly_white',
            height=450,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_supplier_performance_bar(
        df: pd.DataFrame,
        supplier_col: str = 'supplier',
        metric_col: str = 'on_time_delivery_rate',
        title: str = "Supplier Performance Metrics"
    ) -> go.Figure:
        """
        Create a horizontal bar chart for supplier performance comparison.
        
        Args:
            df: DataFrame with supplier and performance metrics
            supplier_col: Name of the supplier column
            metric_col: Name of the metric column to display
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return ChartComponents._create_empty_chart(title)
        
        # Sort by metric value
        df_sorted = df.sort_values(metric_col, ascending=True)
        
        # Color code based on performance threshold
        colors = [
            ChartComponents.COLORS['danger'] if val < 0.7
            else ChartComponents.COLORS['warning'] if val < 0.85
            else ChartComponents.COLORS['success']
            for val in df_sorted[metric_col]
        ]
        
        fig = go.Figure(go.Bar(
            y=df_sorted[supplier_col],
            x=df_sorted[metric_col],
            orientation='h',
            marker=dict(color=colors),
            hovertemplate='%{y}<br>Performance: %{x:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Performance Score",
            yaxis_title="Supplier",
            xaxis=dict(tickformat='.0%'),
            template='plotly_white',
            height=max(300, len(df_sorted) * 30)
        )
        
        return fig
    
    @staticmethod
    def create_prediction_confidence_band(
        df: pd.DataFrame,
        date_col: str = 'date',
        actual_col: str = 'actual',
        predicted_col: str = 'predicted',
        lower_bound_col: str = 'lower_bound',
        upper_bound_col: str = 'upper_bound',
        title: str = "Demand Forecast with Confidence Intervals"
    ) -> go.Figure:
        """
        Create a line chart with confidence bands for predictions.
        
        Args:
            df: DataFrame with dates, actual values, predictions, and bounds
            date_col: Name of the date column
            actual_col: Name of the actual values column
            predicted_col: Name of the predicted values column
            lower_bound_col: Name of the lower confidence bound column
            upper_bound_col: Name of the upper confidence bound column
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return ChartComponents._create_empty_chart(title)
        
        fig = go.Figure()
        
        # Add confidence band
        if lower_bound_col in df.columns and upper_bound_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[upper_bound_col],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[lower_bound_col],
                mode='lines',
                line=dict(width=0),
                fillcolor=ChartComponents.COLORS['prediction'],
                fill='tonexty',
                opacity=0.2,
                name='Confidence Interval',
                hovertemplate='95% CI: %{y:.2f}<extra></extra>'
            ))
        
        # Add predicted values
        if predicted_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[predicted_col],
                mode='lines',