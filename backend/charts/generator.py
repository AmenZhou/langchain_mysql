"""
Chart Generator

Creates interactive Plotly visualizations based on chart configurations.
Handles different chart types with optimized styling and data processing.
"""

from typing import Dict, List, Any
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import logging

from .models import ChartType, ChartConfig, ChartResult, ChartSettings
from .exceptions import ChartRenderingError, ChartConfigurationError

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generates actual chart visualizations using Plotly."""
    
    def __init__(self, settings: ChartSettings = None):
        self.settings = settings or ChartSettings()
        
        # Set default matplotlib style for fallback usage
        try:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        except:
            # Fallback if seaborn style is not available
            pass
    
    def generate_chart(self, data: List[Dict[str, Any]], config: ChartConfig) -> ChartResult:
        """
        Generate a chart based on the configuration.
        
        Args:
            data: List of data dictionaries
            config: Chart configuration
            
        Returns:
            ChartResult with chart data and metadata
        """
        try:
            if not data:
                raise ChartConfigurationError("No data provided for chart generation")
            
            df = self._prepare_dataframe(data)
            if df.empty:
                raise ChartConfigurationError("No non-PII data available for visualization")
            
            # Validate configuration
            self._validate_config(df, config)
            
            # Generate chart based on type
            chart_data = self._generate_chart_by_type(df, config)
            
            return ChartResult(
                chart_type=config.chart_type,
                plotly_json=chart_data['plotly_json'],
                config=config,
                data_summary=chart_data['data_summary'],
                success=True
            )
                
        except Exception as e:
            logger.error(f"Error generating {config.chart_type} chart: {str(e)}")
            return ChartResult(
                chart_type=config.chart_type,
                plotly_json={},
                config=config,
                data_summary={},
                success=False,
                error_message=str(e)
            )
    
    def _prepare_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare DataFrame by removing PII columns."""
        df = pd.DataFrame(data)
        
        # Remove private columns
        columns_to_remove = []
        for col in df.columns:
            if df[col].astype(str).eq("[PRIVATE]").all():
                columns_to_remove.append(col)
        
        if columns_to_remove:
            df = df.drop(columns=columns_to_remove)
            
        return df
    
    def _validate_config(self, df: pd.DataFrame, config: ChartConfig):
        """Validate chart configuration against available data."""
        if config.x_axis not in df.columns:
            raise ChartConfigurationError(f"X-axis column '{config.x_axis}' not found in data")
        
        if config.y_axis and config.y_axis not in df.columns:
            raise ChartConfigurationError(f"Y-axis column '{config.y_axis}' not found in data")
    
    def _generate_chart_by_type(self, df: pd.DataFrame, config: ChartConfig) -> Dict[str, Any]:
        """Generate chart based on type."""
        chart_generators = {
            ChartType.BAR: self._generate_bar_chart,
            ChartType.LINE: self._generate_line_chart,
            ChartType.PIE: self._generate_pie_chart,
            ChartType.SCATTER: self._generate_scatter_plot,
            ChartType.HISTOGRAM: self._generate_histogram,
            ChartType.TIME_SERIES: self._generate_time_series
        }
        
        generator = chart_generators.get(config.chart_type)
        if not generator:
            raise ChartRenderingError(f"Unsupported chart type: {config.chart_type}")
        
        return generator(df, config)
    
    def _generate_bar_chart(self, df: pd.DataFrame, config: ChartConfig) -> Dict[str, Any]:
        """Generate a bar chart using Plotly."""
        try:
            # Group and aggregate data
            if config.y_axis:
                grouped = df.groupby(config.x_axis)[config.y_axis].agg(['sum', 'mean', 'count']).reset_index()
                y_column = 'sum'
                y_label = config.y_axis.title()
            else:
                # Count occurrences if no y-axis specified
                grouped = df[config.x_axis].value_counts().reset_index()
                grouped.columns = [config.x_axis, 'count']
                y_column = 'count'
                y_label = 'Count'
            
            fig = px.bar(
                grouped, 
                x=config.x_axis, 
                y=y_column,
                title=config.title,
                labels={config.x_axis: config.x_axis.title(), y_column: y_label}
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=400,
                showlegend=False,
                plot_bgcolor='white',
                font=dict(size=12)
            )
            
            data_summary = {
                "total_categories": len(grouped),
                "total_value": float(grouped[y_column].sum()),
                "average_value": float(grouped[y_column].mean()),
                "max_value": float(grouped[y_column].max()),
                "chart_type_info": "Bar chart showing categorical data comparison"
            }
            
            return {
                "plotly_json": json.loads(fig.to_json()),
                "data_summary": data_summary
            }
            
        except Exception as e:
            raise ChartRenderingError(f"Failed to generate bar chart: {str(e)}")
    
    def _generate_line_chart(self, df: pd.DataFrame, config: ChartConfig) -> Dict[str, Any]:
        """Generate a line chart using Plotly."""
        try:
            fig = px.line(
                df, 
                x=config.x_axis, 
                y=config.y_axis,
                title=config.title,
                markers=True
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='white',
                font=dict(size=12)
            )
            
            data_summary = {
                "data_points": len(df),
                "min_value": float(df[config.y_axis].min()),
                "max_value": float(df[config.y_axis].max()),
                "average_value": float(df[config.y_axis].mean()),
                "trend": self._calculate_trend(df[config.y_axis]),
                "chart_type_info": "Line chart showing trends over time or sequence"
            }
            
            return {
                "plotly_json": json.loads(fig.to_json()),
                "data_summary": data_summary
            }
            
        except Exception as e:
            raise ChartRenderingError(f"Failed to generate line chart: {str(e)}")
    
    def _generate_pie_chart(self, df: pd.DataFrame, config: ChartConfig) -> Dict[str, Any]:
        """Generate an enhanced pie chart using Plotly."""
        try:
            value_counts = df[config.x_axis].value_counts()
            total_items = value_counts.sum()
            
            # Calculate percentages for better insights
            percentages = (value_counts / total_items * 100).round(1)
            
            # Create color palette - use distinct colors for better visibility
            colors = px.colors.qualitative.Set3[:len(value_counts)]
            
            # Create the pie chart with enhanced styling
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=config.title,
                color_discrete_sequence=colors
            )
            
            # Enhanced styling and formatting
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>' +
                             'Count: %{value}<br>' +
                             'Percentage: %{percent}<br>' +
                             '<extra></extra>',
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            
            fig.update_layout(
                height=500,  # Slightly taller for better readability
                font=dict(size=12),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01
                ),
                margin=dict(l=20, r=120, t=60, b=20)  # More space for legend
            )
            
            # Calculate additional insights
            sorted_percentages = percentages.sort_values(ascending=False)
            
            # Determine if data is well-distributed or concentrated
            top_3_share = sorted_percentages.head(3).sum() if len(sorted_percentages) >= 3 else sorted_percentages.sum()
            concentration = "high" if top_3_share > 75 else "medium" if top_3_share > 50 else "low"
            
            data_summary = {
                "total_categories": len(value_counts),
                "total_items": int(total_items),
                "largest_category": str(value_counts.index[0]),
                "largest_count": int(value_counts.iloc[0]),
                "largest_percentage": float(percentages.iloc[0]),
                "smallest_category": str(value_counts.index[-1]),
                "smallest_count": int(value_counts.iloc[-1]),
                "smallest_percentage": float(percentages.iloc[-1]),
                "top_3_categories_share": float(top_3_share),
                "data_concentration": concentration,
                "diversity_index": float(1 - sum((percentages/100)**2)),  # Simpson's diversity index
                "category_breakdown": [
                    {
                        "category": str(name),
                        "count": int(count),
                        "percentage": float(pct)
                    }
                    for name, count, pct in zip(value_counts.index, value_counts.values, percentages.values)
                ],
                "chart_type_info": f"Pie chart showing distribution of {config.x_axis} across {len(value_counts)} categories"
            }
            
            return {
                "plotly_json": json.loads(fig.to_json()),
                "data_summary": data_summary
            }
            
        except Exception as e:
            raise ChartRenderingError(f"Failed to generate pie chart: {str(e)}")
    
    def _generate_scatter_plot(self, df: pd.DataFrame, config: ChartConfig) -> Dict[str, Any]:
        """Generate a scatter plot using Plotly."""
        try:
            fig = px.scatter(
                df,
                x=config.x_axis,
                y=config.y_axis,
                title=config.title
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='white',
                font=dict(size=12)
            )
            
            # Calculate correlation if both columns are numeric
            correlation = None
            try:
                if df[config.x_axis].dtype in [np.number] and df[config.y_axis].dtype in [np.number]:
                    correlation = float(df[config.x_axis].corr(df[config.y_axis]))
            except:
                pass
            
            data_summary = {
                "data_points": len(df),
                "correlation": correlation,
                "x_range": [float(df[config.x_axis].min()), float(df[config.x_axis].max())],
                "y_range": [float(df[config.y_axis].min()), float(df[config.y_axis].max())],
                "chart_type_info": "Scatter plot showing relationship between two variables"
            }
            
            return {
                "plotly_json": json.loads(fig.to_json()),
                "data_summary": data_summary
            }
            
        except Exception as e:
            raise ChartRenderingError(f"Failed to generate scatter plot: {str(e)}")
    
    def _generate_histogram(self, df: pd.DataFrame, config: ChartConfig) -> Dict[str, Any]:
        """Generate a histogram using Plotly."""
        try:
            fig = px.histogram(
                df,
                x=config.x_axis,
                title=config.title,
                nbins=min(20, len(df[config.x_axis].unique()))
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='white',
                font=dict(size=12)
            )
            
            data_summary = {
                "data_points": len(df),
                "min_value": float(df[config.x_axis].min()),
                "max_value": float(df[config.x_axis].max()),
                "mean": float(df[config.x_axis].mean()),
                "std": float(df[config.x_axis].std()),
                "unique_values": len(df[config.x_axis].unique()),
                "chart_type_info": "Histogram showing distribution of numerical values"
            }
            
            return {
                "plotly_json": json.loads(fig.to_json()),
                "data_summary": data_summary
            }
            
        except Exception as e:
            raise ChartRenderingError(f"Failed to generate histogram: {str(e)}")
    
    def _generate_time_series(self, df: pd.DataFrame, config: ChartConfig) -> Dict[str, Any]:
        """Generate a time series chart using Plotly."""
        try:
            # Try to convert to datetime
            df_copy = df.copy()
            try:
                df_copy[config.x_axis] = pd.to_datetime(df_copy[config.x_axis])
            except:
                # If conversion fails, use as-is
                pass
            
            fig = px.line(
                df_copy,
                x=config.x_axis,
                y=config.y_axis,
                title=config.title,
                markers=True
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='white',
                font=dict(size=12)
            )
            
            trend = self._calculate_trend(df_copy[config.y_axis])
            
            data_summary = {
                "data_points": len(df_copy),
                "time_range": {
                    "start": str(df_copy[config.x_axis].min()),
                    "end": str(df_copy[config.x_axis].max())
                },
                "value_range": [float(df_copy[config.y_axis].min()), float(df_copy[config.y_axis].max())],
                "trend": trend,
                "chart_type_info": "Time series chart showing temporal trends"
            }
            
            return {
                "plotly_json": json.loads(fig.to_json()),
                "data_summary": data_summary
            }
            
        except Exception as e:
            raise ChartRenderingError(f"Failed to generate time series chart: {str(e)}")
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction for time series data."""
        try:
            # Simple trend calculation using correlation with index
            correlation = series.corr(pd.Series(range(len(series))))
            if correlation > 0.1:
                return "increasing"
            elif correlation < -0.1:
                return "decreasing"
            else:
                return "stable"
        except:
            return "unknown"
    
    def get_supported_chart_types(self) -> List[str]:
        """Get list of supported chart types."""
        return [chart_type.value for chart_type in ChartType]
    
    def validate_data_for_chart_type(self, data: List[Dict[str, Any]], chart_type: ChartType) -> Dict[str, Any]:
        """Validate if data is suitable for a specific chart type."""
        try:
            df = self._prepare_dataframe(data)
            if df.empty:
                return {"valid": False, "reason": "No non-PII data available"}
            
            validation_rules = {
                ChartType.BAR: lambda: len(df.select_dtypes(include=['object']).columns) > 0,
                ChartType.LINE: lambda: len(df.select_dtypes(include=[np.number]).columns) >= 1,
                ChartType.PIE: lambda: len(df.select_dtypes(include=['object']).columns) > 0,
                ChartType.SCATTER: lambda: len(df.select_dtypes(include=[np.number]).columns) >= 2,
                ChartType.HISTOGRAM: lambda: len(df.select_dtypes(include=[np.number]).columns) >= 1,
                ChartType.TIME_SERIES: lambda: len(df.select_dtypes(include=[np.number]).columns) >= 1
            }
            
            validator = validation_rules.get(chart_type)
            if validator and validator():
                return {"valid": True, "reason": f"Data is suitable for {chart_type.value} chart"}
            else:
                return {"valid": False, "reason": f"Data requirements not met for {chart_type.value} chart"}
                
        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {str(e)}"} 