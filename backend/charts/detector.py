"""
Chart Type Detector

Automatically detects the best chart types for given data based on
data characteristics, patterns, and visualization best practices.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
import logging

from .models import ChartType, ChartConfig, ChartSettings

logger = logging.getLogger(__name__)


class ChartTypeDetector:
    """Automatically detects the best chart type for given data."""
    
    def __init__(self, settings: ChartSettings = None):
        self.settings = settings or ChartSettings()
    
    def detect_best_charts(self, data: List[Dict[str, Any]]) -> List[ChartConfig]:
        """
        Detect the best chart types for the given data.
        
        Args:
            data: List of data dictionaries
            
        Returns:
            List of ChartConfig objects sorted by confidence score
        """
        if not data:
            return []
            
        try:
            df = self._prepare_dataframe(data)
            if df.empty:
                return []
                
            charts = []
            
            # Detect different chart types
            charts.extend(self._detect_bar_charts(df))
            charts.extend(self._detect_line_charts(df))
            charts.extend(self._detect_pie_charts(df))
            charts.extend(self._detect_scatter_plots(df))
            charts.extend(self._detect_histograms(df))
            charts.extend(self._detect_time_series(df))
            
            # Sort by confidence score and return top recommendations
            charts.sort(key=lambda x: x.confidence_score, reverse=True)
            return charts[:self.settings.max_chart_recommendations]
            
        except Exception as e:
            logger.error(f"Error detecting chart types: {e}")
            return []
    
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
    
    def _detect_bar_charts(self, df: pd.DataFrame) -> List[ChartConfig]:
        """Detect opportunities for bar charts."""
        charts = []
        
        categorical_cols = self._get_categorical_columns(df)
        numeric_cols = self._get_numeric_columns(df)
        
        for cat_col in categorical_cols:
            unique_count = len(df[cat_col].unique())
            if unique_count > self.settings.max_bar_categories:
                continue
                
            for num_col in numeric_cols:
                # Calculate confidence based on category count
                if unique_count <= 10:
                    confidence = 0.8
                elif unique_count <= 20:
                    confidence = 0.6
                else:
                    confidence = 0.4
                
                charts.append(ChartConfig(
                    chart_type=ChartType.BAR,
                    title=f"{num_col.title()} by {cat_col.title()}",
                    x_axis=cat_col,
                    y_axis=num_col,
                    confidence_score=confidence,
                    description=f"Bar chart showing {num_col} values grouped by {cat_col}"
                ))
                    
        return charts
    
    def _detect_line_charts(self, df: pd.DataFrame) -> List[ChartConfig]:
        """Detect opportunities for line charts."""
        charts = []
        
        numeric_cols = self._get_numeric_columns(df)
        
        for col in df.columns:
            if self._is_sequential_column(df, col):
                for num_col in numeric_cols:
                    if col != num_col:
                        charts.append(ChartConfig(
                            chart_type=ChartType.LINE,
                            title=f"{num_col.title()} Over {col.title()}",
                            x_axis=col,
                            y_axis=num_col,
                            confidence_score=0.7,
                            description=f"Line chart showing trend of {num_col} over {col}"
                        ))
                        
        return charts
    
    def _detect_pie_charts(self, df: pd.DataFrame) -> List[ChartConfig]:
        """Detect opportunities for pie charts."""
        charts = []
        
        categorical_cols = self._get_categorical_columns(df)
        
        for col in categorical_cols:
            unique_count = len(df[col].unique())
            if unique_count > self.settings.max_pie_categories:
                continue
                
            # Calculate confidence based on category count (pie charts work best with fewer categories)
            if 2 <= unique_count <= 5:
                confidence = 0.9
            elif unique_count <= 8:
                confidence = 0.6
            else:
                continue
                
            charts.append(ChartConfig(
                chart_type=ChartType.PIE,
                title=f"Distribution of {col.title()}",
                x_axis=col,
                confidence_score=confidence,
                description=f"Pie chart showing distribution of {col} values"
            ))
                
        return charts
    
    def _detect_scatter_plots(self, df: pd.DataFrame) -> List[ChartConfig]:
        """Detect opportunities for scatter plots."""
        charts = []
        
        numeric_cols = self._get_numeric_columns(df)
        
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    # Check for reasonable correlation potential
                    try:
                        correlation = abs(df[col1].corr(df[col2]))
                        confidence = 0.6 + (correlation * 0.3)  # Boost confidence for correlated data
                    except:
                        confidence = 0.6
                    
                    charts.append(ChartConfig(
                        chart_type=ChartType.SCATTER,
                        title=f"{col1.title()} vs {col2.title()}",
                        x_axis=col1,
                        y_axis=col2,
                        confidence_score=min(confidence, 0.9),
                        description=f"Scatter plot showing relationship between {col1} and {col2}"
                    ))
                    
        return charts
    
    def _detect_histograms(self, df: pd.DataFrame) -> List[ChartConfig]:
        """Detect opportunities for histograms."""
        charts = []
        
        numeric_cols = self._get_numeric_columns(df)
        
        for col in numeric_cols:
            unique_count = len(df[col].unique())
            if unique_count >= self.settings.min_histogram_unique_values:
                # Higher confidence for more varied data
                if unique_count > 20:
                    confidence = 0.6
                elif unique_count > 10:
                    confidence = 0.5
                else:
                    confidence = 0.4
                    
                charts.append(ChartConfig(
                    chart_type=ChartType.HISTOGRAM,
                    title=f"Distribution of {col.title()}",
                    x_axis=col,
                    confidence_score=confidence,
                    description=f"Histogram showing distribution of {col} values"
                ))
                
        return charts
    
    def _detect_time_series(self, df: pd.DataFrame) -> List[ChartConfig]:
        """Detect opportunities for time series charts."""
        charts = []
        
        date_cols = self._get_date_columns(df)
        numeric_cols = self._get_numeric_columns(df)
        
        for date_col in date_cols:
            for num_col in numeric_cols:
                charts.append(ChartConfig(
                    chart_type=ChartType.TIME_SERIES,
                    title=f"{num_col.title()} Over Time",
                    x_axis=date_col,
                    y_axis=num_col,
                    confidence_score=0.9,  # High confidence for time series
                    description=f"Time series chart showing {num_col} trends over time"
                ))
                
        return charts
    
    def _get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that contain categorical data."""
        categorical = []
        for col in df.columns:
            if (df[col].dtype == 'object' or 
                len(df[col].unique()) <= len(df) * 0.7):
                categorical.append(col)
        return categorical
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that contain numeric data."""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def _get_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that contain date/time data."""
        date_cols = []
        for col in df.columns:
            # Check column name patterns
            if any(pattern in col.lower() for pattern in ['date', 'time', 'created', 'updated']):
                date_cols.append(col)
            # Could add more sophisticated date detection here
        return date_cols
    
    def _is_sequential_column(self, df: pd.DataFrame, col: str) -> bool:
        """Check if a column represents sequential data."""
        try:
            # Check for numeric sequence
            if df[col].dtype in [np.int64, np.float64]:
                sorted_values = sorted(df[col].unique())
                if len(sorted_values) > 2:
                    return True
            
            # Check for date patterns
            if any(pattern in col.lower() for pattern in ['date', 'time', 'month', 'year', 'day']):
                return True
                
            return False
        except:
            return False
    
    def get_detection_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a summary of chart detection analysis for debugging."""
        if not data:
            return {}
        
        df = self._prepare_dataframe(data)
        if df.empty:
            return {'error': 'No non-PII data available'}
        
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': self._get_numeric_columns(df),
            'categorical_columns': self._get_categorical_columns(df),
            'date_columns': self._get_date_columns(df),
            'column_types': {col: str(df[col].dtype) for col in df.columns},
            'unique_counts': {col: len(df[col].unique()) for col in df.columns}
        } 