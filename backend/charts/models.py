"""
Chart Generation Data Models

Internal data models for the chart generation system.
These are separate from API models to maintain clean separation of concerns.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field


class ChartType(str, Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    TIME_SERIES = "time_series"
    HEATMAP = "heatmap"
    BOX_PLOT = "box_plot"
    AREA = "area"


@dataclass
class ChartConfig:
    """Chart configuration and metadata."""
    chart_type: ChartType
    title: str
    x_axis: str
    y_axis: Optional[str] = None
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    confidence_score: float = 0.0
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'chart_type': self.chart_type.value,
            'title': self.title,
            'x_axis': self.x_axis,
            'y_axis': self.y_axis,
            'color_column': self.color_column,
            'size_column': self.size_column,
            'confidence_score': self.confidence_score,
            'description': self.description
        }


@dataclass
class ChartResult:
    """Result of chart generation."""
    chart_type: ChartType
    plotly_json: Dict[str, Any]
    config: ChartConfig
    data_summary: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'chart_type': self.chart_type.value,
            'plotly_json': self.plotly_json,
            'config': self.config.to_dict(),
            'data_summary': self.data_summary,
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass 
class EligibilityResult:
    """Result of data eligibility analysis."""
    eligible: bool
    reason: str
    data_points: int
    columns_analyzed: int
    pii_columns_excluded: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'eligible': self.eligible,
            'reason': self.reason,
            'data_points': self.data_points,
            'columns_analyzed': self.columns_analyzed,
            'pii_columns_excluded': self.pii_columns_excluded
        }


@dataclass
class ChartProcessingResult:
    """Complete result of chart processing pipeline."""
    eligibility: EligibilityResult
    charts: List[ChartResult]
    processing_time_ms: float
    
    @property
    def eligible(self) -> bool:
        """Convenience property for eligibility status."""
        return self.eligibility.eligible
    
    @property
    def successful_charts(self) -> List[ChartResult]:
        """Get only successfully generated charts."""
        return [chart for chart in self.charts if chart.success]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'eligible': self.eligible,
            'reason': self.eligibility.reason,
            'charts': [chart.to_dict() for chart in self.successful_charts],
            'recommendations': len(self.successful_charts),
            'processing_time_ms': self.processing_time_ms,
            'metadata': {
                'data_points': self.eligibility.data_points,
                'columns_analyzed': self.eligibility.columns_analyzed,
                'pii_columns_excluded': self.eligibility.pii_columns_excluded
            }
        }


class ChartSettings:
    """Configuration settings for chart generation."""
    
    def __init__(self):
        # Data eligibility thresholds
        self.min_rows = 2
        self.max_rows = 1000
        self.min_numeric_variance = 0.01
        
        # Chart type limits
        self.max_pie_categories = 8
        self.max_bar_categories = 20
        self.min_histogram_unique_values = 5
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        self.low_confidence_threshold = 0.4
        
        # Performance settings
        self.max_chart_recommendations = 3
        self.enable_caching = True
        self.chart_generation_timeout = 30  # seconds


# API Models for Chart Endpoints
# These are kept here to maintain isolation of chart functionality

class ChartData(BaseModel):
    """Chart data model for API responses."""
    chart_type: str = Field(description="Type of chart (bar, line, pie, etc.)")
    plotly_json: Dict[str, Any] = Field(description="Plotly chart JSON configuration")
    config: Dict[str, Any] = Field(description="Chart configuration metadata")
    data_summary: Dict[str, Any] = Field(description="Summary statistics about the data")


class ChartResponse(BaseModel):
    """Chart response model for API responses."""
    eligible: bool = Field(description="Whether data is eligible for chart generation")
    reason: str = Field(description="Reason for eligibility status")
    charts: List[ChartData] = Field(default=[], description="Generated charts")
    recommendations: int = Field(default=0, description="Number of chart recommendations") 