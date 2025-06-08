"""
Charts Package - Intelligent Chart Generation for LangChain MySQL

This package provides automated chart generation capabilities that analyze
database query results and create appropriate visualizations while respecting
data privacy and PII filtering.

Core Components:
- ChartOrchestrator: Main interface for chart generation
- ChartEligibilityAnalyzer: Determines if data is suitable for charts
- ChartTypeDetector: Recommends appropriate chart types
- ChartGenerator: Creates interactive Plotly visualizations

Usage:
    from charts import ChartOrchestrator
    
    orchestrator = ChartOrchestrator()
    result = await orchestrator.process_data_for_charts(data)
"""

from .orchestrator import ChartOrchestrator
from .analyzer import ChartEligibilityAnalyzer
from .detector import ChartTypeDetector  
from .generator import ChartGenerator
from .models import ChartType, ChartConfig, ChartResult, ChartData, ChartResponse
from .exceptions import ChartGenerationError, DataIneligibleError

__all__ = [
    'ChartOrchestrator',
    'ChartEligibilityAnalyzer', 
    'ChartTypeDetector',
    'ChartGenerator',
    'ChartType',
    'ChartConfig', 
    'ChartResult',
    'ChartData',
    'ChartResponse',
    'ChartGenerationError',
    'DataIneligibleError'
]

__version__ = '1.0.0' 