"""
Chart Generation Exceptions

Custom exceptions for the chart generation system.
"""


class ChartGenerationError(Exception):
    """Base exception for chart generation errors."""
    
    def __init__(self, message: str, chart_type: str = None, data_size: int = None):
        super().__init__(message)
        self.chart_type = chart_type
        self.data_size = data_size


class DataIneligibleError(ChartGenerationError):
    """Raised when data is not eligible for chart generation."""
    
    def __init__(self, reason: str, data_size: int = None):
        super().__init__(f"Data not eligible for charts: {reason}", data_size=data_size)
        self.reason = reason


class ChartConfigurationError(ChartGenerationError):
    """Raised when chart configuration is invalid."""
    
    def __init__(self, message: str, chart_type: str = None):
        super().__init__(f"Chart configuration error: {message}", chart_type=chart_type)


class ChartRenderingError(ChartGenerationError):
    """Raised when chart rendering fails."""
    
    def __init__(self, message: str, chart_type: str = None):
        super().__init__(f"Chart rendering failed: {message}", chart_type=chart_type)


class InsufficientDataError(DataIneligibleError):
    """Raised when there's insufficient data for chart generation."""
    
    def __init__(self, data_size: int, minimum_required: int = 2):
        super().__init__(
            f"Insufficient data points: {data_size} (minimum {minimum_required} required)",
            data_size=data_size
        )
        self.minimum_required = minimum_required


class PIIOnlyDataError(DataIneligibleError):
    """Raised when all data fields are marked as PII/private."""
    
    def __init__(self):
        super().__init__("All data fields are private/filtered - no data available for visualization") 