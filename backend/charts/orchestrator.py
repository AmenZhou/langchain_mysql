"""
Chart Orchestrator

Main coordinator for the chart generation pipeline.
Manages the flow from data analysis to chart creation.
"""

from typing import Dict, List, Any, Optional
import time
import logging
import asyncio

from .models import ChartProcessingResult, ChartSettings
from .analyzer import ChartEligibilityAnalyzer
from .detector import ChartTypeDetector
from .generator import ChartGenerator
from .exceptions import DataIneligibleError, ChartGenerationError

logger = logging.getLogger(__name__)


class ChartOrchestrator:
    """Main class that orchestrates the chart generation process."""
    
    def __init__(self, settings: ChartSettings = None):
        self.settings = settings or ChartSettings()
        self.analyzer = ChartEligibilityAnalyzer(self.settings)
        self.detector = ChartTypeDetector(self.settings)
        self.generator = ChartGenerator(self.settings)
    
    async def process_data_for_charts(self, data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Process data and generate charts if eligible.
        
        Args:
            data: List of data dictionaries from database query
            
        Returns:
            Dictionary with chart processing results or None if processing fails
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze data eligibility
            try:
                eligibility_result = self.analyzer.analyze(data)
                logger.info(f"Data eligibility check passed: {eligibility_result.reason}")
            except DataIneligibleError as e:
                logger.info(f"Data not eligible for charts: {e.reason}")
                return {
                    "eligible": False,
                    "reason": e.reason,
                    "charts": [],
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "metadata": {
                        "data_points": getattr(e, 'data_size', len(data) if data else 0),
                        "error_type": type(e).__name__
                    }
                }
            
            # Step 2: Detect suitable chart types
            chart_configs = self.detector.detect_best_charts(data)
            if not chart_configs:
                return {
                    "eligible": False,
                    "reason": "No suitable chart types detected for this data",
                    "charts": [],
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "metadata": {
                        "data_points": eligibility_result.data_points,
                        "columns_analyzed": eligibility_result.columns_analyzed
                    }
                }
            
            logger.info(f"Detected {len(chart_configs)} potential chart types")
            
            # Step 3: Generate charts
            chart_results = []
            for config in chart_configs:
                try:
                    chart_result = self.generator.generate_chart(data, config)
                    if chart_result.success:
                        chart_results.append(chart_result.to_dict())
                        logger.debug(f"Successfully generated {config.chart_type} chart")
                    else:
                        logger.warning(f"Failed to generate {config.chart_type} chart: {chart_result.error_message}")
                except Exception as e:
                    logger.error(f"Error generating {config.chart_type} chart: {str(e)}")
                    continue
            
            processing_time = (time.time() - start_time) * 1000
            
            if chart_results:
                return {
                    "eligible": True,
                    "reason": f"Successfully generated {len(chart_results)} chart(s)",
                    "charts": chart_results,
                    "recommendations": len(chart_results),
                    "processing_time_ms": processing_time,
                    "metadata": {
                        "data_points": eligibility_result.data_points,
                        "columns_analyzed": eligibility_result.columns_analyzed,
                        "pii_columns_excluded": eligibility_result.pii_columns_excluded,
                        "chart_types_attempted": [config.chart_type.value for config in chart_configs],
                        "successful_charts": len(chart_results)
                    }
                }
            else:
                return {
                    "eligible": False,
                    "reason": "Chart generation failed for all detected chart types",
                    "charts": [],
                    "processing_time_ms": processing_time,
                    "metadata": {
                        "data_points": eligibility_result.data_points,
                        "chart_types_attempted": [config.chart_type.value for config in chart_configs],
                        "successful_charts": 0
                    }
                }
            
        except Exception as e:
            logger.error(f"Unexpected error in chart processing: {str(e)}")
            return {
                "eligible": False,
                "reason": f"Chart processing error: {str(e)}",
                "charts": [],
                "processing_time_ms": (time.time() - start_time) * 1000,
                "metadata": {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            }
    
    async def analyze_data_only(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze data for chart eligibility without generating charts.
        
        Args:
            data: List of data dictionaries
            
        Returns:
            Dictionary with analysis results
        """
        try:
            eligibility_result = self.analyzer.analyze(data)
            chart_configs = self.detector.detect_best_charts(data)
            
            return {
                "eligible": eligibility_result.eligible,
                "reason": eligibility_result.reason,
                "data_analysis": {
                    "data_points": eligibility_result.data_points,
                    "columns_analyzed": eligibility_result.columns_analyzed,
                    "pii_columns_excluded": eligibility_result.pii_columns_excluded
                },
                "chart_recommendations": [
                    {
                        "chart_type": config.chart_type.value,
                        "title": config.title,
                        "confidence_score": config.confidence_score,
                        "description": config.description
                    }
                    for config in chart_configs
                ],
                "column_info": self.analyzer.get_column_info(data),
                "detection_summary": self.detector.get_detection_summary(data)
            }
            
        except DataIneligibleError as e:
            return {
                "eligible": False,
                "reason": e.reason,
                "data_analysis": {
                    "data_points": getattr(e, 'data_size', len(data) if data else 0),
                    "error_type": type(e).__name__
                },
                "chart_recommendations": [],
                "column_info": self.analyzer.get_column_info(data) if data else {}
            }
        except Exception as e:
            return {
                "eligible": False,
                "reason": f"Analysis error: {str(e)}",
                "error": str(e)
            }
    
    def get_chart_capabilities(self) -> Dict[str, Any]:
        """
        Get information about chart generation capabilities.
        
        Returns:
            Dictionary with capability information
        """
        return {
            "supported_chart_types": self.generator.get_supported_chart_types(),
            "settings": {
                "min_data_points": self.settings.min_rows,
                "max_data_points": self.settings.max_rows,
                "max_recommendations": self.settings.max_chart_recommendations,
                "confidence_thresholds": {
                    "high": self.settings.high_confidence_threshold,
                    "medium": self.settings.medium_confidence_threshold,
                    "low": self.settings.low_confidence_threshold
                }
            },
            "features": [
                "Automatic PII filtering",
                "Intelligent chart type detection",
                "Interactive Plotly visualizations",
                "Confidence scoring",
                "Data quality analysis",
                "Multiple chart recommendations"
            ]
        }
    
    async def validate_chart_request(self, data: List[Dict[str, Any]], chart_type: str = None) -> Dict[str, Any]:
        """
        Validate a chart generation request.
        
        Args:
            data: List of data dictionaries
            chart_type: Optional specific chart type to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": False,
            "issues": [],
            "recommendations": []
        }
        
        # Basic data validation
        if not data:
            validation_result["issues"].append("No data provided")
            return validation_result
        
        if len(data) < self.settings.min_rows:
            validation_result["issues"].append(f"Insufficient data points ({len(data)} < {self.settings.min_rows})")
        
        if len(data) > self.settings.max_rows:
            validation_result["issues"].append(f"Too many data points ({len(data)} > {self.settings.max_rows})")
            validation_result["recommendations"].append("Consider limiting your query results")
        
        # Check for PII-only data
        try:
            eligibility_result = self.analyzer.analyze(data)
            if not eligibility_result.eligible:
                validation_result["issues"].append(eligibility_result.reason)
        except DataIneligibleError as e:
            validation_result["issues"].append(e.reason)
            
        # Validate specific chart type if requested
        if chart_type:
            from .models import ChartType
            try:
                chart_enum = ChartType(chart_type.lower())
                chart_validation = self.generator.validate_data_for_chart_type(data, chart_enum)
                if not chart_validation["valid"]:
                    validation_result["issues"].append(chart_validation["reason"])
            except ValueError:
                validation_result["issues"].append(f"Unsupported chart type: {chart_type}")
                validation_result["recommendations"].append(f"Supported types: {', '.join(self.generator.get_supported_chart_types())}")
        
        # Set overall validity
        validation_result["valid"] = len(validation_result["issues"]) == 0
        
        # Add helpful recommendations
        if not validation_result["valid"]:
            validation_result["recommendations"].extend([
                "Ensure your query returns sufficient non-PII data",
                "Try queries with aggregated results (GROUP BY)",
                "Include both categorical and numeric columns"
            ])
        
        return validation_result
    
    def configure_settings(self, **kwargs) -> None:
        """
        Update chart generation settings.
        
        Args:
            **kwargs: Settings to update
        """
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
                logger.info(f"Updated setting {key} to {value}")
            else:
                logger.warning(f"Unknown setting: {key}")
        
        # Recreate components with new settings
        self.analyzer = ChartEligibilityAnalyzer(self.settings)
        self.detector = ChartTypeDetector(self.settings)
        self.generator = ChartGenerator(self.settings) 