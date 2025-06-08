"""
Chart API Endpoints

Isolated chart generation endpoints separate from main query processing.
Provides dedicated chart functionality with proper error handling.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
import logging

from charts import ChartOrchestrator, ChartResponse, ChartData

logger = logging.getLogger(__name__)

# Create router for chart endpoints
router = APIRouter(prefix="/charts", tags=["charts"])

# Global chart orchestrator instance
_chart_orchestrator: Optional[ChartOrchestrator] = None


def get_chart_orchestrator() -> ChartOrchestrator:
    """Get or create chart orchestrator instance."""
    global _chart_orchestrator
    if _chart_orchestrator is None:
        _chart_orchestrator = ChartOrchestrator()
    return _chart_orchestrator


# Request/Response models specific to chart endpoints
class ChartGenerationRequest(BaseModel):
    """Request model for direct chart generation."""
    data: List[Dict[str, Any]] = Field(description="Data to generate charts from")
    chart_type: Optional[str] = Field(default=None, description="Specific chart type to generate")


class ChartAnalysisRequest(BaseModel):
    """Request model for chart analysis without generation."""
    data: List[Dict[str, Any]] = Field(description="Data to analyze for chart potential")


class ChartValidationRequest(BaseModel):
    """Request model for chart validation."""
    data: List[Dict[str, Any]] = Field(description="Data to validate")
    chart_type: Optional[str] = Field(default=None, description="Specific chart type to validate")


class ChartCapabilitiesResponse(BaseModel):
    """Response model for chart capabilities."""
    supported_chart_types: List[str]
    settings: Dict[str, Any]
    features: List[str]


class ChartAnalysisResponse(BaseModel):
    """Response model for chart analysis."""
    eligible: bool
    reason: str
    data_analysis: Dict[str, Any]
    chart_recommendations: List[Dict[str, Any]]
    column_info: Dict[str, Any]
    detection_summary: Optional[Dict[str, Any]] = None


class ChartValidationResponse(BaseModel):
    """Response model for chart validation."""
    valid: bool
    issues: List[str]
    recommendations: List[str]


@router.post("/generate", response_model=ChartResponse)
async def generate_charts(request: ChartGenerationRequest):
    """
    Generate charts from provided data.
    
    This endpoint accepts raw data and generates appropriate charts
    based on automatic detection or specified chart type.
    """
    try:
        orchestrator = get_chart_orchestrator()
        
        # Validate the request if specific chart type requested
        if request.chart_type:
            validation = await orchestrator.validate_chart_request(
                request.data, 
                request.chart_type
            )
            if not validation["valid"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "message": "Chart validation failed",
                        "issues": validation["issues"],
                        "recommendations": validation["recommendations"]
                    }
                )
        
        # Generate charts
        result = await orchestrator.process_data_for_charts(request.data)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Chart processing failed unexpectedly"
            )
        
        # Convert to response format
        if result.get('eligible') and result.get('charts'):
            chart_objects = []
            for chart in result['charts']:
                chart_objects.append(ChartData(
                    chart_type=chart['chart_type'],
                    plotly_json=chart['plotly_json'],
                    config=chart['config'],
                    data_summary=chart['data_summary']
                ))
            
            return ChartResponse(
                eligible=True,
                reason=result['reason'],
                charts=chart_objects,
                recommendations=len(chart_objects)
            )
        else:
            return ChartResponse(
                eligible=False,
                reason=result.get('reason', 'Chart generation failed'),
                charts=[],
                recommendations=0
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chart generation failed: {str(e)}"
        )


@router.post("/analyze", response_model=ChartAnalysisResponse)
async def analyze_data_for_charts(request: ChartAnalysisRequest):
    """
    Analyze data for chart potential without generating charts.
    
    This endpoint provides detailed analysis of data suitability,
    recommendations, and column information for debugging.
    """
    try:
        orchestrator = get_chart_orchestrator()
        result = await orchestrator.analyze_data_only(request.data)
        
        return ChartAnalysisResponse(
            eligible=result['eligible'],
            reason=result['reason'],
            data_analysis=result.get('data_analysis', {}),
            chart_recommendations=result.get('chart_recommendations', []),
            column_info=result.get('column_info', {}),
            detection_summary=result.get('detection_summary')
        )
        
    except Exception as e:
        logger.error(f"Chart analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chart analysis failed: {str(e)}"
        )


@router.post("/validate", response_model=ChartValidationResponse)
async def validate_chart_request(request: ChartValidationRequest):
    """
    Validate data and chart type for chart generation.
    
    This endpoint validates whether the provided data can be used
    to generate charts, optionally for a specific chart type.
    """
    try:
        orchestrator = get_chart_orchestrator()
        result = await orchestrator.validate_chart_request(
            request.data, 
            request.chart_type
        )
        
        return ChartValidationResponse(
            valid=result['valid'],
            issues=result['issues'],
            recommendations=result['recommendations']
        )
        
    except Exception as e:
        logger.error(f"Chart validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chart validation failed: {str(e)}"
        )


@router.get("/capabilities", response_model=ChartCapabilitiesResponse)
async def get_chart_capabilities():
    """
    Get information about chart generation capabilities.
    
    Returns supported chart types, settings, and features.
    """
    try:
        orchestrator = get_chart_orchestrator()
        capabilities = orchestrator.get_chart_capabilities()
        
        return ChartCapabilitiesResponse(
            supported_chart_types=capabilities['supported_chart_types'],
            settings=capabilities['settings'],
            features=capabilities['features']
        )
        
    except Exception as e:
        logger.error(f"Error getting chart capabilities: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chart capabilities: {str(e)}"
        )


@router.get("/health")
async def chart_service_health():
    """
    Health check for chart generation service.
    
    Verifies that all chart components are working properly.
    """
    try:
        orchestrator = get_chart_orchestrator()
        
        # Test with minimal sample data
        test_data = [
            {"category": "A", "value": 10},
            {"category": "B", "value": 20}
        ]
        
        analysis = await orchestrator.analyze_data_only(test_data)
        
        return {
            "status": "healthy",
            "components": {
                "analyzer": "operational",
                "detector": "operational", 
                "generator": "operational",
                "orchestrator": "operational"
            },
            "test_analysis": {
                "eligible": analysis['eligible'],
                "chart_types_detected": len(analysis.get('chart_recommendations', []))
            }
        }
        
    except Exception as e:
        logger.error(f"Chart service health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "components": {
                "analyzer": "unknown",
                "detector": "unknown",
                "generator": "unknown", 
                "orchestrator": "error"
            }
        } 