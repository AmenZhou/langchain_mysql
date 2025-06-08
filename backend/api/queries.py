"""
Query API Endpoints

Main query processing endpoints with optional chart integration.
Handles natural language to SQL conversion and execution.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, status
from sqlalchemy.exc import ProgrammingError, OperationalError
from openai import RateLimitError, APIError
from pydantic import BaseModel, Field
from enum import Enum
import logging

from langchain_mysql import LangChainMySQL
from charts import ChartOrchestrator, ChartResponse, ChartData

logger = logging.getLogger(__name__)

# Create router for query endpoints
router = APIRouter(prefix="", tags=["queries"])

# Global instances
_langchain_mysql: Optional[LangChainMySQL] = None
_chart_orchestrator: Optional[ChartOrchestrator] = None


# Models defined right where they're used - no separate models.py needed!

class ResponseType(str, Enum):
    """Enum for response types."""
    SQL = "sql"
    DATA = "data"
    NATURAL_LANGUAGE = "natural_language"
    ALL = "all"


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    session_id: Optional[str] = Field(default=None, description="Optional session ID for maintaining conversation history")
    prompt_type: Optional[str] = None
    response_type: ResponseType = Field(
        default=ResponseType.ALL,
        description="Type of response to return: sql (just the SQL query), data (execute SQL and return data), natural_language (return natural language explanation), or all (return everything)"
    )
    enable_charts: bool = Field(
        default=True,
        description="Whether to generate charts if data is suitable for visualization"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    result: Any = Field(description="Primary result based on requested response_type")
    sql: Optional[str] = Field(default=None, description="Generated SQL query")
    data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Data returned from executing the SQL query")
    explanation: Optional[str] = Field(default=None, description="Natural language explanation of the result")
    response_type: ResponseType = Field(default=ResponseType.ALL, description="Type of response that was returned")
    charts: Optional[Dict[str, Any]] = Field(default=None, description="Chart generation results if enabled and data is suitable")


# Helper functions - no global dependency injection needed

async def get_langchain_mysql() -> LangChainMySQL:
    """Get or create LangChain MySQL instance."""
    global _langchain_mysql
    if _langchain_mysql is None:
        _langchain_mysql = LangChainMySQL()
        try:
            await _langchain_mysql.initialize()
        except Exception as e:
            logger.error(f"Error initializing LangChain MySQL: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize LangChain MySQL: {str(e)}"
            )
    return _langchain_mysql


def get_chart_orchestrator() -> ChartOrchestrator:
    """Get or create chart orchestrator instance."""
    global _chart_orchestrator
    if _chart_orchestrator is None:
        _chart_orchestrator = ChartOrchestrator()
    return _chart_orchestrator


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a natural language query with optional chart generation.
    
    This is the main endpoint that converts natural language to SQL,
    executes the query, and optionally generates charts from the results.
    """
    try:
        # Process the query through LangChain MySQL
        langchain_mysql = await get_langchain_mysql()
        result = await langchain_mysql.process_query(
            query=request.query,
            session_id=request.session_id,
            prompt_type=request.prompt_type,
            response_type=request.response_type.value  # Convert enum to string
        )
        
        # Initialize response
        response = QueryResponse(result=result)
        
        # Generate charts if enabled and data is available
        if request.enable_charts and isinstance(result, dict) and result.get('data'):
            try:
                chart_orchestrator = get_chart_orchestrator()
                chart_data = await chart_orchestrator.process_data_for_charts(result['data'])
                
                if chart_data and chart_data.get('eligible'):
                    # Convert chart data to response format
                    chart_objects = []
                    for chart in chart_data.get('charts', []):
                        chart_objects.append(ChartData(
                            chart_type=chart['chart_type'],
                            plotly_json=chart['plotly_json'],
                            config=chart['config'],
                            data_summary=chart['data_summary']
                        ))
                    
                    chart_response = ChartResponse(
                        eligible=True,
                        reason=chart_data['reason'],
                        charts=chart_objects,
                        recommendations=len(chart_objects)
                    )
                    response.charts = chart_response.dict()
                else:
                    chart_response = ChartResponse(
                        eligible=False,
                        reason=chart_data.get('reason', 'No charts generated') if chart_data else 'Chart processing failed',
                        charts=[],
                        recommendations=0
                    )
                    response.charts = chart_response.dict()
            except Exception as chart_error:
                logger.warning(f"Chart generation failed: {str(chart_error)}")
                chart_response = ChartResponse(
                    eligible=False,
                    reason=f"Chart generation error: {str(chart_error)}",
                    charts=[],
                    recommendations=0
                )
                response.charts = chart_response.dict()
        
        return response
        
    except ProgrammingError as e:
        logger.error(f"SQL Programming Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid SQL query: {str(e)}"
        )
    except OperationalError as e:
        logger.error(f"SQL Operational Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database operation failed: {str(e)}"
        )
    except RateLimitError as e:
        logger.error(f"OpenAI Rate Limit Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OpenAI API rate limit exceeded. Please try again later."
        )
    except APIError as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI API error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        ) 