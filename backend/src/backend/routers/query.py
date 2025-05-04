from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any
from ..langchain_mysql import LangChainMySQL, get_langchain_mysql
from ..security import limiter
from ..models import QueryResponse, ResponseType, QueryRequest
import logging
import json
import sys
import traceback

# Configure logging to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create stdout handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(stdout_handler)

router = APIRouter()

class ErrorResponse(BaseModel):
    error: str
    details: Dict[str, Any]

@router.post("/query", response_model=QueryResponse, responses={
    422: {"model": ErrorResponse, "description": "Validation error"},
    500: {"model": ErrorResponse, "description": "Internal server error"}
})
@limiter.limit("5/minute")
async def process_query(request: Request, query_request: QueryRequest, langchain_mysql: LangChainMySQL = Depends(get_langchain_mysql)):
    # Log request details
    logger.info(f"Processing query: {query_request.query}")
    logger.info(f"Prompt type: {query_request.prompt_type}")
    logger.info(f"Response type: {query_request.response_type}")
    
    # Call the LangChainMySQL to process
    try:
        response_data = await langchain_mysql.process_query(query_request.query, query_request.prompt_type, query_request.response_type)
        logger.info(f"Query response generated successfully")
        
        # Create the response object
        response = QueryResponse(
            result=response_data.get("result"),
            sql=response_data.get("sql"),
            data=response_data.get("data"),
            explanation=response_data.get("explanation"),
            response_type=query_request.response_type
        )
        
        return response
    except HTTPException as e:
        # Preserve our formatted error details
        raise e
    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail={"error": "Unexpected error", "details": str(e)})
