from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any
from ..langchain_mysql import LangChainMySQL, get_langchain_mysql
from ..security import limiter
from ..models import QueryResponse
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

@router.post("/query", responses={
    422: {"model": ErrorResponse, "description": "Validation error"},
    500: {"model": ErrorResponse, "description": "Internal server error"}
})
@limiter.limit("5/minute")
async def process_query(request: Request, langchain_mysql: LangChainMySQL = Depends(get_langchain_mysql)):
    # Log request details
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Request client host: {request.client.host}")
    
    # Manually parse and validate JSON body
    raw = await request.body()
    body_str = raw.decode()
    logger.info(f"Raw request body: {body_str}")
    
    try:
        body_json = json.loads(body_str)
        logger.info(f"Parsed JSON body: {json.dumps(body_json, indent=2)}")
    except json.JSONDecodeError as e:
        error_details = {"error": "Invalid JSON", "details": str(e)}
        logger.error(f"Invalid JSON in request body: {e}")
        raise HTTPException(status_code=422, detail=error_details)
    
    # Extract required fields
    query = body_json.get("query")
    prompt_type = body_json.get("prompt_type")
    
    if not query or not isinstance(query, str) or not query.strip():
        error_details = {"error": "Empty query", "details": "Query cannot be empty"}
        logger.error("Empty or invalid query received")
        raise HTTPException(status_code=422, detail=error_details)
    
    logger.info(f"Processing query: {query}")
    logger.info(f"Prompt type: {prompt_type}")
    
    # Call the LangChainMySQL to process
    try:
        result_sql = await langchain_mysql.process_query(query, prompt_type)
        logger.info(f"Query result: {result_sql}")
        return QueryResponse(result=result_sql)
    except HTTPException as e:
        # Preserve our formatted error details
        raise e
    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail={"error": "Unexpected error", "details": str(e)})
