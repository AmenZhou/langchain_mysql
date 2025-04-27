from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any
from ..langchain_mysql import LangChainMySQL
from ..security import limiter
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

class QueryRequest(BaseModel):
    query: str
    prompt_type: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    details: Dict[str, Any]

@router.post("/query", responses={
    422: {"model": ErrorResponse, "description": "Validation error"},
    500: {"model": ErrorResponse, "description": "Internal server error"}
})
@limiter.limit("5/minute")
async def process_query(request: Request, query_request: QueryRequest, langchain_mysql: LangChainMySQL = Depends()):
    try:
        # Log the raw request body before validation
        body = await request.body()
        body_str = body.decode()
        logger.info(f"Raw request body: {body_str}")
        
        # Try to parse the body manually to see if it's valid JSON
        try:
            body_json = json.loads(body_str)
            logger.info(f"Parsed JSON body: {json.dumps(body_json, indent=2)}")
        except json.JSONDecodeError as e:
            error_details = {"error": "Invalid JSON", "details": str(e)}
            logger.error(f"Invalid JSON in request body: {str(e)}")
            raise HTTPException(status_code=422, detail=error_details)
        
        # Log the parsed request
        logger.info(f"Parsed query request: {json.dumps(query_request.dict(), indent=2)}")
        
        # Validate required fields
        if not query_request.query:
            error_details = {"error": "Empty query", "details": "Query cannot be empty"}
            logger.error("Empty query received")
            raise HTTPException(status_code=422, detail=error_details)
        
        logger.info(f"Processing query: {query_request.query}")
        logger.info(f"Prompt type: {query_request.prompt_type}")
        
        result = await langchain_mysql.process_query(query_request.query, query_request.prompt_type)
        logger.info(f"Query result: {result}")
        return result
    except ValidationError as e:
        error_details = {"error": "Validation error", "details": e.errors()}
        logger.error(f"Validation error: {str(e)}")
        logger.error(f"Validation error details: {e.errors()}")
        raise HTTPException(status_code=422, detail=error_details)
    except HTTPException as e:
        error_details = {"error": "HTTP error", "details": e.detail}
        logger.error(f"HTTP Exception: {str(e)}")
        logger.error(f"HTTP Exception details: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=error_details)
    except AttributeError as e:
        error_details = {
            "error": "Attribute error",
            "details": {
                "message": str(e),
                "type": "AttributeError",
                "traceback": traceback.format_exc(),
                "object_type": type(e.__traceback__.tb_frame.f_locals.get('self', None)).__name__ if e.__traceback__ else "Unknown",
                "missing_attribute": str(e).split("'")[1] if "'" in str(e) else str(e)
            }
        }
        logger.error(f"Attribute error occurred: {str(e)}")
        logger.error(f"Object type: {error_details['details']['object_type']}")
        logger.error(f"Missing attribute: {error_details['details']['missing_attribute']}")
        logger.error(f"Traceback:\n{error_details['details']['traceback']}")
        raise HTTPException(status_code=422, detail=error_details)
    except Exception as e:
        error_details = {
            "error": "Unexpected error",
            "details": {
                "message": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "args": e.args,
                "object_info": {
                    "dict": e.__dict__ if hasattr(e, '__dict__') else "No __dict__ attribute",
                    "dir": dir(e)
                }
            }
        }
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback:\n{error_details['details']['traceback']}")
        logger.error(f"Object info: {error_details['details']['object_info']}")
        raise HTTPException(status_code=500, detail=error_details) 
