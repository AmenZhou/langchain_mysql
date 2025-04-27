from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, OperationalError
from pydantic import BaseModel
from .schema_vectorizer import SchemaVectorizer
from .langchain_config import create_db_chain_with_schema, get_relevant_prompt
from .utils import sanitize_sql_response
import os
import asyncio
from .langchain_config import backoff_with_jitter
import traceback
import logging
from openai import RateLimitError, APIError
from backend.exceptions import DatabaseError
from backend.langchain_mysql import LangChainMySQL

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    prompt_type: str | None = None

# Global variables for components
engine = None
vectorizer = None
langchain_mysql = None

def get_db_engine():
    """Get or create the database engine."""
    global engine
    if engine is None:
        db_url = os.getenv("DATABASE_URL", "mysql+pymysql://user:password@localhost/db")
        engine = create_engine(db_url)
    return engine

def get_vectorizer():
    """Get or create the schema vectorizer."""
    global vectorizer
    if vectorizer is None:
        engine = get_db_engine()
        vectorizer = SchemaVectorizer(engine)
        vectorizer.preload_schema_to_vectordb()
    return vectorizer

async def get_langchain_mysql():
    """Get or create the LangChainMySQL instance."""
    global langchain_mysql
    if langchain_mysql is None:
        langchain_mysql = LangChainMySQL()
        await langchain_mysql.initialize()
    return langchain_mysql

async def execute_query(query: str, schema_info: str) -> str:
    """Execute a SQL query with schema information."""
    try:
        # Create database chain
        chain = create_db_chain_with_schema(schema_info)
        
        # Execute query
        result = await chain.ainvoke({"query": query})
        
        # Sanitize response
        return await sanitize_sql_response(result["result"])
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise

@app.post("/query")
async def query(request: QueryRequest):
    """Process a natural language query and return SQL results."""
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Query cannot be empty"
            )
        
        langchain_mysql = await get_langchain_mysql()
        result = await langchain_mysql.process_query(request.query)
        return result
        
    except ProgrammingError as e:
        error_msg = str(e).lower()
        if "no such table" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Table does not exist"
            )
        elif "syntax error" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid SQL syntax"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
            
    except OperationalError as e:
        if "permission denied" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
            
    except RateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OpenAI rate limit exceeded"
        )
        
    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenAI API error: {str(e)}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Check the health of the application."""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        ) 
