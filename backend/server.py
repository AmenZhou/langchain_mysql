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
from openai import OpenAIError, RateLimitError, APIError
from .exceptions import DatabaseError
from .langchain_mysql import LangChainMySQL
from fastapi.middleware.cors import CORSMiddleware
from .database import get_db_engine, get_langchain_mysql
from .models import QueryRequest, QueryResponse
from .db_utils import get_database_url

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str
    prompt_type: str | None = None

# Global variables for components
engine = None
langchain_mysql = None

def get_db_engine():
    """Get or create the database engine."""
    global engine
    if engine is None:
        db_url = os.getenv("DATABASE_URL", get_database_url())
        engine = create_engine(db_url)
    return engine

async def get_langchain_mysql():
    """Get or create the LangChainMySQL instance."""
    global langchain_mysql
    if langchain_mysql is None:
        langchain_mysql = LangChainMySQL()
        try:
            await langchain_mysql.initialize()
        except Exception as e:
            logger.error(f"Error initializing LangChain MySQL: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize LangChain MySQL: {str(e)}"
            )
    return langchain_mysql

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LangChain MySQL API",
        description="API for interacting with MySQL using LangChain",
        version="1.0.0"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app

def register_routers(app: FastAPI) -> None:
    """Register all routers with the application."""
    
    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        try:
            engine = get_db_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Health check failed: {str(e)}"
            )

    @app.post("/query", response_model=QueryResponse)
    async def process_query(request: QueryRequest):
        """Process a natural language query."""
        try:
            langchain_mysql = await get_langchain_mysql()
            result = await langchain_mysql.process_query(request.query)
            return QueryResponse(result=result)
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

# Create the FastAPI application instance
app = create_app()
register_routers(app) 
