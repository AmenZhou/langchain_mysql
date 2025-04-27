from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import time
import asyncio
import logging
import traceback
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, OperationalError
from typing import Optional, Dict, Any
from fastapi import status
from openai import RateLimitError, APIError
from contextlib import asynccontextmanager
from sqlalchemy import text

from .langchain_config import create_db_chain_with_schema, backoff_with_jitter  # Import Langchain components
from .models import QueryRequest  # Import the Pydantic model
from .utils import refine_prompt_with_ai, sanitize_sql_response  # Import utility functions
from .schema_vectorizer import SchemaVectorizer  # Import schema vectorization
from .db_utils import get_db_engine, DATABASE_URL  # Import database functions

# ✅ Load Environment Variables
load_dotenv()

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize schema vectorizer
schema_vectorizer = SchemaVectorizer(db_url=os.getenv("DATABASE_URL", "mysql+pymysql://user:password@localhost:3306/db"))

# ✅ Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        logger.info("Preloading schema to vector database on startup...")
        await asyncio.to_thread(lambda: schema_vectorizer.preload_schema_to_vectordb())
        logger.info("Schema preloaded successfully!")
    except Exception as e:
        logger.error(f"Error preloading schema on startup: {e}")
        logger.error(traceback.format_exc())
    yield
    # Shutdown
    # Add any cleanup code here if needed

class LangChainMySQL:
    def __init__(self):
        self.engine = get_db_engine()
        self.schema_vectorizer = SchemaVectorizer(db_url=DATABASE_URL)

    async def initialize(self):
        """Initialize the LangChain MySQL instance."""
        try:
            # Extract schema information
            schema_info = await self.schema_vectorizer.extract_table_schema()
            if not schema_info:
                raise ValueError("Failed to extract schema information")

            # Initialize vector store with schema
            await self.schema_vectorizer.initialize_vector_store(schema_info)
            logger.info("Successfully initialized LangChain MySQL")
        except Exception as e:
            logger.error(f"Error initializing LangChain MySQL: {e}")
            raise

    async def run_query_with_retry(self, query: str, max_retries: int = 3) -> str:
        """Run a query with retry logic."""
        attempt = 0
        last_error = None
        while attempt < max_retries:
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(query))
                    return str(result.fetchall())
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt == max_retries:
                    logger.error(f"Failed to execute query after {max_retries} attempts: {e}")
                    raise Exception(f"Max retries exceeded: {str(last_error)}")
                delay = backoff_with_jitter(attempt)
                await asyncio.sleep(delay)

    async def process_query(self, query: str, prompt_type: str = None) -> str:
        """Process a natural language query and return SQL results."""
        try:
            logger.info(f"Starting query processing for: {query}")
            if not query:
                logger.error("Empty query received")
                raise HTTPException(
                    status_code=422,
                    detail="Query cannot be empty"
                )

            # Get schema info from vectorizer
            logger.info("Getting relevant schema information")
            schema_info = await self.schema_vectorizer.get_relevant_schema(query)
            logger.info(f"Schema info: {schema_info}")
            
            if not schema_info:
                logger.error("No relevant schema information found")
                raise HTTPException(
                    status_code=422,
                    detail="No relevant schema information found for the query"
                )
            
            # Create chain and run query
            logger.info("Creating database chain")
            chain = await create_db_chain_with_schema(schema_info)
            logger.info("Invoking chain with query")
            result = await chain.ainvoke({"query": query, "schema_info": schema_info})
            logger.info(f"Chain result: {result}")
            
            if not result or not result.get("result"):
                logger.error("Failed to generate SQL query from result")
                raise HTTPException(
                    status_code=422,
                    detail="Failed to generate SQL query from the result"
                )
            
            return result["result"]
        
        except HTTPException as e:
            logger.error(f"HTTP Exception in process_query: {str(e)}")
            raise e
        except RateLimitError as e:
            logger.error(f"Rate limit error: {str(e)}")
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {str(e)}"
            )
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API error: {str(e)}"
            )
        except ProgrammingError as e:
            error_msg = str(e).lower()
            logger.error(f"SQL Programming Error: {error_msg}")
            if "relation" in error_msg and "does not exist" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail=f"Table does not exist: {str(e)}"
                )
            elif "permission" in error_msg or "access denied" in error_msg:
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {str(e)}"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid SQL syntax: {str(e)}"
                )
        except OperationalError as e:
            error_msg = str(e).lower()
            logger.error(f"SQL Operational Error: {error_msg}")
            if "permission" in error_msg or "access denied" in error_msg:
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {str(e)}"
                )
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
        except ValueError as e:
            logger.error(f"Value Error: {str(e)}")
            if len(str(query)) > 5000:
                raise HTTPException(
                    status_code=422,
                    detail="Query too long"
                )
            raise HTTPException(
                status_code=422,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error in process_query: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}"
            )

# Create LangChain MySQL instance
langchain_mysql = LangChainMySQL()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    await langchain_mysql.initialize()
    yield
