from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import time
import asyncio
import logging
import traceback
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, OperationalError
from typing import Optional, Dict, Any, List
from fastapi import status
from openai import RateLimitError, APIError
from contextlib import asynccontextmanager
from sqlalchemy import text, create_engine
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from .langchain_config import create_db_chain_with_schema, backoff_with_jitter  # Import Langchain components
from .models import QueryRequest  # Import the Pydantic model
from .utils import refine_prompt_with_ai, sanitize_sql_response  # Import utility functions
from .schema_vectorizer import SchemaVectorizer  # Import schema vectorization
from .db_utils import get_database_url  # Import database functions

# ✅ Load Environment Variables
load_dotenv()

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MYSQL_URL = get_database_url()

# Initialize schema vectorizer
schema_vectorizer = SchemaVectorizer(db_url=MYSQL_URL)

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
        self.engine = create_engine(MYSQL_URL, pool_pre_ping=True)
        self.schema_vectorizer = SchemaVectorizer(db_url=MYSQL_URL)

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

    async def run_query_with_retry(self, query: str, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Run a query with retry logic and return structured data."""
        attempt = 0
        last_error = None
        while attempt < max_retries:
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(query))
                    column_names = result.keys()
                    rows = result.fetchall()
                    # Convert to list of dictionaries for JSON serialization
                    return [dict(zip(column_names, row)) for row in rows]
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt == max_retries:
                    logger.error(f"Failed to execute query after {max_retries} attempts: {e}")
                    raise Exception(f"Max retries exceeded: {str(last_error)}")
                delay = backoff_with_jitter(attempt)
                await asyncio.sleep(delay)

    async def generate_explanation(self, sql_query: str, data: List[Dict[str, Any]]) -> str:
        """Generate a natural language explanation of the SQL query and results."""
        try:
            # Use a simple prompt for explanation
            explanation_prompt = PromptTemplate(
                input_variables=["sql_query", "data"],
                template="""Given the SQL query:
                {sql_query}
                
                And the results:
                {data}
                
                Provide a clear, concise explanation of what this query is doing and what the results show. 
                Explain in natural language that a non-technical person would understand."""
            )
            
            # Use ChatOpenAI for explanation
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7  # Slightly more creative for natural language
            )
            
            # Create and run the chain
            chain = explanation_prompt | llm
            result = await chain.ainvoke({"sql_query": sql_query, "data": str(data)})
            
            if hasattr(result, 'content'):
                return result.content
            
            return str(result)
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Unable to generate explanation due to error: {str(e)}"

    async def process_query(self, query: str, prompt_type: str = None, response_type: str = "sql") -> Dict[str, Any]:
        """Process a natural language query and return results based on response_type.
        
        Args:
            query: The natural language query
            prompt_type: Type of prompt to use
            response_type: Type of response to return (sql, data, natural_language, all)
        
        Returns:
            Dictionary with results based on response_type
        """
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
                    detail={
                        "error": "Schema Error",
                        "details": "No relevant schema information found for the query"
                    }
                )
            
            # Create chain and run query
            logger.info("Creating database chain")
            chain = await create_db_chain_with_schema(schema_info)
            logger.info("Invoking chain with query")
            result = await chain.ainvoke({"query": query, "schema_info": schema_info})
            logger.info(f"Chain result type: {type(result)}")
            
            if not result:
                logger.error("Chain returned empty result")
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Chain Error",
                        "details": "Chain returned empty result"
                    }
                )
            
            if not hasattr(result, 'content'):
                logger.error(f"Result missing content attribute. Result type: {type(result)}")
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Result Error",
                        "details": "Failed to generate SQL query from the result - missing content"
                    }
                )
            
            if not result.content:
                logger.error("Result content is empty")
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Result Error",
                        "details": "Generated SQL query is empty"
                    }
                )
            
            # Extract the SQL query
            sql_query = result.content.strip()
            logger.info(f"Successfully generated SQL query: {sql_query}")
            
            # Initialize response with SQL query
            response = {
                "sql": sql_query,
                "response_type": response_type
            }
            
            # Based on response type, fetch additional information
            if response_type in ["data", "all"]:
                try:
                    logger.info(f"Executing SQL query: {sql_query}")
                    data = await self.run_query_with_retry(sql_query)
                    logger.info(f"Query execution successful. Rows returned: {len(data)}")
                    response["data"] = data
                except Exception as e:
                    logger.error(f"Error executing SQL query: {e}")
                    # Don't fail the entire request if data execution fails
                    response["data_error"] = str(e)
            
            if response_type in ["natural_language", "all"]:
                try:
                    # Only generate explanation if we have data or if we're just asking for natural language
                    if "data" in response or response_type == "natural_language":
                        data = response.get("data", [])
                        logger.info("Generating natural language explanation")
                        explanation = await self.generate_explanation(sql_query, data)
                        logger.info(f"Explanation generated successfully")
                        response["explanation"] = explanation
                except Exception as e:
                    logger.error(f"Error generating explanation: {e}")
                    # Don't fail the entire request if explanation fails
                    response["explanation_error"] = str(e)
            
            # Set the result based on response_type
            if response_type == "sql":
                response["result"] = sql_query
            elif response_type == "data":
                response["result"] = response.get("data", [])
            elif response_type == "natural_language":
                response["result"] = response.get("explanation", "No explanation available")
            else:  # "all"
                response["result"] = {
                    "sql": sql_query,
                    "data": response.get("data", []),
                    "explanation": response.get("explanation", "No explanation available")
                }
            
            return response
        
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
                status_code=422,
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

# Initialize LangChainMySQL instance
langchain_mysql = LangChainMySQL()

async def get_langchain_mysql() -> LangChainMySQL:
    """Dependency function to get LangChainMySQL instance."""
    return langchain_mysql

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    try:
        logger.info("Initializing LangChainMySQL instance...")
        await langchain_mysql.initialize()
        logger.info("LangChainMySQL initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing LangChainMySQL: {e}")
        logger.error(traceback.format_exc())
    yield

# Initialize FastAPI application
app = FastAPI(
    title="LangChain MySQL API",
    description="API for interacting with MySQL using LangChain",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add query endpoint
@app.post("/query")
async def process_query_endpoint(query_request: QueryRequest):
    """Process a natural language query."""
    try:
        langchain_mysql = LangChainMySQL()
        await langchain_mysql.initialize()
        result = await langchain_mysql.process_query(query_request.query)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
