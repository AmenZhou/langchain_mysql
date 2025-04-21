from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import time
import asyncio
from openai import OpenAIError
from sqlalchemy import text
from pymysql.err import ProgrammingError
import logging
import traceback
import sqlalchemy.exc
from contextlib import asynccontextmanager

from database import engine, db  # Import database objects
from langchain_config import chat_model, memory, db_chain, create_db_chain_with_schema, backoff_with_jitter  # Import Langchain components
from models import QueryRequest  # Import the Pydantic model
from utils import refine_prompt_with_ai, sanitize_sql_response  # Import utility functions
from schema_vectorizer import preload_schema_to_vectordb  # Import schema vectorization

# ✅ Load Environment Variables
load_dotenv()

# ✅ Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        logger.info("Preloading schema to vector database on startup...")
        await asyncio.to_thread(preload_schema_to_vectordb)
        logger.info("Schema preloaded successfully!")
    except Exception as e:
        logger.error(f"Error preloading schema on startup: {e}")
        logger.error(traceback.format_exc())
    yield
    # Shutdown
    # Add any cleanup code here if needed

# ✅ FastAPI App
app = FastAPI(lifespan=lifespan)

# ✅ Configure Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ✅ CORS Middleware to Allow OPTIONS Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Asynchronous Query Execution to Improve Speed
async def run_query_with_retry(user_query, retries=5, use_schema=True):
    refined_query = await refine_prompt_with_ai(user_query)

    if not refined_query:
        logger.error("Failed to refine the prompt. Using the original query.")
        refined_query = user_query

    for attempt in range(retries):
        try:
            logger.info(f"Attempt {attempt + 1} to run refined query: {refined_query}")
            
            # Use schema-enhanced chain if requested
            if use_schema:
                # Get a custom chain with schema information for this query
                custom_chain, prompt_with_schema = create_db_chain_with_schema(refined_query)
                logger.info(f"Using schema-enhanced prompt: {prompt_with_schema[:100]}...")
                response_dict = await asyncio.to_thread(lambda: custom_chain.invoke({"query": refined_query}))
            else:
                # Use the original chain without schema enhancement
                response_dict = await asyncio.to_thread(lambda: db_chain.invoke({"query": refined_query}))
            
            logger.info(f"Raw response from db_chain: {response_dict}")
            response = response_dict.get("result", "")
            response = await sanitize_sql_response(response, chat_model)
            memory.save_context({"query": user_query}, {"result": response})
            return response
        except OpenAIError as e:
            if "rate_limit_exceeded" in str(e) or "429" in str(e):
                # Use our custom backoff strategy
                wait_time = await asyncio.to_thread(backoff_with_jitter, attempt)
                logger.warning(f"Rate limit exceeded, retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{retries})")
                await asyncio.sleep(wait_time)
                # If this was our last retry, raise the exception
                if attempt == retries - 1:
                    logger.error(f"Failed after {retries} retries due to rate limits")
                    raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded. Please try again later.")
            else:
                logger.error(f"OpenAI Error during query: {e}")
                logger.error(traceback.format_exc())
                raise
        except (ProgrammingError, sqlalchemy.exc.ProgrammingError) as e:
            logger.error(f"Database Error during query: {e}")
            logger.error(traceback.format_exc())
            error_msg = str(e)
            if "Table" in error_msg and "doesn't exist" in error_msg:
                table_name = error_msg.split("'")[1]
                raise HTTPException(
                    status_code=400,
                    detail=f"Database Error: The table '{table_name}' does not exist in the database."
                )
            raise HTTPException(
                status_code=400,
                detail=f"Database Error: {error_msg}"
            )
        except Exception as e:
            logger.error(f"General error during query: {e}")
            logger.error(traceback.format_exc())
            if isinstance(e, ProgrammingError) and "You have an error in your SQL syntax" in str(e) and "near 'LIMIT 5' at line 4" in str(e):
                try:
                    intermediate_steps = response_dict.get("intermediate_steps")
                    if intermediate_steps and len(intermediate_steps) > 1:
                        sql_command = intermediate_steps[1].sql_
                        if sql_command.lower().startswith("select count(*)"):
                            modified_sql = sql_command.replace("LIMIT 5;", "").replace("LIMIT 5", "")
                            logger.info(f"Retrying with modified SQL: {modified_sql}")
                            with engine.connect() as connection:
                                result = connection.execute(text(modified_sql)).scalar_one()
                            return str(result)
                except Exception as ex:
                    logger.error(f"Error during workaround: {ex}")
                    logger.error(traceback.format_exc())
            raise
    logger.error("Failed after retries.")
    return None

# ✅ FastAPI Endpoint to Handle User Queries
@app.post("/query")
async def query_database(request: QueryRequest):
    try:
        logger.info(f"Received user query: {request.question}")
        response = await run_query_with_retry(request.question, use_schema=True)
        if not response:
            logger.error("No result returned from LangChain.")
            raise HTTPException(status_code=500, detail="No result returned from LangChain.")
        logger.info(f"Final response: {response}")
        return {"result": response}
    except HTTPException as e:
        # Re-raise HTTPExceptions as they are already properly formatted
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred in /query endpoint: {e}")
        logger.error(traceback.format_exc())
        if "rate_limit" in str(e).lower():
            raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded. Please try again later.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Check server logs for details.")

# ✅ Endpoint to Clear Chat Memory
@app.post("/reset_memory")
async def reset_memory():
    memory.clear()
    return {"message": "Chat memory cleared successfully!"}

# ✅ Endpoint to Preload Schema to Vector Database
@app.post("/preload_schema")
async def preload_schema():
    try:
        await asyncio.to_thread(preload_schema_to_vectordb)
        return {"message": "Schema preloaded to vector database successfully!"}
    except Exception as e:
        logger.error(f"Error preloading schema: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error preloading schema: {e}")
