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

from backend.database import engine, db  # Import database objects
from backend.langchain_config import llm, memory, db_chain, create_db_chain_with_schema  # Import Langchain components
from backend.models import QueryRequest  # Import the Pydantic model
from backend.utils import refine_prompt_with_ai, sanitize_sql_response  # Import utility functions
from backend.schema_vectorizer import preload_schema_to_vectordb  # Import schema vectorization

# ✅ Load Environment Variables
load_dotenv()

# ✅ FastAPI App
app = FastAPI()

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
async def run_query_with_retry(user_query, retries=3, use_schema=True):
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
                response_dict = await asyncio.to_thread(custom_chain, {"query": refined_query})
            else:
                # Use the original chain without schema enhancement
                response_dict = await asyncio.to_thread(db_chain, {"query": refined_query})
            
            logger.info(f"Raw response from db_chain: {response_dict}")
            response = response_dict.get("result", "")
            response = await sanitize_sql_response(response, llm)
            memory.save_context({"query": user_query}, {"result": response})
            return response
        except OpenAIError as e:
            if "rate_limit_exceeded" in str(e):
                wait_time = 2**attempt
                print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"OpenAI Error during query: {e}")
                logger.error(traceback.format_exc())
                raise
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
    except OpenAIError as e:
        logger.error(f"OpenAI Error in /query endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"OpenAI Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in /query endpoint: {e}")
        logger.error(traceback.format_exc())
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

# ✅ Preload schema on startup
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Preloading schema to vector database on startup...")
        await asyncio.to_thread(preload_schema_to_vectordb)
        logger.info("Schema preloaded successfully!")
    except Exception as e:
        logger.error(f"Error preloading schema on startup: {e}")
        logger.error(traceback.format_exc())
