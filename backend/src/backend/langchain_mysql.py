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

from .database import get_db_engine, get_db  # Import database functions
from .langchain_config import chat_model, memory, db_chain, create_db_chain_with_schema, backoff_with_jitter  # Import Langchain components
from .models import QueryRequest  # Import the Pydantic model
from .utils import refine_prompt_with_ai, sanitize_sql_response  # Import utility functions
from .schema_vectorizer import SchemaVectorizer  # Import schema vectorization

# ✅ Load Environment Variables
load_dotenv()

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize schema vectorizer
schema_vectorizer = SchemaVectorizer()

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
    def __init__(self, db_engine, schema_vectorizer):
        self.db_engine = db_engine
        self.schema_vectorizer = schema_vectorizer
        self.logger = logging.getLogger(__name__)
    
    # ✅ Asynchronous Query Execution to Improve Speed
    async def run_query_with_retry(self, user_query, retries=5, use_schema=True):
        refined_query = await refine_prompt_with_ai(user_query)

        if not refined_query:
            self.logger.error("Failed to refine the prompt. Using the original query.")
            refined_query = user_query

        for attempt in range(retries):
            try:
                self.logger.info(f"Attempt {attempt + 1} to run refined query: {refined_query}")
                
                # Use schema-enhanced chain if requested
                if use_schema:
                    # Get a custom chain with schema information for this query
                    custom_chain, prompt_with_schema = create_db_chain_with_schema(refined_query)
                    self.logger.info(f"Using schema-enhanced prompt: {prompt_with_schema[:100]}...")
                    response_dict = await asyncio.to_thread(lambda: custom_chain.invoke({"query": refined_query}))
                else:
                    # Use the original chain without schema enhancement
                    response_dict = await asyncio.to_thread(lambda: db_chain.invoke({"query": refined_query}))
                
                self.logger.info(f"Raw response from db_chain: {response_dict}")
                response = response_dict.get("result", "")
                response = await sanitize_sql_response(response)
                memory.save_context({"query": user_query}, {"result": response})
                return response
            except OpenAIError as e:
                if "rate_limit_exceeded" in str(e) or "429" in str(e):
                    # Use our custom backoff strategy
                    wait_time = await asyncio.to_thread(backoff_with_jitter, attempt)
                    self.logger.warning(f"Rate limit exceeded, retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{retries})")
                    await asyncio.sleep(wait_time)
                    # If this was our last retry, raise the exception
                    if attempt == retries - 1:
                        self.logger.error(f"Failed after {retries} retries due to rate limits")
                        raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded. Please try again later.")
                else:
                    self.logger.error(f"OpenAI Error during query: {e}")
                    self.logger.error(traceback.format_exc())
                    raise
            except (ProgrammingError, sqlalchemy.exc.ProgrammingError) as e:
                self.logger.error(f"Database Error during query: {e}")
                self.logger.error(traceback.format_exc())
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
                self.logger.error(f"General error during query: {e}")
                self.logger.error(traceback.format_exc())
                if isinstance(e, ProgrammingError) and "You have an error in your SQL syntax" in str(e) and "near 'LIMIT 5' at line 4" in str(e):
                    try:
                        intermediate_steps = response_dict.get("intermediate_steps")
                        if intermediate_steps and len(intermediate_steps) > 1:
                            sql_command = intermediate_steps[1].sql_
                            if sql_command.lower().startswith("select count(*)"):
                                modified_sql = sql_command.replace("LIMIT 5;", "").replace("LIMIT 5", "")
                                self.logger.info(f"Retrying with modified SQL: {modified_sql}")
                                with get_db_engine().connect() as connection:
                                    result = connection.execute(text(modified_sql)).scalar_one()
                                return str(result)
                    except Exception as ex:
                        self.logger.error(f"Error during workaround: {ex}")
                        self.logger.error(traceback.format_exc())
                raise
        self.logger.error("Failed after retries.")
        return None

    # ✅ Query Handler
    async def query_database(self, question: str):
        try:
            self.logger.info(f"Received user query: {question}")
            response = await self.run_query_with_retry(question, use_schema=True)
            if not response:
                self.logger.error("No result returned from LangChain.")
                raise HTTPException(status_code=500, detail="No result returned from LangChain.")
            self.logger.info(f"Final response: {response}")
            return {"result": response}
        except HTTPException as e:
            raise e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in query: {e}")
            self.logger.error(traceback.format_exc())
            if "rate_limit" in str(e).lower():
                raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded. Please try again later.")
            raise HTTPException(status_code=500, detail="An unexpected error occurred. Check server logs for details.")

    # ✅ Memory Management
    async def reset_memory(self):
        memory.clear()
        return {"message": "Chat memory cleared successfully!"}

    # ✅ Schema Management
    async def preload_schema(self):
        try:
            await asyncio.to_thread(lambda: self.schema_vectorizer.preload_schema_to_vectordb())
            return {"message": "Schema preloaded to vector database successfully!"}
        except Exception as e:
            self.logger.error(f"Error preloading schema: {e}")
            self.logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error preloading schema: {e}")
