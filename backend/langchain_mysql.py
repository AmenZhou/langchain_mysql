from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ✅ Import CORS Middleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
import asyncio
from openai import OpenAIError, OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, MetaData, text
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
import re
from backend.included_tables import INCLUDED_TABLES
import logging
import traceback
from pymysql.err import ProgrammingError

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
    allow_origins=["*"],  # Allow all origins (set specific domains for security)
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # ✅ Allow all headers
)

# ✅ Secure API Key Handling
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

# ✅ Correct Database Connection (Ensure password is set!)
DB_URI = "mysql+pymysql://root:@mysql:3306/dev_tas_live"
engine = create_engine(DB_URI)

# ✅ Solution 1: Minimize or disable reflection for SQLDatabase
metadata = MetaData()  # Do not auto-reflect large foreign key relationships

# ✅ Include only necessary tables, skipping reflection
db = SQLDatabase(engine, metadata=metadata, include_tables=INCLUDED_TABLES)

# ✅ Use GPT-3.5-Turbo for Faster Processing
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)

# ✅ Summarized Memory to Reduce Token Usage
memory = ConversationSummaryMemory(llm=llm, memory_key="history", max_token_limit=200, output_key="result")

# ✅ Define a prompt template with memory integration
prompt = PromptTemplate(
    input_variables=["history", "query"],
    template="Chat History: {history}\nUser: {query}"
)

# ✅ Solution 2: top_k=1 to reduce query complexity
db_chain = SQLDatabaseChain.from_llm(
    llm,
    db,
    verbose=True,
    return_intermediate_steps=True,
    memory=memory,
    top_k=1
)

# ✅ Input model
class QueryRequest(BaseModel):
    question: str

# ✅ Prompt for the Refinement AI
PROMPT_REFINE = """You are an expert prompt engineer. Your task is to take a user's natural language query and refine it so that it will reliably generate a raw SQL query using a language model connected to a SQL database via Langchain. The goal is to get only the SQL query without any extra explanations or execution results.

Here is the user's query: '{user_query}'

Please rewrite the user's query to be more explicit and direct in asking for the raw SQL query. Ensure the refined query clearly specifies the tables and columns involved and asks for the SQL to be returned without execution or additional text."""

# ✅ Function to refine the prompt using AI
async def refine_prompt_with_ai(user_query: str) -> str | None:
    try:
        refine_prompt = PROMPT_REFINE.format(user_query=user_query)
        refinement_response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can choose a different model if you prefer
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=0.2,  # Keep it relatively low for more consistent refinement
        )
        refined_query = refinement_response.choices[0].message.content.strip()
        logger.info(f"Refined query: {refined_query}")
        return refined_query
    except OpenAIError as e:
        logger.error(f"Error during prompt refinement: {e}")
        logger.error(traceback.format_exc())
        return None
    except Exception as e:
        logger.error(f"Unexpected error during prompt refinement: {e}")
        logger.error(traceback.format_exc())
        return None

# ✅ Secondary Prompt to Review SQL Response and Remove PHI/PII (Allow IDs)
def sanitize_sql_response(response: str) -> str:
    if len(response) > 1000:
        response = response[:1000] + " ... [Truncated for Speed]"

    review_prompt = f"""
    You are a data privacy filter. Your job is to review the SQL response below and redact any Protected Health Information (PHI) or Personally Identifiable Information (PII), including names, addresses, medical records, and other sensitive details.

    SQL Response:
    {response}

    Please return the sanitized response with all PHI/PII redacted, but allow numeric IDs (e.g., user IDs, document IDs) to remain.
    """
    try:
        sanitized_response = llm.predict(review_prompt)
        return sanitized_response
    except OpenAIError as e:
        logger.error(f"Error during sanitization: {e}")
        return response  # Return the original response if sanitization fails

# ✅ Asynchronous Query Execution to Improve Speed
async def run_query_with_retry(user_query, retries=3):
    refined_query = await refine_prompt_with_ai(user_query)

    if not refined_query:
        logger.error("Failed to refine the prompt. Using the original query.")
        refined_query = user_query

    for attempt in range(retries):
        try:
            logger.info(f"Attempt {attempt + 1} to run refined query: {refined_query}")
            response_dict = await asyncio.to_thread(db_chain, {"query": refined_query})
            logger.info(f"Raw response from db_chain: {response_dict}")
            response = response_dict.get("result", "")
            response = sanitize_sql_response(response)
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
        response = await run_query_with_retry(request.question)
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
