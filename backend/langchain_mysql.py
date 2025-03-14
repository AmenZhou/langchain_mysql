from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
from openai import OpenAIError
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, MetaData
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
import re
from backend.included_tables import INCLUDED_TABLES

# ✅ Load Environment Variables
load_dotenv()

# ✅ FastAPI App
app = FastAPI()

# ✅ Secure API Key Handling
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Correct Database Connection (Ensure password is set!)
DB_URI = "mysql+pymysql://root:@mysql:3306/dev_tas_live"
engine = create_engine(DB_URI)

# ✅ Solution 1: Minimize or disable reflection for SQLDatabase
metadata = MetaData()  # Do not auto-reflect large foreign key relationships

# ✅ Include only necessary tables, skipping reflection
db = SQLDatabase(engine, metadata=metadata, include_tables=INCLUDED_TABLES)

# ✅ Use GPT-4 with Output Limit to Avoid Rate Limits
llm = ChatOpenAI(model="gpt-4", temperature=0, max_tokens=500)

# ✅ Summarized Memory to Reduce Token Usage
memory = ConversationSummaryMemory(llm=llm, memory_key="history")

# ✅ Define a prompt template with memory integration
prompt = PromptTemplate(
    input_variables=["history", "query"],
    template="Chat History: {history}\nUser: {query}"
)

# ✅ Solution 2: top_k=1 to reduce query complexity
db_chain = SQLDatabaseChain.from_llm(
    llm,
    db,
    verbose=False,
    return_intermediate_steps=False,
    memory=memory,
    top_k=1
)

# ✅ Input model
class QueryRequest(BaseModel):
    question: str

# ✅ Secondary Prompt to Review SQL Response and Remove PHI/PII (Allow IDs)
def sanitize_sql_response(response: str) -> str:
    review_prompt = f"""
    You are a data privacy filter. Your job is to review the SQL response below and redact any Protected Health Information (PHI) or Personally Identifiable Information (PII), including names, addresses, medical records, and other sensitive details.

    SQL Response:
    {response}

    Please return the sanitized response with all PHI/PII redacted, but allow numeric IDs (e.g., user IDs, document IDs) to remain.
    """
    sanitized_response = llm.predict(review_prompt)
    return sanitized_response

# ✅ Automatic Retry Handling for Rate Limits
def run_query_with_retry(query, retries=3):
    for attempt in range(retries):
        try:
            response = db_chain.run(query)
            response = sanitize_sql_response(response)  # Apply AI-based PHI/PII filtering
            memory.save_context({"query": query}, {"response": response})
            return response
        except OpenAIError as e:
            if "rate_limit_exceeded" in str(e):
                wait_time = 2**attempt
                print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    print("Failed after retries.")
    return None

# ✅ FastAPI Endpoint to Handle User Queries
@app.post("/query")
async def query_database(request: QueryRequest):
    try:
        response = run_query_with_retry(request.question)
        if not response:
            raise HTTPException(status_code=500, detail="No result returned from LangChain.")
        return {"result": response}  # Removed chat_history from response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Endpoint to Clear Chat Memory
@app.post("/reset_memory")
async def reset_memory():
    memory.clear()
    return {"message": "Chat memory cleared successfully!"}
