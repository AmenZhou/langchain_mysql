from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
from openai import OpenAIError
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine

# ✅ Load Environment Variables
load_dotenv()

# ✅ FastAPI App
app = FastAPI()

# ✅ Secure API Key Handling
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Correct Database Connection (Ensure password is set!)
DB_URI = "mysql+pymysql://root:@mysql_telapp:3306/dev_tas_live"
engine = create_engine(DB_URI)

# ✅ Include only necessary tables
db = SQLDatabase(engine, include_tables=["consultations", "members", "persons", "invoices", "invoice_items"])

# ✅ Use GPT-3.5 Turbo with Output Limit to Avoid Rate Limits
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)

# ✅ Optimize Database Chain to Prevent Large Schema Fetching
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=False)

# ✅ Input model
class QueryRequest(BaseModel):
    question: str

# ✅ Automatic Retry Handling for Rate Limits
def run_query_with_retry(query, retries=3):
    for attempt in range(retries):
        try:
            return db_chain.run(query)
        except OpenAIError as e:
            if "rate_limit_exceeded" in str(e):
                wait_time = 2**attempt  # Exponential backoff (2, 4, 8 sec)
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
        return {"result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
