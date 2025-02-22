# ✅ Secure API Key Handling
try:
    from langchain.cache import BaseCache
except ImportError:
    class BaseCache:
        """A dummy BaseCache to satisfy type annotations."""
        pass

# Define a dummy Callbacks type if it's not available.
try:
    from langchain.schema import Callbacks
except ImportError:
    from typing import Any, List, Optional
    Callbacks = Optional[List[Any]]

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, MetaData
from openai import OpenAIError
import time
from langchain_community.utilities.sql_database import SQLDatabase as BaseSQLDatabase
# from pydantic import model_validator

# ✅ Load API Key Securely
openai_api_key = os.getenv("OPENAI_API_KEY")

# print("API Key", openai_api_key)

# ✅ Correct Database Connection (Ensure password is set!)
DB_URI = "mysql+pymysql://root:@mysql_telapp:3306/dev_tas_live"
engine = create_engine(DB_URI)

db = SQLDatabase(engine, include_tables=["consultations", "members"])

# ✅ Use GPT-4 Turbo with Output Limit to Avoid Rate Limits
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)

# ✅ Optimize Database Chain to Prevent Large Schema Fetching
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=False)

# ✅ Automatic Retry Handling for Rate Limits
def run_query_with_retry(query, retries=1):
    for attempt in range(retries):
        try:
            return db_chain.run(query)
        except OpenAIError as e:
            if "rate_limit_exceeded" in str(e):
                print(f"Rate limit exceeded, retrying in {2**attempt} seconds...")
                time.sleep(2**attempt)  # Exponential backoff
            else:
                raise e
    print("Failed after retries.")
    return None

query = "please fetch the last row on the members table, then count the total number of consultations that are linked to the member record"
response = run_query_with_retry(query)
print("Query Result:", response)


