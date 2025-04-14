from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from backend.database import db
from backend.schema_vectorizer import get_schema_as_text
import os
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
import openai
import random

# Cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def backoff_with_jitter(attempt: int, base_delay: float = 1.0, max_delay: float = 32.0) -> float:
    """Exponential backoff with jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter

class CachedChatOpenAI(ChatOpenAI):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Create cache key
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        # Check cache
        if cache_file.exists():
            cached_data = json.loads(cache_file.read_text())
            if time.time() - cached_data["timestamp"] < 3600:  # 1 hour cache
                return cached_data["response"]
        
        # Rate limiting
        max_retries = 5
        base_delay = 2
        for attempt in range(max_retries):
            try:
                response = super()._call(prompt, stop)
                # Cache the response
                cache_data = {
                    "timestamp": time.time(),
                    "response": response
                }
                cache_file.write_text(json.dumps(cache_data))
                return response
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    delay = backoff_with_jitter(attempt, base_delay)
                    time.sleep(delay)
                    continue
                raise

class MinimalSQLDatabase(SQLDatabase):
    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Return minimal table info to reduce token usage."""
        return ""

# Initialize OpenAI client with reduced token limits
openai_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=5,
    timeout=30
)

# Initialize chat model with caching and reduced token limits
chat_model = CachedChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    max_tokens=50,
    max_retries=5,
    timeout=30
)

# Initialize memory with reduced token limits
memory = ConversationSummaryMemory(
    llm=chat_model,
    memory_key="history",
    max_token_limit=50,
    output_key="result"
)

# Initialize database with minimal wrapper
db = MinimalSQLDatabase.from_uri(
    "mysql+pymysql://root:@mysql:3306/dev_tas_live",
    include_tables=["active_messages", "message_participants", "message_room_trigger_relations", "message_rooms", "system_messages"]
)

# Define a prompt template with memory integration and schema information
prompt = PromptTemplate(
    input_variables=["history", "query", "schema_info"],
    template="""Chat History: {history}
User: {query}

Relevant Tables and Columns:
{schema_info}

Write SQL based on the schema above. Use only the tables and columns shown. Keep queries simple and efficient."""
)

# Function to get relevant schema information based on the query
def get_relevant_schema_info(query):
    try:
        schema_info = get_schema_as_text(query=query, k=1)
        return schema_info
    except Exception as e:
        print(f"Error getting schema info: {e}")
        return ""

# Custom chain creation with schema info
def create_db_chain_with_schema(query):
    schema_info = get_relevant_schema_info(query)
    
    custom_prompt = prompt.format(
        history=memory.buffer,
        query=query,
        schema_info=schema_info
    )
    
    chain = SQLDatabaseChain.from_llm(
        chat_model,
        db,
        verbose=True,
        return_intermediate_steps=True,
        memory=memory,
        top_k=1,
        use_query_checker=False
    )
    
    return chain, custom_prompt

# Initialize the main database chain
db_chain = SQLDatabaseChain.from_llm(
    chat_model,
    db,
    verbose=True,
    return_intermediate_steps=True,
    memory=memory,
    top_k=1,
    use_query_checker=False
)

# Export llm for backward compatibility
llm = chat_model
