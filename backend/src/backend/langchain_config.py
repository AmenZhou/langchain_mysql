import os
import json
import random
from typing import Optional, List, Dict, Any, Union
import logging
from pathlib import Path
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.utilities import SQLDatabase
from openai import RateLimitError, APIError

from .prompts import PROMPT_REFINE, PROMPT_TABLE_QUERY, get_sanitize_prompt
from .schema_vectorizer import SchemaVectorizer
from .exceptions import OpenAIRateLimitError, OpenAIAPIError

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def backoff_with_jitter(attempt: int, base_delay: float = 1.0, max_delay: float = 32.0) -> float:
    """Exponential backoff with jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter

class MinimalSQLDatabase(SQLDatabase):
    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Return minimal table info to reduce token usage."""
        return ""

class CachedChatOpenAI(ChatOpenAI):
    def __init__(self, cache_dir: str = ".cache", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "openai_cache.json")
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def _generate_cache_key(self, messages) -> str:
        return str(hash(str(messages)))

    async def agenerate(self, messages, **kwargs):
        try:
            cache_key = self._generate_cache_key(messages)
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            response = await super().agenerate(messages, **kwargs)
            self.cache[cache_key] = response
            self._save_cache()
            return response
        except RateLimitError as e:
            raise OpenAIRateLimitError("OpenAI API rate limit exceeded") from e
        except APIError as e:
            raise OpenAIAPIError("OpenAI API error occurred") from e

def get_table_query_prompt(query: str) -> str:
    """Get the prompt for table name queries."""
    if not query:
        raise ValueError("Query cannot be empty")
    return PROMPT_TABLE_QUERY.format(schema="", query=query)

def get_column_query_prompt(query: str) -> str:
    """Get the prompt for column name queries."""
    if not query:
        raise ValueError("Query cannot be empty")
    return PROMPT_REFINE.format(schema="", query=query)

def get_sanitize_prompt(sql_result: str) -> str:
    """Get the prompt for sanitizing SQL responses."""
    if not sql_result:
        raise ValueError("SQL result cannot be empty")
    return f"""Please clean up and sanitize the following SQL result to ensure no sensitive data is exposed:

{sql_result}

Rules for sanitization:
1. Replace sensitive column values with [PRIVATE]
2. Keep table names and column names visible
3. Keep SQL keywords and syntax visible
4. Keep non-sensitive values (like IDs) visible
5. Return only the sanitized SQL, no explanations"""

def get_relevant_prompt(query: str, prompt_type: Optional[str], vectorizer: Optional[SchemaVectorizer]) -> str:
    """
    Returns a relevant prompt based on the query and type.
    Falls back to default prompts if vectorizer is not available.
    """
    if not query:
        raise ValueError("Query cannot be empty")
        
    if not vectorizer:
        if prompt_type == "table":
            return PROMPT_TABLE_QUERY
        elif prompt_type == "sanitize":
            return get_sanitize_prompt
        return PROMPT_REFINE
        
    try:
        return vectorizer.get_relevant_prompt(query, prompt_type)
    except Exception as e:
        logger.warning(f"Failed to get relevant prompt from vectorizer: {e}")
        # Fall back to default prompts
        if prompt_type == "table":
            return PROMPT_TABLE_QUERY
        elif prompt_type == "sanitize":
            return get_sanitize_prompt
        return PROMPT_REFINE

async def create_db_chain_with_schema(schema_info: str) -> Chain:
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=1000
        )
        
        prompt = PromptTemplate(
            input_variables=["schema_info", "query"],
            template="""Given the following SQL schema information:
            {schema_info}
            
            Convert the following natural language query into a SQL query:
            {query}
            
            Return only the SQL query without any additional text or explanation."""
        )
        
        chain = prompt | llm
        return chain
    except Exception as e:
        logging.error(f"Error creating database chain: {str(e)}")
        raise
