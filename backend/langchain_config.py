import os
import json
import random
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
import logging
from pathlib import Path
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.utilities import SQLDatabase
from openai import RateLimitError, APIError
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from pydantic import Field
import sys

from prompts import PROMPT_REFINE, PROMPT_TABLE_QUERY, get_sanitize_prompt
from exceptions import OpenAIRateLimitError, OpenAIAPIError

if TYPE_CHECKING:
    from schema_vectorizer import SchemaVectorizer

logger = logging.getLogger(__name__)

# Cache directory - this is a global fallback, but instance will manage its own
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
    # Declare cache_dir_path as a Pydantic field for this subclass
    # This allows it to be configured during instantiation if needed, e.g. CachedChatOpenAI(cache_dir_path="my_new_cache_dir")
    cache_dir_path: str = Field(default=".langchain_cache")

    # These will be instance variables, not Pydantic fields, initialized in model_post_init
    _cache: Dict[str, Any]
    _cache_file: str

    def model_post_init(self, __context: Any) -> None:
        """Initialize cache-related attributes after Pydantic model fully initializes."""
        super().model_post_init(__context)

        # Initialize instance variables for cache
        # self.cache_dir_path is now a proper Pydantic field, set either by default or instantiation
        os.makedirs(self.cache_dir_path, exist_ok=True)
        self._cache_file = os.path.join(self.cache_dir_path, "openai_llm_cache.json")
        self._cache = self._load_cache_from_file()
        logger.info(f"CachedChatOpenAI initialized. Cache file: {self._cache_file}")

    def _load_cache_from_file(self) -> Dict[str, Any]:
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Cache file {self._cache_file} is corrupted. Starting with an empty cache.")
                return {}
        return {}

    def _save_cache_to_file(self) -> None:
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.error(f"Error saving cache to file {self._cache_file}: {e}")

    def _generate_cache_key(self, messages) -> str:
        # Simple hash of messages. Consider a more robust serialization if messages are complex.
        return str(hash(str(messages)))

    async def agenerate(self, messages, **kwargs):
        # Ensure cache is initialized (it should be by model_post_init)
        if not hasattr(self, '_cache'):
            # This is a fallback, should not be normally hit if model_post_init worked
            logger.warning("Cache not initialized in agenerate, attempting re-init.")
            self.cache_dir_path = getattr(self, 'cache_dir_path', ".langchain_cache")
            os.makedirs(self.cache_dir_path, exist_ok=True)
            self._cache_file = os.path.join(self.cache_dir_path, "openai_llm_cache.json")
            self._cache = self._load_cache_from_file()

        cache_key = self._generate_cache_key(messages)
        if cache_key in self._cache:
            logger.info(f"Returning cached LLM response for key: {cache_key}")
            return self._cache[cache_key]
        
        logger.info(f"Cache miss for LLM key: {cache_key}. Calling API.")
        try:
            response = await super().agenerate(messages, **kwargs)
            self._cache[cache_key] = response
            self._save_cache_to_file()
            return response
        except RateLimitError as e:
            raise OpenAIRateLimitError("OpenAI API rate limit exceeded") from e
        except APIError as e:
            raise OpenAIAPIError("OpenAI API error occurred") from e
        except Exception as e:
            logger.error(f"Unexpected error during LLM call or caching: {e}")
            raise

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

def get_relevant_prompt(query: str, prompt_type: Optional[str], vectorizer: Optional['SchemaVectorizer']) -> str:
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

async def create_db_chain_with_schema(
    schema_info: str, 
    memory: ConversationBufferMemory,
    llm: Optional[ChatOpenAI] = None
) -> LLMChain:
    try:
        if llm is None:
            # Instantiate CachedChatOpenAI. If cache_dir_path needs to be different from default, pass it here.
            llm = CachedChatOpenAI( 
                model_name="gpt-3.5-turbo",
                temperature=0,
                max_tokens=1000
                # e.g., cache_dir_path=".custom_cache_directory"
            )
        
        template = """You are an AI assistant that converts natural language queries into SQL queries against the provided database schema.
Use the conversation history to understand context from previous questions if relevant.

Database Schema:
{schema_info}

Conversation History:
{history}

User's Current Question: {input}

Based on the conversation history and the user's current question, generate the SQL query.
Return only the SQL query. Do not include any additional text, preamble, or explanation.
SQL Query:"""
        
        prompt = PromptTemplate(
            input_variables=["schema_info", "history", "input"],
            template=template
        )
        
        try:
            chain = LLMChain(
                llm=llm,
                prompt=prompt,
                memory=memory,
                verbose=True
            )
            return chain
        except Exception as e:
            logging.error(f"Error creating LLMChain: {str(e)}")
            raise Exception(f"Failed to create database chain with LLMChain: {str(e)}")
    except Exception as e:
        logging.error(f"Error creating database chain: {str(e)}")
        raise Exception(f"Failed to create database chain: {str(e)}")
