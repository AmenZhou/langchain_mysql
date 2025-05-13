import os
import traceback
from openai import OpenAIError, AsyncOpenAI
from sqlalchemy import text
from typing import Optional, List, Dict, Any
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ..database import get_db
from langchain_openai import ChatOpenAI
from ..langchain_config import (
    get_table_query_prompt,
    get_column_query_prompt,
    get_sanitize_prompt,
    create_db_chain_with_schema
)
import re

# ✅ Secure API Key Handling
def get_openai_client() -> AsyncOpenAI:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    return AsyncOpenAI(api_key=openai_api_key)

# ✅ Configure Logging
import logging
logger = logging.getLogger(__name__)

async def get_sql_chain():
    """Create a SQL chain using LangChain's built-in tools."""
    db = SQLDatabase.from_uri(os.getenv("DATABASE_URL"))
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    return SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        verbose=True,
        return_intermediate_steps=True
    )

async def generate_column_query(query: str, table_name: str = None, schema_info: str = None) -> str:
    """Generate SQL query for column information."""
    try:
        chain = await get_sql_chain()
        result = await chain.ainvoke({
            "query": query,
            "table_name": table_name,
            "schema_info": schema_info
        })
        return result["result"]
    except Exception as e:
        logger.error(f"Error in generate_column_query: {str(e)}")
        raise

async def generate_table_query(query: str, schema_info: str = None) -> str:
    """Generate SQL query for table information."""
    try:
        chain = await get_sql_chain()
        result = await chain.ainvoke({
            "query": query,
            "schema_info": schema_info
        })
        return result["result"]
    except Exception as e:
        logger.error(f"Error in generate_table_query: {str(e)}")
        raise

async def refine_prompt_with_ai(query: str, schema_info: str = None, prompt_type: str = None) -> str:
    """Refine natural language query into SQL using AI."""
    try:
        if prompt_type == "table":
            return await generate_table_query(query, schema_info)
        elif prompt_type == "column":
            return await generate_column_query(query, schema_info=schema_info)
        else:
            chain = await get_sql_chain()
            result = await chain.ainvoke({
                "query": query,
                "schema_info": schema_info
            })
            return result["result"]
    except Exception as e:
        logger.error(f"Error in refine_prompt_with_ai: {str(e)}")
        raise

async def sanitize_sql_response(sql_result: str) -> str:
    """Sanitize SQL query results using AI."""
    if not sql_result:
        raise ValueError("Invalid SQL response")
    return sql_result.strip().rstrip(';')

def extract_table_name(query: str) -> str:
    """Extract table name from a query string."""
    if not query:
        raise ValueError("Query cannot be empty")
        
    # Common SQL patterns
    patterns = [
        r"FROM\s+(\w+)",  # SELECT ... FROM table
        r"INTO\s+(\w+)",  # INSERT INTO table
        r"UPDATE\s+(\w+)", # UPDATE table
        r"JOIN\s+(\w+)",  # ... JOIN table
        r"TABLE\s+(\w+)"  # ... TABLE table
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
            
    raise ValueError("Could not extract table name from query")
