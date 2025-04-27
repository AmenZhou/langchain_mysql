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
from ..schema_vectorizer import SchemaVectorizer
from langchain_openai import ChatOpenAI
from ..langchain_config import (
    get_table_query_prompt,
    get_column_query_prompt,
    get_sanitize_prompt,
    create_db_chain_with_schema
)

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

async def generate_column_query(query: str) -> str:
    """Generate a query to get column information using LangChain's SQL chain."""
    try:
        # Get column query prompt
        prompt = get_column_query_prompt(query)
        
        # Extract table name
        table_name = extract_table_name(query)
        
        if not table_name:
            sql = """
            SELECT 
                table_name,
                GROUP_CONCAT(
                    CONCAT(
                        column_name,
                        ' (', column_type, ')',
                        IF(is_nullable = 'YES', ' NULL', ' NOT NULL'),
                        IF(column_comment != '', CONCAT(' - ', column_comment), '')
                    )
                    ORDER BY ordinal_position
                    SEPARATOR '\\n'
                ) as columns
            FROM information_schema.columns 
            GROUP BY table_name;
            """
        else:
            sql = f"""
            SELECT 
                column_name,
                column_type,
                is_nullable,
                column_comment,
                ordinal_position
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
            """
            
        chain = await get_sql_chain()
        result = await chain.invoke({
            "query": sql
        })
        return result["result"]
    except Exception as e:
        print(f"Error in generate_column_query: {e}")
        return None

async def generate_table_query(query: str) -> str:
    """Generate a query to get table information using LangChain's SQL chain."""
    try:
        # Get table query prompt
        prompt = get_table_query_prompt(query)
        
        # Extract meaningful words
        words = set(query.lower().split()) - {
            "show", "list", "tables", "table", "what", "are", "the", "in", "database",
            "with", "name", "names", "containing", "contains", "include", "includes",
            "having", "has", "that", "which", "where", "please", "can", "you", "tell",
            "me", "about", "find", "search", "for"
        }
        
        if not words:
            sql = "SHOW TABLES;"
        else:
            conditions = []
            for word in words:
                conditions.append(f"table_name LIKE '%{word}%'")
            sql = f"SELECT table_name FROM information_schema.tables WHERE {' AND '.join(conditions)};"
            
        chain = await get_sql_chain()
        result = await chain.invoke({
            "query": sql
        })
        return result["result"]
    except Exception as e:
        print(f"Error in generate_table_query: {e}")
        return None

async def refine_prompt_with_ai(query: str, schema_info: Optional[str] = None) -> Optional[str]:
    """Refine a natural language query into SQL using AI."""
    try:
        # Check if this is a table name query
        if any(phrase in query.lower() for phrase in ["show tables", "list tables", "what tables"]):
            return await generate_table_query(query)
            
        # Check if this is a column query    
        if any(phrase in query.lower() for phrase in ["what columns", "show columns", "list columns", "what's in", "what is in"]):
            return await generate_column_query(query)
            
        # For other queries, use the AI chain
        chain = await create_db_chain_with_schema(schema_info or "")
        response = await chain.ainvoke({"query": query})
        
        if not response or "error" in response.lower():
            # Try table query as fallback
            return await generate_table_query(query)
            
        return response
        
    except Exception as e:
        print(f"Error in refine_prompt_with_ai: {str(e)}")
        # Fallback to table query
        return await generate_table_query(query)

async def sanitize_sql_response(sql_result: str) -> str:
    """Sanitize SQL query results using AI."""
    try:
        # Get sanitize prompt from vector database
        prompt = get_sanitize_prompt(sql_result)
        
        # Initialize chat model with token-efficient settings
        chat = ChatOpenAI(
            model_name="gpt-3.5-turbo",  # Use smaller model
            temperature=0,
            request_timeout=30,
            max_tokens=500,  # Limit output length
            frequency_penalty=0.5,  # Encourage concise responses
            presence_penalty=0.5    # Encourage focused responses
        )
        
        # Get response
        response = await chat.agenerate([[AIMessage(content=prompt)]])
        
        if response and response.generations:
            return response.generations[0][0].text
            
        return sql_result
        
    except Exception as e:
        print(f"Error in sanitize_sql_response: {str(e)}")
        return sql_result

def extract_table_name(query: str) -> Optional[str]:
    """Extract table name from query."""
    query = query.lower()
    
    # Common patterns
    patterns = [
        "in table ", "table ", "for ", "of "
    ]
    
    for pattern in patterns:
        if pattern in query:
            parts = query.split(pattern)
            if len(parts) > 1:
                # Get the word after the pattern
                table_name = parts[1].split()[0]
                return table_name
                
    return None 
