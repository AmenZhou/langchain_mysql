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
from database import get_db
from langchain_openai import ChatOpenAI
from langchain_config import (
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
from config import is_pii_filtering_enabled

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
    """Sanitize SQL query results using AI to remove PII/PHI."""
    if not sql_result:
        raise ValueError("Invalid SQL response")
    
    # Check global configuration for PII filtering
    if not is_pii_filtering_enabled():
        logger.info("⚠️  PII sanitization DISABLED - skipping LLM call and returning original response")
        return str(sql_result).strip().rstrip(';')
    
    # PII filtering is enabled - perform sanitization
    try:
        # Get the sanitization prompt
        sanitize_prompt = get_sanitize_prompt(str(sql_result))
        
        # Create LLM instance for sanitization
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["sanitize_prompt"],
            template="{sanitize_prompt}"
        )
        
        # Create chain and invoke
        chain = prompt_template | llm
        result = await chain.ainvoke({"sanitize_prompt": sanitize_prompt})
        
        # Extract content from result
        if hasattr(result, 'content'):
            sanitized_result = result.content
        else:
            sanitized_result = str(result)
            
        logger.info(f"Successfully sanitized SQL result")
        return sanitized_result.strip()
        
    except Exception as e:
        logger.error(f"Error in PII sanitization: {str(e)}")
        # Fallback to basic sanitization if LLM fails
        logger.warning("Falling back to basic sanitization due to error")
        return str(sql_result).strip().rstrip(';')

async def sanitize_query_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize structured query data while preserving format."""
    if not data:
        return data
    
    # Check global configuration for PII filtering
    if not is_pii_filtering_enabled():
        logger.info(f"⚠️  PII filtering DISABLED - returning {len(data)} rows of unfiltered data")
        return data
    
    # PII filtering is enabled - perform sanitization
    try:
        # Process each row to sanitize PII fields
        sanitized_data = []
        
        for row in data:
            sanitized_row = {}
            for key, value in row.items():
                # Check if this field contains PII that should be filtered
                if value is None:
                    sanitized_row[key] = value
                elif _is_pii_field(key, str(value)):
                    sanitized_row[key] = "[PRIVATE]"
                else:
                    sanitized_row[key] = value
            sanitized_data.append(sanitized_row)
        
        logger.info(f"Successfully sanitized {len(sanitized_data)} rows of structured query data")
        return sanitized_data
        
    except Exception as e:
        logger.error(f"Error in structured data PII sanitization: {str(e)}")
        # Fallback to original data if sanitization fails
        logger.warning("Falling back to original data due to sanitization error")
        return data

def _is_pii_field(field_name: str, value: str) -> bool:
    """Check if a field contains PII based on field name and value patterns."""
    if not value or value == "None":
        return False
    
    # Field names that typically contain PII
    pii_field_names = {
        'email', 'phone', 'ssn', 'social_security', 'passport', 'license',
        'first_name', 'last_name', 'full_name', 'name', 'address', 'street',
        'city', 'zip', 'postal', 'credit_card', 'bank_account', 'routing',
        'dob', 'birth_date', 'birthday', 'personal_id', 'patient_id',
        'medical_record', 'diagnosis', 'prescription', 'salary', 'income'
    }
    
    # Check if field name contains PII indicators
    field_lower = field_name.lower()
    if any(pii_term in field_lower for pii_term in pii_field_names):
        return True
    
    # Value patterns that indicate PII
    import re
    
    # Email pattern
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
        return True
    
    # Phone pattern (various formats)
    if re.match(r'^[\+]?[1-9]?[\d\s\-\(\)\.]{7,15}$', value.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')):
        return True
    
    # SSN pattern (XXX-XX-XXXX)
    if re.match(r'^\d{3}-\d{2}-\d{4}$', value):
        return True
    
    # Credit card pattern (basic check for 13-19 digits)
    if re.match(r'^\d{13,19}$', value.replace(' ', '').replace('-', '')):
        return True
    
    return False

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
