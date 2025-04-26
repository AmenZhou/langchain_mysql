import os
import traceback
from openai import OpenAIError, AsyncOpenAI
from sqlalchemy import text
from typing import Optional, List
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
from .database import get_db
from .schema_vectorizer import SchemaVectorizer
from langchain_openai import ChatOpenAI

# ✅ Secure API Key Handling
def get_openai_client() -> AsyncOpenAI:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    return AsyncOpenAI(api_key=openai_api_key)

# ✅ Configure Logging (You might keep this in the main file as well)
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
        chain = await get_sql_chain()
        result = await chain.invoke({
            "query": f"Show me the columns and their details for the table {query}"
        })
        return result["result"]
    except Exception as e:
        print(f"Error in generate_column_query: {e}")
        return None

async def generate_table_query(query: str) -> str:
    """Generate a query to get table information using LangChain's SQL chain."""
    try:
        chain = await get_sql_chain()
        result = await chain.invoke({
            "query": f"Show me all tables in the database that match: {query}"
        })
        return result["result"]
    except Exception as e:
        print(f"Error in generate_table_query: {e}")
        return None

async def refine_prompt_with_ai(query: str) -> Optional[str]:
    try:
        schema_vectorizer = SchemaVectorizer()
        schema_docs = schema_vectorizer.get_relevant_schema()
        
        # Check query type
        query_lower = query.lower()
        is_table_query = any(keyword in query_lower for keyword in [
            "table", "tables", "schema", "database", "show me", "list", "what tables",
            "which table", "which tables", "find table", "find tables"
        ])
        
        is_column_query = any(keyword in query_lower for keyword in [
            "column", "columns", "what columns", "which columns", "show columns",
            "describe", "structure", "fields", "what's in", "what is in"
        ])
        
        # Handle column queries
        if is_column_query:
            try:
                result = await generate_column_query(query)
                if result:
                    return result
            except Exception as e:
                print(f"Error in column query: {e}")
                return None
            
        # Handle table queries
        if is_table_query:
            try:
                result = await generate_table_query(query)
                if result:
                    return result
            except Exception as e:
                print(f"Error in table query: {e}")
                return None
            
        # For other queries, use the AI model
        try:
            openai_client = get_openai_client()
            messages = [
                {"role": "system", "content": "You are a helpful assistant that refines SQL queries based on schema information."},
                {"role": "user", "content": f"Schema information:\n{schema_docs}\n\nOriginal query: {query}\n\nPlease refine this query."}
            ]
            
            response = await openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAI call: {e}")
            return None
            
    except Exception as e:
        print(f"Error in refine_prompt_with_ai: {e}")
        return None

# ✅ Secondary Prompt to Review SQL Response and Remove PHI/PII (Allow IDs)
async def sanitize_sql_response(response: str) -> str:
    """Sanitize SQL response by removing sensitive information while preserving structure."""
    try:
        system_prompt = """You are a data privacy assistant. Your task is to sanitize SQL query results by:
        1. Replacing all personal information (names, emails, phone numbers) with [PRIVATE]
        2. Preserving IDs, dates, and non-sensitive data
        3. Maintaining the table structure and formatting
        4. If the input is a SQL query or error message, return it unchanged
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please sanitize this SQL response:\n{response}")
        ]
        
        llm = ChatOpenAI(model_name="gpt-4")
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm
        
        result = await chain.ainvoke({"response": response})
        return result.content if isinstance(result, AIMessage) else response
        
    except Exception as e:
        logger.error(f"Error in sanitize_sql_response: {str(e)}\n{traceback.format_exc()}")
        return response
