from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import time
import asyncio
import logging
import traceback
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, OperationalError
from typing import Optional, Dict, Any, List
from fastapi import status
from openai import RateLimitError, APIError
from contextlib import asynccontextmanager
from sqlalchemy import text, create_engine
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from .langchain_config import create_db_chain_with_schema, backoff_with_jitter, CachedChatOpenAI
from .models import QueryRequest, QueryResponse, ResponseType
from .utils import refine_prompt_with_ai, sanitize_sql_response
from .schema_vectorizer import SchemaVectorizer
from .db_utils import get_database_url

# ✅ Load Environment Variables
load_dotenv()

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MYSQL_URL = get_database_url()

# Initialize schema vectorizer
schema_vectorizer = SchemaVectorizer(db_url=MYSQL_URL)

# Global store for session chat histories (simple in-memory implementation)
# For production, consider a more robust persistent store (e.g., Redis, DB)
session_chat_histories: Dict[str, ConversationBufferMemory] = {}
DEFAULT_SESSION_ID = "global_session" # For requests without a session_id

# Corrected Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("!!!!!!!!!! EXECUTING THE CORRECT LIFESPAN FUNCTION NOW !!!!!!!!!!")
    # Startup
    logger.info("Application startup: Initiating schema preloading...")
    try:
        # Preloading logic using the global schema_vectorizer:
        schema_info = await schema_vectorizer.extract_table_schema()
        if schema_info:
            await schema_vectorizer.initialize_vector_store(schema_info)
            logger.info("Schema preloaded and vector store initialized successfully on startup!")
        else:
            logger.warning("No schema info extracted on startup, vector store not initialized with schema.")
    except Exception as e:
        logger.error(f"Error during application startup (schema preloading): {e}")
        logger.error(traceback.format_exc())

    logger.info("Application startup sequence complete.")
    yield

    # Shutdown
    logger.info("Application shutdown sequence started.")
    # Add any cleanup code here if needed
    logger.info("Application shutdown complete.")

class LangChainMySQL:
    def __init__(self):
        self.engine = create_engine(MYSQL_URL, pool_pre_ping=True)
        self.schema_vectorizer = SchemaVectorizer(db_url=MYSQL_URL)
        self.llm = CachedChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

    async def run_query_with_retry(self, query: str, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Run a query with retry logic and return structured data."""
        attempt = 0
        last_error = None
        while attempt < max_retries:
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(query))
                    column_names = result.keys()
                    rows = result.fetchall()
                    # Convert to list of dictionaries for JSON serialization
                    return [dict(zip(column_names, row)) for row in rows]
            except Exception as e:
                last_error = e
                logger.warning(f"Query attempt {attempt + 1} failed: {e}. Retrying...")
                attempt += 1
                if attempt == max_retries:
                    logger.error(f"Failed to execute query after {max_retries} attempts: {last_error}")
                    raise Exception(f"Max retries exceeded: {str(last_error)}")
                delay = backoff_with_jitter(attempt)
                await asyncio.sleep(delay)
        # This line should not be reached if logic is correct, added for pylint        
        raise Exception("Query execution failed unexpectedly after retries.") 

    async def generate_explanation(self, sql_query: str, data: List[Dict[str, Any]]) -> str:
        """Generate a natural language explanation of the SQL query and results."""
        try:
            explanation_prompt_template = PromptTemplate(
                input_variables=["sql_query", "data"],
                template='''Given the SQL query:
{sql_query}

And the results:
{data}

Provide a clear, concise explanation of what this query is doing and what the results show. 
Explain in natural language that a non-technical person would understand.'''
            )
            # Use a fresh LLM instance or the instance's LLM for explanation
            llm_explainer = self.llm # Or CachedChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
            chain = explanation_prompt_template | llm_explainer
            result = await chain.ainvoke({"sql_query": sql_query, "data": str(data)})
            
            if hasattr(result, 'content'):
                return result.content
            
            return str(result)
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return f"Unable to generate explanation due to error: {str(e)}"

    async def process_query(self, query: str, session_id: str, prompt_type: Optional[str] = None, response_type: str = "all") -> Dict[str, Any]:
        """Process a natural language query and return results based on response_type.
        
        Args:
            query: The natural language query
            session_id: The session ID for conversation memory
            prompt_type: Type of prompt to use
            response_type: Type of response to return (sql, data, natural_language, all)
        
        Returns:
            Dictionary with results based on response_type
        """
        try:
            logger.info(f"Processing query: '{query}' for session_id: {session_id}, response_type: {response_type}")
            if not query:
                logger.error("Empty query received")
                raise HTTPException(
                    status_code=422,
                    detail="Query cannot be empty"
                )

            # Get or create conversation memory for the session
            if session_id not in session_chat_histories:
                logger.info(f"Creating new memory for session_id: {session_id}")
                # input_key='input' matches the 'input' variable in the LLMChain prompt
                session_chat_histories[session_id] = ConversationBufferMemory(memory_key="history", input_key="input", return_messages=True)
            current_memory = session_chat_histories[session_id]

            logger.info("Getting relevant schema using global schema_vectorizer for query context.")
            # Use the global schema_vectorizer that was preloaded
            schema_info_str = await schema_vectorizer.get_relevant_schema(query)
            logger.info(f"Schema info string: {schema_info_str[:500]}...") # Log a snippet
            
            if not schema_info_str:
                logger.warning("No relevant schema information found by vectorizer for the query.")
                # Fallback or error handling - for now, proceed but chain might fail or hallucinate
                # Consider raising an HTTPException here if schema is absolutely critical
                # schema_info_str = "No schema information available." # Or raise error
                raise HTTPException(
                    status_code=422,
                    detail={"error": "Schema Error", "details": "No relevant schema information found for the query. Please try rephrasing."}
                )
            
            logger.info("Creating database chain with memory and schema.")
            # Pass the current_memory to the chain creation function
            db_chain = await create_db_chain_with_schema(schema_info_str, memory=current_memory, llm=self.llm)
            
            logger.info("Invoking database chain.")
            # LLMChain expects a dictionary input. 'input' is for the current query.
            # 'schema_info' is also needed by our prompt.
            # History is automatically managed by the memory object.
            chain_input = {"input": query, "schema_info": schema_info_str}
            result_payload = await db_chain.ainvoke(chain_input)
            logger.info(f"Chain result_payload: {result_payload}")
            
            # LLMChain returns a dictionary, the output is usually in the key specified by output_key (default 'text')
            sql_query = result_payload.get('text', '').strip() if isinstance(result_payload, dict) else str(result_payload).strip()
            if not sql_query:
                logger.error(f"Chain returned empty SQL query. Payload: {result_payload}")
                raise HTTPException(status_code=500, detail="Failed to generate SQL query.")
            
            logger.info(f"Generated SQL query: {sql_query}")

            output = {"sql": sql_query}

            if response_type in [ResponseType.DATA.value, ResponseType.NATURAL_LANGUAGE.value, ResponseType.ALL.value]:
                logger.info(f"Executing SQL query: {sql_query}")
                data = await self.run_query_with_retry(sql_query)
                output["data"] = data
                logger.info(f"Query executed, {len(data)} rows returned.")

                if response_type in [ResponseType.NATURAL_LANGUAGE.value, ResponseType.ALL.value]:
                    logger.info("Generating natural language explanation.")
                    explanation = await self.generate_explanation(sql_query, data)
                    output["explanation"] = explanation
                    logger.info("Natural language explanation generated.")
            
            final_result_key = "result"
            if response_type == ResponseType.SQL.value:
                output[final_result_key] = output["sql"]
            elif response_type == ResponseType.DATA.value:
                output[final_result_key] = output.get("data")
            elif response_type == ResponseType.NATURAL_LANGUAGE.value:
                output[final_result_key] = output.get("explanation")
            else: # ALL
                output[final_result_key] = {k: v for k, v in output.items() if v is not None and k != final_result_key}
            
            output["response_type"] = response_type
            return output

        except HTTPException: # Re-raise HTTPException to be caught by FastAPI
            raise
        except (SQLAlchemyError, ProgrammingError, OperationalError) as db_err:
            logger.error(f"Database error: {db_err}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail={"error": "Database Error", "message": str(db_err)})
        except (RateLimitError, APIError) as openai_err:
            logger.error(f"OpenAI API error: {openai_err}\n{traceback.format_exc()}")
            raise HTTPException(status_code=503, detail={"error": "OpenAI API Error", "message": str(openai_err)})
        except Exception as e:
            logger.error(f"Unexpected error processing query: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail={"error": "Internal Server Error", "message": str(e)})

# Initialize LangChainMySQL instance
langchain_mysql = LangChainMySQL()

async def get_langchain_mysql() -> LangChainMySQL:
    """Dependency function to get LangChainMySQL instance."""
    return langchain_mysql

# Initialize FastAPI application
app = FastAPI(
    title="LangChain MySQL API",
    description="API for interacting with MySQL using LangChain",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get LangChainMySQL instance
async def get_langchain_mysql_instance() -> LangChainMySQL:
    # This could be enhanced to manage a pool or a singleton instance if needed
    return LangChainMySQL()

@app.post("/query", response_model=QueryResponse)
async def process_query_endpoint(
    query_request: QueryRequest, 
    lc_mysql: LangChainMySQL = Depends(get_langchain_mysql_instance)
):
    # Use a default session_id if not provided, or generate a unique one for true session management
    session_id_to_use = query_request.session_id or DEFAULT_SESSION_ID 
    if not query_request.session_id:
        logger.info(f"No session_id provided, using default: {DEFAULT_SESSION_ID}")

    return await lc_mysql.process_query(
        query=query_request.query, 
        session_id=session_id_to_use,
        prompt_type=query_request.prompt_type, 
        response_type=query_request.response_type.value # Ensure using enum value
    )

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

# Main entry point for Uvicorn if running directly (though Docker uses a command)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
