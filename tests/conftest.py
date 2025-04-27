import pytest
from unittest.mock import MagicMock, AsyncMock
from openai import OpenAIError, RateLimitError, APIError, APIStatusError
from sqlalchemy.exc import ProgrammingError, OperationalError
import os
from dotenv import load_dotenv
import httpx
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types import CompletionUsage
from fastapi import FastAPI
from httpx import AsyncClient
from src.backend.main import app, router

@pytest.fixture
def mock_rate_limit_error():
    """Mock RateLimitError with proper error format."""
    response = httpx.Response(429, json={
        "error": {
            "message": "Rate limit exceeded. Please try again in 20s.",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded"
        }
    })
    return RateLimitError("Rate limit exceeded", response=response)

@pytest.fixture
def mock_api_error():
    """Mock APIError with proper error format."""
    response = httpx.Response(500, json={
        "error": {
            "message": "Internal server error",
            "type": "api_error",
            "code": "internal_error"
        }
    })
    return APIError("API error occurred", response=response)

@pytest.fixture
def mock_langchain_mysql():
    """Mock LangChainMySQL with proper response format."""
    mock = AsyncMock()
    mock.process_query = AsyncMock()
    mock.process_query.return_value = {
        "result": "Query executed successfully",
        "sql": "SELECT * FROM users",
        "explanation": "Query to fetch all users"
    }
    mock.schema_vectorizer = MagicMock()
    mock.schema_vectorizer.get_relevant_schema = MagicMock(return_value={
        "table_name": "users",
        "columns": [
            {"name": "id", "type": "INTEGER", "description": "Primary key"},
            {"name": "name", "type": "VARCHAR(255)", "description": "User's full name"},
            {"name": "email", "type": "VARCHAR(255)", "description": "User's email address"}
        ]
    })
    return mock

@pytest.fixture
def mock_sql_database():
    """Mock SQLDatabase with proper schema format."""
    mock = MagicMock()
    mock.get_table_info = MagicMock(return_value="table_info")
    mock.dialect = MagicMock()
    mock.dialect.name = "mysql"
    return mock

@pytest.fixture
def mock_schema_extractor():
    """Mock SchemaExtractor with async support."""
    mock = AsyncMock()
    mock.extract_table_schema = AsyncMock()
    mock.extract_table_schema.return_value = {
        "columns": ["id", "name", "email"],
        "description": "User information table"
    }
    return mock

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client with proper response format."""
    mock = AsyncMock()
    mock.chat.completions.create = AsyncMock()
    mock.chat.completions.create.return_value = ChatCompletion(
        id="chatcmpl-123",
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "SELECT * FROM users"
            },
            "finish_reason": "stop"
        }],
        created=1677652288,
        model="gpt-3.5-turbo",
        object="chat.completion",
        usage={
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    )
    return mock

@pytest.fixture
def mock_schema_vectorizer():
    """Mock SchemaVectorizer with proper schema format."""
    mock = MagicMock()
    mock.initialized = True
    mock.extract_table_schema = MagicMock(return_value={
        "table_name": "users",
        "columns": [
            {"name": "id", "type": "INTEGER", "description": "Primary key"},
            {"name": "name", "type": "VARCHAR(255)", "description": "User's full name"},
            {"name": "email", "type": "VARCHAR(255)", "description": "User's email address"}
        ]
    })
    mock.get_table_info = MagicMock(return_value="Table users contains user information")
    mock.get_relevant_schema = MagicMock(return_value={
        "table_name": "users",
        "columns": [
            {"name": "id", "type": "INTEGER", "description": "Primary key"},
            {"name": "name", "type": "VARCHAR(255)", "description": "User's full name"},
            {"name": "email", "type": "VARCHAR(255)", "description": "User's email address"}
        ]
    })
    return mock

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["DATABASE_URL"] = "mysql+pymysql://test_user:testpassword@localhost:3308/test_db"
    os.environ["OPENAI_API_KEY"] = "test_key"
    load_dotenv()

@pytest_asyncio.fixture
async def test_client(mock_langchain_mysql, mock_schema_vectorizer, mock_openai_client):
    """Create an async test client."""
    app.include_router(router)
    
    # Override dependencies
    app.dependency_overrides = {
        "get_langchain_mysql": lambda: mock_langchain_mysql,
        "get_schema_vectorizer": lambda: mock_schema_vectorizer,
        "get_openai_client": lambda: mock_openai_client,
    }
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
        
    # Clear dependency overrides after test
    app.dependency_overrides = {} 
