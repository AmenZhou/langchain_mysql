import pytest
import pytest_asyncio
import os
from dotenv import load_dotenv
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from langchain.schema import Document
import numpy as np
from openai import OpenAIError, RateLimitError, APIError
import httpx
from sqlalchemy.engine import Engine
from langchain_community.utilities import SQLDatabase

pytest_plugins = ["pytest_asyncio"]

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Load environment variables from .env file
load_dotenv()

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Set test-specific environment variables
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    os.environ["DATABASE_URL"] = "mysql://test:test@localhost/test_db"
    
    yield
    
    # Clean up after tests
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    if "DATABASE_URL" in os.environ:
        del os.environ["DATABASE_URL"]

@pytest.fixture
def mock_db_engine(monkeypatch):
    """Mock SQLAlchemy engine."""
    mock_engine = MagicMock(spec=Engine)
    mock_engine.connect.return_value.__enter__.return_value = MagicMock()
    mock_engine.dialect.name = "mysql"
    
    def mock_get_engine():
        return mock_engine
    
    # Patch the database functions
    monkeypatch.setattr("backend.database._get_engine", mock_get_engine)
    monkeypatch.setattr("backend.database.get_db_engine", mock_get_engine)
    
    return mock_engine

@pytest.fixture
def mock_db_session(monkeypatch):
    """Mock SQLAlchemy session."""
    mock_session = MagicMock()
    mock_session_instance = MagicMock()
    mock_session.return_value = mock_session_instance
    
    def mock_get_db():
        return mock_session_instance
    
    monkeypatch.setattr("backend.database.get_db", mock_get_db)
    return mock_session_instance

@pytest.fixture
def mock_langchain_db(monkeypatch):
    """Mock LangChain SQLDatabase."""
    mock_db = MagicMock(spec=SQLDatabase)
    mock_db.dialect = "mysql"
    mock_db.get_table_info.return_value = ""
    
    def mock_get_langchain_db():
        return mock_db
    
    monkeypatch.setattr("backend.database.get_langchain_db", mock_get_langchain_db)
    return mock_db

@pytest_asyncio.fixture
async def mock_openai_embeddings():
    """Mock OpenAI embeddings."""
    mock = AsyncMock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock

@pytest_asyncio.fixture
async def mock_openai_client():
    """Mock OpenAI client with error handling."""
    mock = MagicMock()
    
    # Mock successful response
    mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Mocked response"))]
    )

    # Mock rate limit error
    mock_rate_limit_response = httpx.Response(
        status_code=429,
        content=b'{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}',
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )
    mock.rate_limit_error = RateLimitError(
        message="Rate limit exceeded",
        response=mock_rate_limit_response,
        body={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}
    )

    # Mock API error
    mock_api_error_response = httpx.Response(
        status_code=500,
        content=b'{"error": {"message": "Internal server error", "type": "api_error"}}',
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )
    mock.api_error = APIError(
        message="Internal server error",
        response=mock_api_error_response,
        body={"error": {"message": "Internal server error", "type": "api_error"}}
    )

    return mock

@pytest.fixture
def mock_openai():
    """Mock OpenAI client and embeddings."""
    with patch("openai.OpenAI") as mock_client, \
         patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings:
        # Mock OpenAI client
        mock_client.return_value.embeddings.create.return_value = {
            "data": [{"embedding": [0.1] * 1536}]
        }
        
        # Mock embeddings
        mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 1536]
        mock_embeddings.return_value.embed_query.return_value = [0.1] * 1536
        
        yield mock_client, mock_embeddings 
