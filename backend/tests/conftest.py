import pytest
import pytest_asyncio
import os
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.exc import ProgrammingError, OperationalError
from openai import APIError, RateLimitError
from fastapi.testclient import TestClient
from httpx import AsyncClient, Request, Response
from langchain_community.utilities import SQLDatabase
from src.backend.main import app

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["DATABASE_URL"] = "mysql://test_user:testpassword@localhost:3307/test_db"
    os.environ["OPENAI_API_KEY"] = "test_key"

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value={"choices": [{"message": {"content": "Mocked response"}}]})
    return client

@pytest.fixture
def mock_schema_vectorizer():
    """Mock schema vectorizer."""
    mock = AsyncMock()
    mock.extract_table_schema = AsyncMock(return_value={"users": {"columns": [{"name": "id", "type": "INTEGER"}]}})
    mock.get_relevant_prompt = AsyncMock(return_value="SELECT * FROM users")
    mock.get_relevant_schema = AsyncMock(return_value="users table schema")
    return mock

@pytest.fixture
def mock_sql_database():
    """Mock SQL database."""
    mock = MagicMock(spec=SQLDatabase)
    mock.get_table_info = MagicMock(return_value="table info")
    mock.dialect = MagicMock()
    mock.dialect.name = "mysql"
    return mock

@pytest.fixture
def mock_programming_error():
    return ProgrammingError("statement", "params", "orig")

@pytest.fixture
def mock_operational_error():
    return OperationalError("statement", "params", "orig")

@pytest.fixture
def mock_rate_limit_error():
    request = Request("POST", "https://api.openai.com/v1/chat/completions")
    response = Response(429, request=request)
    return RateLimitError(message="Rate limit exceeded", response=response, body={"error": {"message": "Rate limit exceeded"}})

@pytest.fixture
def mock_api_error():
    request = Request("POST", "https://api.openai.com/v1/chat/completions")
    response = Response(500, request=request)
    return APIError(message="Internal server error", response=response, body={"error": {"message": "Internal server error"}})

@pytest.fixture
def test_app():
    """Create a test FastAPI application."""
    return app

@pytest.fixture
async def test_client(test_app):
    """Create an async test client."""
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client
