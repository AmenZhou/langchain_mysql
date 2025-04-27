import pytest
import pytest_asyncio
import httpx
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, OperationalError
from openai import RateLimitError, APIError, OpenAIError
from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from backend.server import app, get_db_engine, get_vectorizer
from backend.schema_vectorizer import SchemaVectorizer
from contextlib import contextmanager

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from backend.main import app
from backend.utils import sanitize_sql_response
from backend.exceptions import DatabaseError, OpenAIRateLimitError, OpenAIAPIError
from backend.prompts import get_sanitize_prompt
from backend.langchain_mysql import LangChainMySQL

# Mock OpenAI API responses
mock_rate_limit_response = httpx.Response(
    status_code=429,
    content=b'{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}',
    request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
)

mock_api_error_response = httpx.Response(
    status_code=500,
    content=b'{"error": {"message": "Internal server error", "type": "api_error"}}',
    request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
)

@pytest_asyncio.fixture
async def async_client():
    """Create async test client."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client

@pytest.fixture
def mock_db_engine():
    """Mock database engine for testing."""
    engine = MagicMock()
    connection = MagicMock()
    connection.__enter__.return_value = connection
    engine.connect.return_value = connection
    with patch('backend.database.get_db_engine', return_value=engine):
        yield engine

@pytest.fixture
def mock_schema_vectorizer():
    with patch('backend.schema_vectorizer.SchemaVectorizer') as mock:
        mock_instance = mock.return_value
        mock_instance.get_relevant_prompt.return_value = "Test prompt"
        yield mock_instance

@pytest.fixture
def mock_chat_model():
    with patch('langchain_community.chat_models.ChatOpenAI') as mock:
        mock_instance = mock.return_value
        mock_instance.agenerate = AsyncMock(return_value="Test response")
        yield mock_instance

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_engine():
    with patch('backend.server.get_db_engine') as mock:
        engine = Mock()
        connection = Mock()
        connection.__enter__.return_value = connection
        engine.connect.return_value = connection
        mock.return_value = engine
        yield engine

@pytest.fixture
def mock_vectorizer():
    with patch('backend.server.get_vectorizer') as mock:
        vectorizer = Mock()
        vectorizer.get_relevant_schema = AsyncMock()
        vectorizer.get_relevant_prompt = AsyncMock()
        mock.return_value = vectorizer
        yield vectorizer

@pytest.mark.asyncio
async def test_health_check_success(client):
    """Test successful health check."""
    with patch("backend.server.get_db_engine") as mock_engine:
        mock_conn = Mock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_health_check_failure(client):
    """Test health check failure."""
    with patch("backend.server.get_db_engine") as mock_engine:
        mock_engine.return_value.connect.side_effect = Exception("Connection failed")
        response = client.get("/health")
        assert response.status_code == 500
        assert "Health check failed" in response.json()["detail"]

@pytest.mark.asyncio
async def test_nonexistent_table_error(client):
    """Test handling of nonexistent table error."""
    with patch("backend.server.get_langchain_mysql") as mock_langchain:
        mock_langchain.return_value.process_query.side_effect = ProgrammingError(
            "no such table: users", None, None
        )
        response = client.post("/query", json={"query": "select * from users"})
        assert response.status_code == 422
        assert "Table does not exist" in response.json()["detail"]

@pytest.mark.asyncio
async def test_sql_syntax_error(client):
    """Test handling of SQL syntax error."""
    with patch("backend.server.get_langchain_mysql") as mock_langchain:
        mock_langchain.return_value.process_query.side_effect = ProgrammingError(
            "syntax error near 'from'", None, None
        )
        response = client.post("/query", json={"query": "select from users"})
        assert response.status_code == 422
        assert "Invalid SQL syntax" in response.json()["detail"]

@pytest.mark.asyncio
async def test_permission_error(client):
    """Test handling of permission error."""
    with patch("backend.server.get_langchain_mysql") as mock_langchain:
        mock_langchain.return_value.process_query.side_effect = OperationalError(
            "permission denied for table users", None, None
        )
        response = client.post("/query", json={"query": "select * from users"})
        assert response.status_code == 403
        assert "Permission denied" in response.json()["detail"]

@pytest.mark.asyncio
async def test_empty_query(client):
    """Test handling of empty query."""
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422
    assert "Query cannot be empty" in response.json()["detail"]

@pytest.mark.asyncio
async def test_missing_query(client):
    """Test handling of missing query field."""
    response = client.post("/query", json={})
    assert response.status_code == 422
    assert "Field required" in response.json()["detail"][0]["msg"]

@pytest.mark.asyncio
async def test_openai_rate_limit_error(mock_openai_client):
    """Test handling of OpenAI rate limit error."""
    # Simulate rate limit error
    mock_openai_client.chat.completions.create.side_effect = mock_openai_client.rate_limit_error
    
    with pytest.raises(RateLimitError) as exc_info:
        await handle_openai_error(mock_openai_client.chat.completions.create())
    
    assert "Rate limit exceeded" in str(exc_info.value)
    assert exc_info.value.response.status_code == 429

@pytest.mark.asyncio
async def test_openai_api_error(mock_openai_client):
    """Test handling of OpenAI API error."""
    # Simulate API error
    mock_openai_client.chat.completions.create.side_effect = mock_openai_client.api_error
    
    with pytest.raises(APIError) as exc_info:
        await handle_openai_error(mock_openai_client.chat.completions.create())
    
    assert "Internal server error" in str(exc_info.value)
    assert exc_info.value.response.status_code == 500

@pytest.mark.asyncio
async def test_openai_generic_error(mock_openai_client):
    """Test handling of generic OpenAI error."""
    # Simulate generic OpenAI error
    mock_openai_client.chat.completions.create.side_effect = OpenAIError("Unknown error")
    
    with pytest.raises(OpenAIError) as exc_info:
        await handle_openai_error(mock_openai_client.chat.completions.create())
    
    assert "Unknown error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_unexpected_error(client):
    """Test handling of unexpected error."""
    with patch("backend.server.get_langchain_mysql") as mock_langchain:
        mock_langchain.return_value.process_query.side_effect = Exception("Unexpected error")
        response = client.post("/query", json={"query": "select * from users"})
        assert response.status_code == 500
        assert "Unexpected error" in response.json()["detail"]

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
