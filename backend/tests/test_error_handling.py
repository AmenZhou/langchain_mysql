import pytest
import pytest_asyncio
import httpx
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, OperationalError
from openai import RateLimitError, APIError, OpenAIError
from fastapi import HTTPException, status, Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from contextlib import contextmanager
import respx
from httpx import Response

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ..main import app
from ..langchain_mysql import get_langchain_mysql, LangChainMySQL
from ..schema_vectorizer import SchemaVectorizer
from ..utils.error_handling import handle_openai_error
from ..models import QueryRequest
from ..security import limiter
from ..utils import sanitize_sql_response
from ..exceptions import DatabaseError, OpenAIRateLimitError, OpenAIAPIError
from ..prompts import get_sanitize_prompt

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
    with patch('..database.get_db_engine', return_value=engine):
        yield engine

@pytest.fixture
def mock_schema_vectorizer():
    with patch('..schema_vectorizer.SchemaVectorizer') as mock:
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
def mock_langchain_mysql():
    """Mock LangChainMySQL for testing."""
    with patch('..langchain_mysql.LangChainMySQL') as mock:
        mock_instance = mock.return_value
        mock_instance.get_relevant_prompt.return_value = "Test prompt"
        mock_instance.get_relevant_schema.return_value = "mocked schema"
        mock_instance.preload_schema_to_vectordb = AsyncMock()
        mock_instance.initialize_vector_store = AsyncMock()
        yield mock_instance

@pytest.fixture
def client(mock_langchain_mysql):
    """Create a test client with mocked dependencies."""
    # Override the rate limiter key function to always return a test key
    def get_test_key(request: Request):
        return "test_client"
    limiter.key_func = get_test_key
    
    with patch('..server.get_langchain_mysql', return_value=mock_langchain_mysql):
        return TestClient(app)

@pytest.fixture
def mock_engine():
    with patch('..server.get_db_engine') as mock:
        engine = Mock()
        connection = Mock()
        connection.__enter__.return_value = connection
        engine.connect.return_value = connection
        mock.return_value = engine
        yield engine

@pytest.fixture
def mock_vectorizer():
    """Mock SchemaVectorizer with proper async methods."""
    mock = AsyncMock(spec=SchemaVectorizer)
    mock.get_relevant_prompt = AsyncMock(return_value="SELECT * FROM table")
    mock.get_relevant_schema = AsyncMock(return_value="mocked schema")
    mock.preload_schema_to_vectordb = AsyncMock()
    mock.initialize_vector_store = AsyncMock()
    return mock

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('openai.OpenAI') as mock:
        mock_instance = mock.return_value
        mock_instance.chat.completions.create = AsyncMock()
        yield mock_instance

@pytest.fixture
def mock_sql_chain():
    """Mock SQL chain for testing."""
    with patch('langchain.chains.sql_database.query.create_sql_query_chain') as mock:
        mock_instance = mock.return_value
        mock_instance.invoke = AsyncMock()
        yield mock_instance

@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key for testing."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key'}):
        yield

@pytest.mark.asyncio
async def test_health_check_success(client):
    """Test successful health check."""
    with patch("..server.get_db_engine") as mock_engine:
        mock_conn = Mock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_health_check_failure(mock_db_engine):
    """Test health check failure."""
    async def mock_connect():
        raise OperationalError("statement", "params", "Database connection failed")
    mock_db_engine.connect = mock_connect
    with pytest.raises(HTTPException) as exc_info:
        await handle_openai_error(mock_connect())
    assert exc_info.value.status_code == 500
    assert "Database connection error" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_nonexistent_table_error(mock_sql_chain):
    """Test nonexistent table error."""
    mock_sql_chain.invoke.side_effect = ProgrammingError("relation 'nonexistent_table' does not exist", "params", "orig")
    with pytest.raises(HTTPException) as exc_info:
        await handle_openai_error(mock_sql_chain.invoke({"query": "SELECT * FROM nonexistent_table"}))
    assert exc_info.value.status_code == 422
    assert "Table does not exist" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_sql_syntax_error(mock_sql_chain):
    """Test SQL syntax error."""
    mock_sql_chain.invoke.side_effect = ProgrammingError("syntax error", "params", "orig")
    with pytest.raises(HTTPException) as exc_info:
        await handle_openai_error(mock_sql_chain.invoke({"query": "SELECT * FROM users WHERE"}))
    assert exc_info.value.status_code == 422
    assert "Invalid SQL syntax" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_permission_error(mock_sql_chain):
    """Test permission error."""
    mock_sql_chain.invoke.side_effect = ProgrammingError("permission denied", "params", "orig")
    with pytest.raises(HTTPException) as exc_info:
        await handle_openai_error(mock_sql_chain.invoke({"query": "DROP TABLE users"}))
    assert exc_info.value.status_code == 403
    assert "Permission denied" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_empty_query():
    """Test empty query error."""
    with pytest.raises(HTTPException) as exc_info:
        await handle_openai_error(None)
    assert exc_info.value.status_code == 422
    assert "Query cannot be empty" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_missing_query(client):
    """Test handling of missing query field."""
    response = client.post("/query", json={})
    assert response.status_code == 422
    assert "Field required" in response.json()["detail"][0]["msg"]

@pytest.mark.asyncio
async def test_openai_rate_limit_error(mock_openai_client):
    """Test OpenAI rate limit error."""
    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_response = httpx.Response(429, request=mock_request)
    error = RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}}
    )
    mock_openai_client.chat.completions.create.side_effect = error
    with pytest.raises(HTTPException) as exc_info:
        await handle_openai_error(mock_openai_client.chat.completions.create())
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_openai_api_error(mock_openai_client):
    """Test OpenAI API error."""
    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = APIError(
        message="API error",
        request=mock_request,
        body={"error": {"message": "API error"}}
    )
    mock_openai_client.chat.completions.create.side_effect = error
    with pytest.raises(HTTPException) as exc_info:
        await handle_openai_error(mock_openai_client.chat.completions.create())
    assert exc_info.value.status_code == 422
    assert "OpenAI API error" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_openai_generic_error(mock_openai_client):
    """Test OpenAI generic error."""
    mock_openai_client.chat.completions.create.side_effect = Exception("Generic error")
    with pytest.raises(HTTPException) as exc_info:
        await handle_openai_error(mock_openai_client.chat.completions.create())
    assert exc_info.value.status_code == 500
    assert "Unexpected error" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_unexpected_error(mock_sql_chain):
    """Test unexpected error."""
    mock_sql_chain.invoke.side_effect = Exception("Unexpected error")
    with pytest.raises(HTTPException) as exc_info:
        await handle_openai_error(mock_sql_chain.invoke({"query": "SELECT * FROM users"}))
    assert exc_info.value.status_code == 500
    assert "Unexpected error" in str(exc_info.value.detail)

@pytest.mark.asyncio
@respx.mock
async def test_no_relevant_schema_error(client, mock_openai_api_key):
    """Test error when no relevant schema information is found for the query."""
    # Create a mock LangChainMySQL instance
    mock_langchain = Mock()
    mock_langchain.process_query = AsyncMock()
    mock_langchain.process_query.side_effect = HTTPException(
        status_code=422,
        detail={
            "error": "Schema Error",
            "details": "No relevant schema information found for the query"
        }
    )

    # Override the dependency
    app.dependency_overrides[get_langchain_mysql] = lambda: mock_langchain

    # Mock OpenAI embeddings
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 1536  # Standard OpenAI embedding size
    
    # Mock ChatOpenAI
    mock_chat = Mock()
    mock_chat.agenerate.return_value = "test response"
    
    # Mock all OpenAI-related dependencies
    with patch('langchain_openai.OpenAIEmbeddings', return_value=mock_embeddings), \
         patch('langchain_openai.ChatOpenAI', return_value=mock_chat), \
         patch('langchain_community.chat_models.ChatOpenAI', return_value=mock_chat), \
         patch('openai.OpenAI') as mock_openai, \
         patch('backend.utils.sql_utils.AsyncOpenAI') as mock_async_openai:
        
        # Set up the OpenAI mock
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = {"data": [{"embedding": [0.1] * 1536}]}
        
        # Set up the AsyncOpenAI mock
        mock_async_openai_instance = mock_async_openai.return_value
        mock_async_openai_instance.embeddings.create.return_value = {"data": [{"embedding": [0.1] * 1536}]}
        
        try:
            # Make a POST request to the query endpoint
            response = client.post("/query", json={"query": "test query"})
            
            # Verify the response
            assert response.status_code == 422
            response_json = response.json()
            assert isinstance(response_json, dict)
            assert "detail" in response_json
            assert isinstance(response_json["detail"], dict)
            assert response_json["detail"]["error"] == "Schema Error"
            assert response_json["detail"]["details"] == "No relevant schema information found for the query"
            
            # Verify the mock was called
            mock_langchain.process_query.assert_called_once()
        finally:
            # Clean up any dependency overrides
            app.dependency_overrides.clear()

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
