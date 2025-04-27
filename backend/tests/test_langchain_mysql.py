import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, OperationalError
from openai import RateLimitError, APIError
from fastapi import HTTPException
from sqlalchemy import text
import httpx

from backend.langchain_mysql import LangChainMySQL
from backend.schema_vectorizer import SchemaVectorizer
from backend.exceptions import DatabaseError

# Mock OpenAI API responses
mock_response = httpx.Response(status_code=429, json={"error": {"message": "Rate limit exceeded"}})
mock_request = httpx.Request(method="POST", url="https://api.openai.com/v1/chat/completions")

@pytest.fixture
def mock_db_engine():
    """Mock database engine for testing."""
    engine = MagicMock()
    connection = MagicMock()
    cursor = MagicMock()
    
    # Setup the connection context manager
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = None
    engine.connect.return_value = connection
    
    # Setup cursor methods
    connection.execute.return_value = cursor
    cursor.fetchall.return_value = [("test_data",)]
    
    return engine

@pytest.fixture
def mock_schema_vectorizer():
    """Mock schema vectorizer for testing."""
    vectorizer = MagicMock(spec=SchemaVectorizer)
    vectorizer.preload_schema_to_vectordb = AsyncMock()
    vectorizer.get_relevant_schema = AsyncMock(return_value="test schema")
    return vectorizer

@pytest.fixture
def mock_refine_prompt():
    """Mock the refine prompt function."""
    with patch('backend.langchain_mysql.refine_prompt_with_ai') as mock:
        mock.return_value = "SELECT * FROM users"
        yield mock

@pytest_asyncio.fixture
async def langchain_mysql(mock_db_engine, mock_schema_vectorizer):
    """Create a LangChainMySQL instance with mocked dependencies."""
    with patch('backend.langchain_mysql.get_db_engine', return_value=mock_db_engine), \
         patch('backend.langchain_mysql.SchemaVectorizer', return_value=mock_schema_vectorizer):
        instance = LangChainMySQL()
        await instance.initialize()
        return instance

@pytest.mark.asyncio
async def test_initialization(langchain_mysql, mock_schema_vectorizer):
    """Test successful initialization of LangChainMySQL."""
    assert langchain_mysql.engine is not None
    assert langchain_mysql.schema_vectorizer is not None
    mock_schema_vectorizer.preload_schema_to_vectordb.assert_called_once()

@pytest.mark.asyncio
async def test_run_query_with_retry_success(langchain_mysql, mock_db_engine):
    """Test successful query execution with retry."""
    # Setup mock return value
    mock_db_engine.connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [("test_data",)]
    
    result = await langchain_mysql.run_query_with_retry("SELECT * FROM users")
    
    # Verify the query was executed
    mock_db_engine.connect.assert_called_once()
    mock_db_engine.connect.return_value.__enter__.return_value.execute.assert_called_once_with(text("SELECT * FROM users"))
    assert result == "[('test_data',)]"

@pytest.mark.asyncio
async def test_run_query_with_retry_failure_and_retry(langchain_mysql, mock_db_engine):
    """Test query execution with failure and retry."""
    # Setup mock to fail twice then succeed
    mock_execute = mock_db_engine.connect.return_value.__enter__.return_value.execute
    mock_execute.side_effect = [
        SQLAlchemyError("Connection failed"),
        SQLAlchemyError("Connection failed"),
        MagicMock(fetchall=lambda: [("success",)])
    ]
    
    result = await langchain_mysql.run_query_with_retry("SELECT * FROM users", max_retries=3)
    
    assert mock_execute.call_count == 3
    assert result == "[('success',)]"

@pytest.mark.asyncio
async def test_run_query_with_retry_max_retries_exceeded(langchain_mysql, mock_db_engine):
    """Test query execution failing after max retries."""
    # Setup mock to always fail
    mock_db_engine.connect.return_value.__enter__.return_value.execute.side_effect = SQLAlchemyError("Connection failed")
    
    with pytest.raises(SQLAlchemyError) as exc_info:
        await langchain_mysql.run_query_with_retry("SELECT * FROM users", max_retries=2)
    
    assert "Connection failed" in str(exc_info.value)
    assert mock_db_engine.connect.return_value.__enter__.return_value.execute.call_count == 2

@pytest.mark.asyncio
async def test_process_query_success(langchain_mysql, mock_refine_prompt):
    """Test successful query processing."""
    mock_schema = "users(id, name, email)"
    langchain_mysql.schema_vectorizer.get_relevant_schema.return_value = mock_schema
    mock_refine_prompt.return_value = "SELECT * FROM users"
    result = await langchain_mysql.process_query("Show me all users")
    assert result == {"result": "SELECT * FROM users"}
    mock_refine_prompt.assert_called_once_with({"query": "Show me all users", "schema_info": mock_schema})

@pytest.mark.asyncio
async def test_process_query_with_schema(langchain_mysql, mock_refine_prompt, mock_schema_vectorizer):
    """Test query processing with schema information."""
    mock_schema = "users(id, name, email)"
    mock_schema_vectorizer.get_relevant_schema.return_value = mock_schema
    result = await langchain_mysql.process_query("Show me all users", prompt_type="table")
    assert result == {"result": "SELECT * FROM users"}
    mock_refine_prompt.assert_called_once_with({"query": "Show me all users", "schema_info": mock_schema})

@pytest.mark.asyncio
async def test_process_query_empty_query(langchain_mysql):
    """Test handling of empty query."""
    with pytest.raises(HTTPException) as exc_info:
        await langchain_mysql.process_query("")
    assert exc_info.value.status_code == 422
    assert "Query cannot be empty" in exc_info.value.detail

@pytest.mark.asyncio
async def test_process_query_database_error(langchain_mysql, mock_refine_prompt):
    """Test handling of database error."""
    mock_refine_prompt.side_effect = DatabaseError("Database connection failed")
    with pytest.raises(HTTPException) as exc_info:
        await langchain_mysql.process_query("Show me all users")
    assert exc_info.value.status_code == 500
    assert "Database error" in exc_info.value.detail

@pytest.mark.asyncio
async def test_process_query_openai_rate_limit(langchain_mysql, mock_refine_prompt):
    """Test handling of OpenAI rate limit error."""
    mock_refine_prompt.side_effect = RateLimitError(message="Rate limit exceeded", response=mock_response, body={"error": {"message": "Rate limit exceeded"}})
    with pytest.raises(HTTPException) as exc_info:
        await langchain_mysql.process_query("Show me all users")
    assert exc_info.value.status_code == 429
    assert "rate limit" in exc_info.value.detail.lower()

@pytest.mark.asyncio
async def test_process_query_openai_api_error(langchain_mysql, mock_refine_prompt):
    """Test handling of OpenAI API error."""
    mock_refine_prompt.side_effect = APIError(message="API error", request=mock_request)
    with pytest.raises(HTTPException) as exc_info:
        await langchain_mysql.process_query("Show me all users")
    assert exc_info.value.status_code == 500
    assert "OpenAI API error" in exc_info.value.detail
