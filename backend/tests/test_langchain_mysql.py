import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, OperationalError
from openai import RateLimitError, APIError
from fastapi import HTTPException
from sqlalchemy import text

from backend.langchain_mysql import LangChainMySQL
from backend.schema_vectorizer import SchemaVectorizer
from backend.exceptions import DatabaseError

@pytest.fixture
def mock_db_engine():
    """Mock database engine for testing."""
    engine = MagicMock()
    connection = MagicMock()
    cursor = MagicMock()
    
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = None
    engine.connect.return_value = connection
    connection.execute.return_value = cursor
    cursor.fetchall.return_value = [("test_data",)]
    return engine

@pytest.fixture
def mock_schema_vectorizer():
    """Mock SchemaVectorizer with proper async methods."""
    mock = MagicMock(spec=SchemaVectorizer)
    mock.get_relevant_prompt = AsyncMock(return_value="SELECT * FROM table")
    mock.get_relevant_schema = AsyncMock(return_value="mocked schema")
    mock.preload_schema_to_vectordb = AsyncMock()
    mock.initialize_vector_store = AsyncMock()
    mock.extract_table_schema = AsyncMock(return_value={"users": {"columns": [{"name": "id", "type": "INTEGER"}]}})
    return mock

@pytest.fixture
def mock_chain():
    """Mock chain for testing."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value={"result": "SQL query result"})
    return mock

@pytest_asyncio.fixture
async def langchain_mysql(mock_db_engine, mock_schema_vectorizer):
    """Create a LangChainMySQL instance with mocked dependencies."""
    with patch('backend.langchain_mysql.get_db_engine', return_value=mock_db_engine), \
         patch('backend.langchain_mysql.SchemaVectorizer', return_value=mock_schema_vectorizer):
        instance = LangChainMySQL()
        await instance.initialize()
        return instance

@pytest.mark.asyncio
async def test_initialization(mock_schema_vectorizer):
    """Test LangChainMySQL initialization"""
    with patch('backend.langchain_mysql.SchemaVectorizer', return_value=mock_schema_vectorizer):
        langchain_mysql = LangChainMySQL()
        await langchain_mysql.initialize()
        assert langchain_mysql.schema_vectorizer == mock_schema_vectorizer
        mock_schema_vectorizer.extract_table_schema.assert_called_once()
        mock_schema_vectorizer.initialize_vector_store.assert_called_once()

@pytest.mark.asyncio
async def test_run_query_with_retry_success(langchain_mysql):
    """Test successful query execution with retry."""
    result = await langchain_mysql.run_query_with_retry("SELECT * FROM users")
    assert result is not None

@pytest.mark.asyncio
async def test_process_query_success(mock_chain, mock_schema_vectorizer):
    """Test successful query processing."""
    with patch('backend.langchain_mysql.create_db_chain_with_schema', return_value=mock_chain), \
         patch('backend.langchain_mysql.SchemaVectorizer', return_value=mock_schema_vectorizer):
        langchain_mysql = LangChainMySQL()
        result = await langchain_mysql.process_query("Show me users", "select")
        assert result == "SQL query result"

@pytest.mark.asyncio
async def test_process_query_empty_query():
    """Test empty query handling."""
    langchain_mysql = LangChainMySQL()
    with pytest.raises(HTTPException) as exc_info:
        await langchain_mysql.process_query("", "select")
    assert exc_info.value.status_code == 422
    assert "Query cannot be empty" in str(exc_info.value.detail)
