import sys
import os

# Explicitly add the project root to the Python path
sys.path.insert(0, '/app')

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.exc import ProgrammingError, OperationalError
from openai import APIError, RateLimitError
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from backend.schema_vectorizer import SchemaVectorizer
from backend.vector_store import VectorStoreManager
from langchain.schema import Document
import logging
from backend.db_utils import get_database_url

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["DATABASE_URL"] = get_database_url()
    os.environ["OPENAI_API_KEY"] = "test_key"

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store manager."""
    mock = AsyncMock(spec=VectorStoreManager)
    mock.initialize_schema_store = AsyncMock(return_value=None)
    mock.add_documents = AsyncMock(return_value=None)
    mock.similarity_search = AsyncMock(return_value=[Document(page_content="test content", metadata={"table": "test_table"})])
    mock.query_schema = AsyncMock(return_value=[Document(page_content="test content", metadata={"table": "test_table"})])
    mock.query_prompts = AsyncMock(return_value=[Document(page_content="test prompt", metadata={"type": "refine"})])
    return mock

@pytest.fixture
def mock_schema_extractor():
    """Create a mock schema extractor."""
    mock = AsyncMock()
    mock.extract_table_schema = AsyncMock(return_value={"test_table": {"columns": ["id", "name"]}})
    mock.create_schema_documents = AsyncMock(return_value=[Document(page_content="test schema", metadata={"table": "test_table"})])
    return mock

@pytest.fixture
def mock_schema_vectorizer(mock_vector_store, mock_schema_extractor):
    """Create a mock schema vectorizer."""
    mock = AsyncMock(spec=SchemaVectorizer)
    mock.vector_store_manager = mock_vector_store
    mock.schema_extractor = mock_schema_extractor
    mock.preload_schema_to_vectordb = AsyncMock(return_value=None)
    mock.get_relevant_prompt = AsyncMock(return_value="test prompt")
    mock.get_relevant_schema = AsyncMock(return_value={"test_table": {"columns": ["id", "name"]}})
    mock.initialize_vector_store = AsyncMock(return_value=None)
    mock.extract_table_schema = AsyncMock(return_value={"test_table": {"columns": ["id", "name"]}})
    return mock

@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model."""
    mock = AsyncMock()
    mock.embed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return mock

@pytest.fixture
def mock_faiss():
    """Create a mock FAISS vector store."""
    mock = AsyncMock()
    mock.from_documents = AsyncMock(return_value=mock)
    mock.similarity_search = AsyncMock(return_value=[Document(page_content="test content", metadata={"table": "test_table"})])
    return mock

@pytest.fixture
def mock_sql_database():
    """Mock SQLDatabase with predefined table info."""
    mock = MagicMock(spec=SQLDatabase)
    mock.get_table_info.return_value = """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name VARCHAR(255),
        email VARCHAR(255),
        created_at TIMESTAMP
    );
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        amount DECIMAL(10,2),
        status VARCHAR(50),
        created_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """
    mock.dialect = "mysql"
    return mock

@pytest.fixture
def mock_db_engine():
    """Mock database engine with predefined responses."""
    mock = MagicMock()
    mock.execute.return_value = [
        {"id": 1, "name": "John Doe", "email": "john@example.com"},
        {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
    ]
    return mock

@pytest.fixture
def mock_programming_error():
    return ProgrammingError("statement", "params", "orig")

@pytest.fixture
def mock_operational_error():
    return OperationalError("statement", "params", "orig")

@pytest.fixture
def mock_rate_limit_error():
    """Mock OpenAI rate limit error."""
    error = RateLimitError()
    error.message = "Rate limit exceeded"
    error.status_code = 429
    return error

@pytest.fixture
def mock_api_error():
    """Mock OpenAI API error."""
    error = APIError()
    error.message = "API error occurred"
    error.status_code = 500
    return error
