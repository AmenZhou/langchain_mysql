import pytest
from unittest.mock import AsyncMock, patch
from langchain_mysql import LangChainMySQL

@pytest.fixture
def mock_db_engine():
    """Mock database engine."""
    mock = AsyncMock()
    mock.connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [{"id": 1}]
    return mock

@pytest.fixture
def mock_schema_vectorizer():
    """Mock schema vectorizer."""
    mock = AsyncMock()
    mock.extract_table_schema = AsyncMock(return_value={"users": {"columns": ["id"]}})
    mock.initialize_vector_store = AsyncMock()
    return mock

@pytest.mark.asyncio
async def test_initialization(mock_db_engine, mock_schema_vectorizer):
    """Test that LangChainMySQL can be initialized."""
    with patch('langchain_mysql.create_engine', return_value=mock_db_engine), \
         patch('langchain_mysql.SchemaVectorizer', return_value=mock_schema_vectorizer):
        instance = LangChainMySQL()
        assert instance.schema_vectorizer is not None
        assert instance.engine is not None
        assert instance.llm is not None
