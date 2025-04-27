import pytest
from unittest.mock import AsyncMock
from fastapi import HTTPException

@pytest.fixture
def mock_langchain_mysql():
    """Mock LangChainMySQL instance for testing."""
    mock = AsyncMock()
    mock.process_query = AsyncMock(return_value="Query result")
    return mock

@pytest.mark.asyncio
async def test_process_query_success(mock_langchain_mysql):
    """Test successful query processing."""
    result = await mock_langchain_mysql.process_query("test query")
    assert result == "Query result"
    mock_langchain_mysql.process_query.assert_called_once_with("test query")

@pytest.mark.asyncio
async def test_process_query_error(mock_langchain_mysql):
    """Test error handling during query processing."""
    mock_langchain_mysql.process_query.side_effect = Exception("Database error")
    with pytest.raises(Exception) as exc_info:
        await mock_langchain_mysql.process_query("test query")
    assert str(exc_info.value) == "Database error" 
