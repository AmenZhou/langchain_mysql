import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from sqlalchemy import MetaData, inspect
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from langchain_core.outputs import ChatResult
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_community.utilities import SQLDatabase
from src.backend.utils import sanitize_sql_response
from src.backend.prompts import get_sanitize_prompt
from langchain_community.chat_models import ChatOpenAI
from fastapi import HTTPException
from sqlalchemy.exc import ProgrammingError, OperationalError
from openai import RateLimitError, APIError, OpenAIError
from openai import AsyncOpenAI

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ..utils.sql_utils import (
    sanitize_sql_response,
    extract_table_name,
    get_openai_client,
)

@pytest.fixture(autouse=True)
def mock_env():
    """Set up environment variables for testing."""
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'DATABASE_URL': 'mysql://test:test@localhost/test'
    }):
        yield

@pytest.mark.asyncio
async def test_get_openai_client_success():
    """Test successful OpenAI client initialization."""
    with patch('backend.utils.sql_utils.AsyncOpenAI') as mock:
        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock.return_value = mock_client
        client = get_openai_client()
        assert isinstance(client, AsyncOpenAI)

@pytest.mark.asyncio
async def test_sanitize_sql_response_success():
    """Test successful SQL response sanitization."""
    response = "SELECT * FROM users;"
    result = await sanitize_sql_response(response)
    assert result == "SELECT * FROM users"

@pytest.mark.asyncio
async def test_extract_table_name_valid():
    """Test extracting table name from valid SQL query."""
    query = "SELECT * FROM users WHERE id = 1"
    result = extract_table_name(query)
    assert result == "users"

if __name__ == "__main__":
    pytest.main([__file__])
