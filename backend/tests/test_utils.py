import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy import MetaData

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock the database connection before importing any modules
mock_engine = MagicMock()
mock_db = MagicMock()

# Mock the database module
mock_database = MagicMock()
mock_database.engine = mock_engine
mock_database.db = mock_db

# Mock the entire database module
with patch.dict('sys.modules', {'backend.database': mock_database}):
    from backend.utils import refine_prompt_with_ai, sanitize_sql_response

@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_success():
    mock_schema_vectorizer = AsyncMock()
    mock_schema_vectorizer.get_schema_documents.return_value = ["schema1", "schema2"]
    
    mock_openai_client = AsyncMock()
    mock_openai_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="refined query"))]
    )
    
    with patch("backend.utils.SchemaVectorizer", return_value=mock_schema_vectorizer), \
         patch("backend.utils.AsyncOpenAI", return_value=mock_openai_client):
        result = await refine_prompt_with_ai("test query")
        assert result == "refined query"

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_schema_error():
    mock_schema_vectorizer = AsyncMock()
    mock_schema_vectorizer.get_schema_documents.side_effect = Exception("Schema error")
    
    with patch("backend.utils.SchemaVectorizer", return_value=mock_schema_vectorizer):
        result = await refine_prompt_with_ai("test query")
        assert result is None

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_openai_error():
    mock_schema_vectorizer = AsyncMock()
    mock_schema_vectorizer.get_schema_documents.return_value = ["schema1", "schema2"]
    
    mock_openai_client = AsyncMock()
    mock_openai_client.chat.completions.create.side_effect = Exception("OpenAI error")
    
    with patch("backend.utils.SchemaVectorizer", return_value=mock_schema_vectorizer), \
         patch("backend.utils.AsyncOpenAI", return_value=mock_openai_client):
        result = await refine_prompt_with_ai("test query")
        assert result is None

@pytest.mark.asyncio
async def test_sanitize_sql_response_success():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content="sanitized response")
    
    with patch("backend.utils.ChatOpenAI", return_value=mock_llm):
        result = await sanitize_sql_response("test response")
        assert result == "sanitized response"

@pytest.mark.asyncio
async def test_sanitize_sql_response_error():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = Exception("LLM error")
    
    with patch("backend.utils.ChatOpenAI", return_value=mock_llm):
        result = await sanitize_sql_response("test response")
        assert result == "test response" 
