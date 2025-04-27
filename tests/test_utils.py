import pytest
from unittest.mock import AsyncMock, MagicMock
from src.backend.utils.sql_utils import (
    get_openai_client,
    generate_column_query,
    generate_table_query,
    refine_prompt_with_ai,
    sanitize_sql_response,
    extract_table_name
)
from openai import APIError

@pytest.mark.asyncio
async def test_get_openai_client_success():
    """Test successful OpenAI client initialization."""
    client = get_openai_client()
    assert client is not None

@pytest.mark.asyncio
async def test_generate_column_query_success(mock_openai_client):
    """Test successful column query generation."""
    result = await generate_column_query("users", mock_openai_client)
    assert "SHOW COLUMNS FROM users" in result

@pytest.mark.asyncio
async def test_generate_table_query_success(mock_openai_client):
    """Test successful table query generation."""
    result = await generate_table_query(mock_openai_client)
    assert "SHOW TABLES" in result

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_success(mock_openai_client):
    """Test successful prompt refinement."""
    result = await refine_prompt_with_ai("Show me all users", mock_openai_client)
    assert "SELECT" in result
    assert isinstance(result, str)

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_error(mock_api_error):
    """Test prompt refinement with API error."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = mock_api_error
    with pytest.raises(APIError):
        await refine_prompt_with_ai("Show me all users", mock_client)

def test_sanitize_sql_response_success():
    """Test successful SQL response sanitization."""
    result = sanitize_sql_response("SELECT * FROM users")
    assert result == "SELECT * FROM users"

def test_sanitize_sql_response_empty():
    """Test empty SQL response sanitization."""
    with pytest.raises(ValueError, match="Empty SQL response"):
        sanitize_sql_response("")

def test_extract_table_name_valid():
    """Test valid table name extraction."""
    queries = [
        "SELECT * FROM users",
        "INSERT INTO users (name) VALUES ('test')",
        "UPDATE users SET name = 'test'",
        "DELETE FROM users WHERE id = 1"
    ]
    for query in queries:
        result = extract_table_name(query)
        assert result == "users"

def test_extract_table_name_invalid():
    """Test invalid table name extraction."""
    invalid_queries = [
        "",
        "SHOW TABLES",
        "SELECT 1",
        "invalid sql"
    ]
    for query in invalid_queries:
        with pytest.raises(ValueError, match="No table name found"):
            extract_table_name(query)

class TestSchemaExtraction:
    @pytest.mark.asyncio
    async def test_schema_extraction_success(self, mock_schema_vectorizer):
        """Test successful schema extraction."""
        result = mock_schema_vectorizer.extract_table_schema("users")
        assert "columns" in result
        assert len(result["columns"]) > 0

    def test_schema_extraction_error(self, mock_schema_extractor_error):
        """Test schema extraction error handling."""
        with pytest.raises(Exception):
            mock_schema_extractor_error.extract_table_schema("nonexistent_table") 
