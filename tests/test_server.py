import pytest
from sqlalchemy.exc import ProgrammingError, OperationalError
from openai.error import RateLimitError, APIError
from fastapi import FastAPI
from httpx import AsyncClient
from src.backend.main import app, router
from src.backend.schema_vectorizer import SchemaVectorizer

@pytest.mark.asyncio
async def test_health_check(test_client):
    """Test health check endpoint."""
    response = await test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_query_endpoint_success(test_client, mock_langchain_mysql):
    """Test successful query processing."""
    response = await test_client.post("/query", json={
        "query": "Show me all users",
        "prompt_type": "select"
    })
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "sql" in data
    assert "explanation" in data

@pytest.mark.asyncio
async def test_empty_query(test_client):
    """Test empty query handling."""
    response = await test_client.post("/query", json={
        "query": "",
        "prompt_type": "select"
    })
    assert response.status_code == 422
    assert "Query cannot be empty" in response.json()["detail"]

@pytest.mark.asyncio
async def test_missing_prompt_type(test_client):
    """Test missing prompt type handling."""
    response = await test_client.post("/query", json={
        "query": "Show me all users"
    })
    assert response.status_code == 422
    assert "field required" in response.json()["detail"][0]["msg"]

@pytest.mark.asyncio
async def test_programming_error(test_client, mock_programming_error):
    """Test SQL programming error handling."""
    response = await test_client.post("/query", json={
        "query": "Show me nonexistent_table",
        "prompt_type": "select"
    })
    assert response.status_code == 422
    assert "nonexistent_table" in response.json()["detail"]

@pytest.mark.asyncio
async def test_operational_error(test_client, mock_operational_error):
    """Test SQL operational error handling."""
    response = await test_client.post("/query", json={
        "query": "DROP TABLE users",
        "prompt_type": "select"
    })
    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]

@pytest.mark.asyncio
async def test_rate_limit_error(test_client, mock_rate_limit_error):
    """Test OpenAI rate limit error handling."""
    response = await test_client.post("/query", json={
        "query": "Show me users",
        "prompt_type": "select"
    })
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]

@pytest.mark.asyncio
async def test_api_error(test_client, mock_api_error):
    """Test OpenAI API error handling."""
    response = await test_client.post("/query", json={
        "query": "Show me users",
        "prompt_type": "select"
    })
    assert response.status_code == 500
    assert "API error" in response.json()["detail"]

@pytest.mark.asyncio
async def test_invalid_prompt_type(test_client):
    """Test invalid prompt type handling."""
    response = await test_client.post("/query", json={
        "query": "Show me users",
        "prompt_type": "invalid"
    })
    assert response.status_code == 422
    assert "Invalid prompt type" in response.json()["detail"]

@pytest.mark.asyncio
async def test_large_query(test_client):
    """Test large query handling."""
    large_query = "x" * 1001  # Exceeds max length
    response = await test_client.post("/query", json={
        "query": large_query,
        "prompt_type": "select"
    })
    assert response.status_code == 422
    assert "Query too long" in response.json()["detail"]

@pytest.mark.asyncio
async def test_programming_error_no_table(client, mock_langchain_mysql):
    """Test handling of ProgrammingError for nonexistent table."""
    error_msg = "Table 'test_db.nonexistent' doesn't exist"
    mock_langchain_mysql.process_query.side_effect = ProgrammingError(error_msg, None, None)
    response = await client.post("/query", json={"query": "Select * from nonexistent", "prompt_type": "sql"})
    assert response.status_code == 422
    assert response.json() == {"error": error_msg}

@pytest.mark.asyncio
async def test_operational_error(client, mock_langchain_mysql):
    """Test handling of database operational errors."""
    error_msg = "Lost connection to MySQL server"
    mock_langchain_mysql.process_query.side_effect = OperationalError(error_msg, None, None)
    response = await client.post("/query", json={"query": "Show me users", "prompt_type": "sql"})
    assert response.status_code == 500
    assert response.json() == {"error": error_msg}

@pytest.mark.asyncio
async def test_rate_limit_error(client, mock_langchain_mysql, mock_rate_limit_error):
    """Test handling of rate limit errors."""
    mock_langchain_mysql.process_query.side_effect = mock_rate_limit_error
    response = await client.post("/query", json={"query": "Show me users", "prompt_type": "sql"})
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["error"]

@pytest.mark.asyncio
async def test_api_error(client, mock_langchain_mysql, mock_api_error):
    """Test handling of general API errors."""
    mock_langchain_mysql.process_query.side_effect = mock_api_error
    response = await client.post("/query", json={"query": "Show me users", "prompt_type": "sql"})
    assert response.status_code == 500
    assert "API error occurred" in response.json()["error"]

@pytest.mark.asyncio
async def test_missing_prompt_type(client):
    """Test validation of missing prompt type."""
    response = await client.post("/query", json={"query": "Show me users"})
    assert response.status_code == 422
    assert response.json() == {"error": "prompt_type is required"}

@pytest.mark.asyncio
async def test_invalid_prompt_type(client):
    """Test validation of invalid prompt type."""
    response = await client.post("/query", json={"query": "Show me users", "prompt_type": "invalid"})
    assert response.status_code == 422
    assert response.json() == {"error": "Invalid prompt_type. Must be one of: sql, select, insert, update, delete"} 
