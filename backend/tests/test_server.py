import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import httpx
import asyncio

from backend.server import app, get_db_engine, get_vectorizer, get_langchain_mysql
from backend.schema_vectorizer import SchemaVectorizer

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_db_engine():
    engine = MagicMock()
    connection = MagicMock()
    connection.__enter__.return_value = connection
    engine.connect.return_value = connection
    return engine

@pytest.fixture
def mock_schema_vectorizer():
    vectorizer = MagicMock(spec=SchemaVectorizer)
    vectorizer.preload_schema_to_vectordb = AsyncMock()
    return vectorizer

@pytest.fixture
def mock_langchain_mysql():
    langchain_mysql = MagicMock()
    langchain_mysql.query = AsyncMock()
    return langchain_mysql

@pytest.mark.asyncio
async def test_server_initialization_components():
    """Test that all components are properly initialized."""
    # Initialize components
    engine = get_db_engine()
    vectorizer = get_vectorizer()
    langchain_mysql = await get_langchain_mysql()
    
    # Test engine
    assert engine is not None
    assert engine.pool.size() > 0
    
    # Test vectorizer
    assert vectorizer is not None
    assert hasattr(vectorizer, 'vector_store')
    assert hasattr(vectorizer, 'schema_extractor')
    
    # Test langchain_mysql
    assert langchain_mysql is not None
    assert hasattr(langchain_mysql, 'db_chain')
    assert hasattr(langchain_mysql, 'schema_vectorizer')

@pytest.mark.asyncio
async def test_query_endpoint(mock_db_engine, mock_schema_vectorizer, mock_langchain_mysql):
    """Test the query endpoint with mocked components."""
    with patch('backend.server.get_db_engine', return_value=mock_db_engine), \
         patch('backend.server.get_vectorizer', return_value=mock_schema_vectorizer), \
         patch('backend.server.get_langchain_mysql', return_value=mock_langchain_mysql):
        
        # Mock the query result
        mock_langchain_mysql.process_query.return_value = {"result": "Test query result"}
        
        # Make request to query endpoint
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/query", json={"query": "select * from users"})
            
            # Verify response
            assert response.status_code == 200
            assert response.json() == {"result": "Test query result"}
            
            # Verify components were used
            mock_langchain_mysql.process_query.assert_called_once_with("select * from users")
            
@pytest.mark.asyncio
async def test_query_endpoint_error_handling(mock_db_engine, mock_schema_vectorizer, mock_langchain_mysql):
    """Test error handling in the query endpoint."""
    with patch('backend.server.get_db_engine', return_value=mock_db_engine), \
         patch('backend.server.get_vectorizer', return_value=mock_schema_vectorizer), \
         patch('backend.server.get_langchain_mysql', return_value=mock_langchain_mysql):
        
        # Mock an error
        mock_langchain_mysql.process_query.side_effect = Exception("Test error")
        
        # Make request to query endpoint
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/query", json={"query": "select * from users"})
            
            # Verify error response
            assert response.status_code == 500
            assert "Test error" in response.json()["detail"]

@pytest.mark.asyncio
async def test_invalid_query(client):
    """Test the query endpoint with an invalid query."""
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422
    assert "Query cannot be empty" in response.json()["detail"]

@pytest.mark.asyncio
async def test_health_check(client, mock_db_engine):
    """Test the health check endpoint."""
    with patch('backend.server.get_db_engine', return_value=mock_db_engine):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
