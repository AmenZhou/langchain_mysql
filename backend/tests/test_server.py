import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, Mock
from ..main import app
from ..langchain_mysql import LangChainMySQL
from src.backend.security import limiter
from starlette.requests import Request

@pytest.fixture
def mock_langchain_mysql():
    """Mock LangChainMySQL instance."""
    mock = AsyncMock()
    mock.process_query = AsyncMock(return_value={"result": "test result"})
    return mock

@pytest.fixture
def test_client(mock_langchain_mysql):
    """Create a test client."""
    # Clear any existing overrides
    app.dependency_overrides = {}
    
    # Override the rate limiter key function to always return a test key
    def get_test_key(request: Request):
        return "test_client"
    limiter.key_func = get_test_key
    
    # Mock OpenAI embeddings
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 1536  # Standard OpenAI embedding size
    
    # Mock ChatOpenAI
    mock_chat = Mock()
    mock_chat.agenerate.return_value = "test response"
    
    # Mock all OpenAI-related dependencies
    with patch('backend.server.get_langchain_mysql', return_value=mock_langchain_mysql), \
         patch('langchain_openai.OpenAIEmbeddings', return_value=mock_embeddings), \
         patch('langchain_openai.ChatOpenAI', return_value=mock_chat), \
         patch('langchain_community.chat_models.ChatOpenAI', return_value=mock_chat), \
         patch('openai.OpenAI') as mock_openai, \
         patch('backend.utils.sql_utils.AsyncOpenAI') as mock_async_openai:
        
        # Set up the OpenAI mock
        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.embeddings.create.return_value = {"data": [{"embedding": [0.1] * 1536}]}
        
        # Set up the AsyncOpenAI mock
        mock_async_openai_instance = mock_async_openai.return_value
        mock_async_openai_instance.embeddings.create.return_value = {"data": [{"embedding": [0.1] * 1536}]}
        
        client = TestClient(app)
        yield client
        
        # Clean up after test
        app.dependency_overrides = {}

def test_server_starts(test_client):
    """Test that the server starts and responds to health check."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_query_endpoint_exists(test_client):
    """Test that the query endpoint exists."""
    response = test_client.post("/query", json={"query": "test"})
    assert response.status_code in [200, 422]  # Either success or validation error
