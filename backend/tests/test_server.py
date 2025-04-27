import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from backend.main import app
from backend.langchain_mysql import LangChainMySQL

@pytest.fixture
def mock_langchain_mysql():
    """Mock LangChainMySQL instance."""
    mock = AsyncMock()
    mock.process_query = AsyncMock(return_value={"result": "test result"})
    return mock

@pytest.fixture
def test_client(mock_langchain_mysql):
    """Create a test client."""
    app.dependency_overrides = {}  # Clear any existing overrides
    app.dependency_overrides[LangChainMySQL] = lambda: mock_langchain_mysql
    client = TestClient(app)
    yield client
    app.dependency_overrides = {}  # Clean up after test

def test_server_starts(test_client):
    """Test that the server starts and responds to health check."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_query_endpoint_exists(test_client):
    """Test that the query endpoint exists."""
    response = test_client.post("/query", json={"query": "test"})
    assert response.status_code in [200, 422]  # Either success or validation error
