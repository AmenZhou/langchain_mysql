from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from src.backend.database import get_db_engine
from src.backend.schema_vectorizer import SchemaVectorizer
from src.backend.server import app

client = TestClient(app)

def test_server_initialization_components():
    """Test that all components are properly initialized during server startup."""
    mock_db_engine = Mock()
    mock_db_engine.connect.return_value = mock_db_engine
    mock_schema_vectorizer = Mock(spec=SchemaVectorizer)

    with patch('src.backend.database.engine', None), \
         patch('src.backend.database.create_engine', return_value=mock_db_engine) as mock_create_engine, \
         patch('src.backend.schema_vectorizer.engine', mock_db_engine), \
         patch('src.backend.schema_vectorizer.SchemaVectorizer', return_value=mock_schema_vectorizer):
        # Initialize components
        db_engine = get_db_engine()
        schema_vectorizer = SchemaVectorizer()

        # Verify components are initialized
        assert mock_create_engine.called
        assert isinstance(schema_vectorizer, Mock)  # Verify schema_vectorizer was created
        assert schema_vectorizer is mock_schema_vectorizer  # Verify it's our mock

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_query_endpoint():
    """Test the query endpoint with mocked components."""
    mock_response = {"result": "Mocked query result"}
    
    with patch('src.backend.server.run_query_with_retry', return_value=mock_response) as mock_query:
        response = client.post(
            "/query",
            json={"query": "SELECT * FROM test_table"}
        )
        
        assert response.status_code == 200
        assert response.json() == mock_response
        mock_query.assert_called_once()

def test_schema_endpoint():
    """Test the schema endpoint with mocked schema vectorizer."""
    mock_schema = {
        "tables": ["table1", "table2"],
        "columns": {
            "table1": ["id", "name"],
            "table2": ["id", "description"]
        }
    }
    
    with patch('src.backend.server.schema_vectorizer.get_all_table_names', return_value=mock_schema["tables"]), \
         patch('src.backend.server.schema_vectorizer.extract_table_schema', return_value=mock_schema["columns"]):
        response = client.get("/schema")
        
        assert response.status_code == 200
        assert response.json() == mock_schema

def test_invalid_query():
    """Test handling of invalid SQL queries."""
    with patch('src.backend.server.run_query_with_retry', side_effect=Exception("Invalid SQL syntax")):
        response = client.post(
            "/query",
            json={"query": "INVALID SQL"}
        )
        
        assert response.status_code == 500
        assert "error" in response.json()

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
