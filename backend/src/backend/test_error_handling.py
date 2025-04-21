from fastapi.testclient import TestClient
from langchain_mysql import app
import pytest
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_nonexistent_table_error():
    """Test that querying a non-existent table returns a proper error message"""
    response = client.post(
        "/query",
        json={"question": "Show me all records from the medical_service_specialty table"}
    )
    
    assert response.status_code == 400
    assert "Database Error" in response.json()["detail"]
    assert "medical_service_specialty" in response.json()["detail"]
    assert "does not exist" in response.json()["detail"]

def test_openai_rate_limit_error():
    """Test that OpenAI rate limit errors are properly handled"""
    with patch('langchain_mysql.run_query_with_retry') as mock_run:
        mock_run.side_effect = Exception("rate_limit_exceeded")
        
        response = client.post(
            "/query",
            json={"question": "Any question"}
        )
        
        assert response.status_code == 429
        assert "rate limit exceeded" in response.json()["detail"].lower()

def test_general_error_handling():
    """Test that general errors are properly handled"""
    with patch('langchain_mysql.run_query_with_retry') as mock_run:
        mock_run.side_effect = Exception("Some unexpected error")
        
        response = client.post(
            "/query",
            json={"question": "Any question"}
        )
        
        assert response.status_code == 500
        assert "unexpected error" in response.json()["detail"].lower()

if __name__ == "__main__":
    pytest.main([__file__]) 
