import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.backend.main import app
from src.backend.schema_vectorizer import SchemaVectorizer

# Create a test client
client = TestClient(app)

@pytest.fixture
def mock_db_engine():
    """Mock database engine for testing."""
    engine = Mock()
    with patch('src.backend.database.get_db_engine', return_value=engine):
        yield engine

@pytest.fixture
def mock_schema_vectorizer():
    """Mock schema vectorizer for testing."""
    vectorizer = Mock(spec=SchemaVectorizer)
    vectorizer.preload_schema_to_vectordb.return_value = None
    with patch('src.backend.main.SchemaVectorizer', return_value=vectorizer):
        yield vectorizer

@pytest.fixture
def mock_langchain():
    """Mock LangChain components."""
    with patch('src.backend.langchain_config.memory') as mock_memory, \
         patch('src.backend.langchain_config.chat_model') as mock_chat_model, \
         patch('src.backend.langchain_config.db_chain') as mock_db_chain:
        mock_memory.clear.return_value = None
        yield {
            'memory': mock_memory,
            'chat_model': mock_chat_model,
            'db_chain': mock_db_chain
        }

def test_nonexistent_table_error():
    """Test that querying a non-existent table returns a proper error message"""
    with patch('src.backend.langchain_mysql.LangChainMySQL.run_query_with_retry') as mock_run_query:
        mock_run_query.side_effect = Exception("Table 'test_db.nonexistent_table' doesn't exist")
        
        response = client.post("/query", json={"question": "Select * from nonexistent_table"})
        assert response.status_code == 500
        error_message = response.json()["detail"]
        assert "table" in error_message.lower()
        assert "exist" in error_message.lower()

def test_syntax_error():
    """Test that SQL syntax errors are handled properly"""
    with patch('src.backend.langchain_mysql.LangChainMySQL.run_query_with_retry') as mock_run_query:
        mock_run_query.side_effect = Exception("You have an error in your SQL syntax")
        
        response = client.post("/query", json={"question": "SELET * FROM users"})
        assert response.status_code == 500
        error_message = response.json()["detail"]
        assert "syntax" in error_message.lower()

def test_permission_error():
    """Test that permission errors are handled properly"""
    with patch('src.backend.langchain_mysql.LangChainMySQL.run_query_with_retry') as mock_run_query:
        mock_run_query.side_effect = Exception("Access denied for user")
        
        response = client.post("/query", json={"question": "DROP TABLE users"})
        assert response.status_code == 500
        error_message = response.json()["detail"]
        assert "access denied" in error_message.lower()

def test_openai_rate_limit_error():
    """Test that OpenAI rate limit errors are properly handled"""
    with patch('src.backend.langchain_mysql.LangChainMySQL.run_query_with_retry') as mock_run_query:
        mock_run_query.side_effect = Exception("Rate limit exceeded")
        
        response = client.post("/query", json={"question": "What are the users?"})
        assert response.status_code == 500
        error_message = response.json()["detail"]
        assert "rate limit" in error_message.lower()

def test_general_error_handling():
    """Test that general errors are properly handled"""
    with patch('src.backend.langchain_mysql.LangChainMySQL.run_query_with_retry') as mock_run_query:
        mock_run_query.side_effect = Exception("Unexpected error occurred")
        
        response = client.post("/query", json={"question": "What are the users?"})
        assert response.status_code == 500
        error_message = response.json()["detail"]
        assert "error" in error_message.lower()

if __name__ == "__main__":
    pytest.main([__file__]) 
