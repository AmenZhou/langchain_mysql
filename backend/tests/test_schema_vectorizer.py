import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, ForeignKey
from sqlalchemy.engine import Engine
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.backend.schema_vectorizer import SchemaVectorizer

@pytest.fixture
def temp_persist_dir():
    """Create a temporary directory for vector database persistence."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_engine():
    """Create a mock SQLAlchemy engine."""
    engine = Mock(spec=Engine)
    engine.connect.return_value = engine
    with patch('src.backend.database.engine', engine), \
         patch('src.backend.schema_vectorizer.engine', engine):
        yield engine

@pytest.fixture
def mock_inspector():
    inspector = Mock()
    # Mock get_table_names to return test tables
    inspector.get_table_names.return_value = ['users', 'orders']
    
    # Mock get_columns to return test columns
    test_columns = [
        {"name": "id", "type": Mock(python_type=int), "nullable": False, "default": None},
        {"name": "username", "type": Mock(python_type=str), "nullable": True, "default": None}
    ]
    inspector.get_columns.return_value = test_columns
    
    # Mock get_pk_constraint
    inspector.get_pk_constraint.return_value = {"constrained_columns": ["id"]}
    
    # Mock get_foreign_keys
    inspector.get_foreign_keys.return_value = []
    
    # Mock get_indexes
    inspector.get_indexes.return_value = [{"name": "idx_username", "column_names": ["username"]}]
    
    return inspector

@pytest.fixture
def vectorizer(temp_persist_dir, mock_engine, mock_inspector):
    """Create a SchemaVectorizer instance with mocked dependencies."""
    with patch('src.backend.schema_vectorizer.OpenAIEmbeddings') as mock_embeddings, \
         patch('src.backend.schema_vectorizer.Chroma') as mock_chroma, \
         patch('sqlalchemy.inspect', return_value=mock_inspector):
        
        vectorizer = SchemaVectorizer(persist_directory=temp_persist_dir)
        return vectorizer

def test_init(vectorizer, temp_persist_dir):
    """Test SchemaVectorizer initialization."""
    assert vectorizer.persist_directory == temp_persist_dir
    assert vectorizer.embeddings is not None
    assert vectorizer.vectordb is None

def test_get_all_tables(vectorizer, mock_inspector):
    """Test getting all table names."""
    with patch('sqlalchemy.inspect', return_value=mock_inspector):
        tables = vectorizer.get_all_tables()
        assert tables == ['users', 'orders']
        mock_inspector.get_table_names.assert_called_once()

def test_extract_table_schema(vectorizer, mock_inspector):
    """Test extracting schema for a specific table."""
    with patch('sqlalchemy.inspect', return_value=mock_inspector):
        schema = vectorizer.extract_table_schema('users')
        assert schema['table_name'] == 'users'
        assert len(schema['columns']) == 2
        assert schema['primary_keys'] == ['id']
        assert len(schema['indexes']) == 1
        assert schema['indexes'][0]['name'] == 'idx_username'

def test_create_schema_documents(vectorizer, mock_inspector):
    """Test creating document objects from schema."""
    with patch('sqlalchemy.inspect', return_value=mock_inspector):
        documents = vectorizer.create_schema_documents()
        assert len(documents) == 2
        assert any(doc.page_content.startswith('users:') for doc in documents)
        assert any(doc.page_content.startswith('orders:') for doc in documents)

@pytest.mark.asyncio
async def test_preload_schema_to_vectordb(vectorizer, mock_inspector):
    """Test preloading schema to vector database."""
    with patch('sqlalchemy.inspect', return_value=mock_inspector), \
         patch('src.backend.schema_vectorizer.Chroma.from_documents') as mock_from_docs:
        mock_from_docs.return_value = Mock()
        documents = vectorizer.preload_schema_to_vectordb()
        assert len(documents) == 2
        mock_from_docs.assert_called_once()

def test_query_schema_vectordb(vectorizer):
    """Test querying the vector database."""
    mock_results = [
        Mock(page_content="users: id username", metadata={"table_name": "users"}),
        Mock(page_content="orders: id user_id total", metadata={"table_name": "orders"})
    ]
    
    with patch.object(vectorizer, '_get_chroma_db') as mock_get_db:
        mock_db = Mock()
        mock_db.similarity_search.return_value = mock_results
        mock_get_db.return_value = mock_db
        
        results = vectorizer.query_schema_vectordb("find user related tables", k=2)
        
        assert len(results) == 2
        mock_db.similarity_search.assert_called_once_with("find user related tables", k=2)

def test_get_relevant_schema_no_query(vectorizer, mock_inspector):
    """Test getting schema information without a query."""
    with patch('sqlalchemy.inspect', return_value=mock_inspector):
        schema_text = vectorizer.get_relevant_schema()
        assert "users" in schema_text
        assert "orders" in schema_text

def test_get_relevant_schema_with_query(vectorizer, mock_inspector):
    """Test getting schema information with a query."""
    mock_results = [
        Mock(page_content="users: id username", metadata={"table_name": "users"})
    ]
    
    with patch('sqlalchemy.inspect', return_value=mock_inspector), \
         patch.object(vectorizer, 'query_schema_vectordb') as mock_query:
        mock_query.return_value = mock_results
        schema_text = vectorizer.get_relevant_schema("find user table", k=1)
        assert "users" in schema_text
        assert "id" in schema_text
        assert "username" in schema_text

def test_get_relevant_schema_fallback(vectorizer, mock_inspector):
    """Test fallback behavior when vector search fails."""
    with patch('sqlalchemy.inspect', return_value=mock_inspector), \
         patch.object(vectorizer, 'query_schema_vectordb', side_effect=Exception("Vector search failed")):
        schema_text = vectorizer.get_relevant_schema("find user table", k=1)
        # Should fall back to returning all schema info
        assert "users" in schema_text
        assert "orders" in schema_text

def test_error_handling_extract_table_schema(vectorizer, mock_inspector):
    """Test error handling in extract_table_schema."""
    mock_inspector.get_columns.side_effect = Exception("Database error")
    with patch('sqlalchemy.inspect', return_value=mock_inspector):
        schema = vectorizer.extract_table_schema('users')
        assert schema == {'table_name': 'users', 'error': 'Failed to extract schema'} 
