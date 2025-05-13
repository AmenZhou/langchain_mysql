import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from langchain.schema import Document
from backend.vector_store import VectorStoreManager
from ..schema_vectorizer import SchemaVectorizer
import asyncio
from langchain_community.vectorstores import FAISS
import respx
from httpx import Response

@pytest.fixture
def mock_embeddings():
    embeddings = Mock()
    embeddings.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    embeddings.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    return embeddings

@pytest.fixture
def mock_faiss():
    faiss = Mock(spec=FAISS)
    faiss.similarity_search = Mock(return_value=[
        Document(
            page_content="Test content",
            metadata={'table_name': 'users', 'columns': ['id', 'username']}
        )
    ])
    return faiss

@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Table users contains:\nid (INTEGER) - Primary key\nusername (VARCHAR) - User's login name",
            metadata={
                'table_name': 'users',
                'columns': ['id', 'username'],
                'description': 'User account information'
            }
        ),
        Document(
            page_content="Table posts contains:\nid (INTEGER) - Primary key\ntitle (VARCHAR) - Post title",
            metadata={
                'table_name': 'posts',
                'columns': ['id', 'title'],
                'description': 'Blog posts'
            }
        )
    ]

@pytest.mark.asyncio
async def test_initialize_schema_store(mock_embeddings, sample_documents, monkeypatch):
    # Arrange
    manager = VectorStoreManager(embeddings=mock_embeddings)
    mock_faiss = MagicMock()
    monkeypatch.setattr("backend.vector_store.vector_store.FAISS", mock_faiss)
    
    # Act
    await manager.initialize_schema_store(sample_documents)
    
    # Assert
    assert manager.schema_vectordb is not None
    mock_embeddings.embed_documents.assert_not_called()  # We're using from_texts now
    mock_faiss.from_texts.assert_called_once()

@pytest.mark.asyncio
async def test_query_schema(mock_embeddings, sample_documents, mock_faiss, monkeypatch):
    # Arrange
    manager = VectorStoreManager(embeddings=mock_embeddings)
    monkeypatch.setattr("backend.vector_store.vector_store.FAISS", Mock(return_value=mock_faiss))
    await manager.initialize_schema_store(sample_documents)
    manager.schema_vectordb = mock_faiss
    
    # Act
    results = await manager.query_schema("find user information")
    
    # Assert
    assert len(results) == 1
    assert results[0].metadata['table_name'] == 'users'
    assert 'username' in results[0].metadata['columns']

@pytest.mark.asyncio
async def test_query_schema_empty_query():
    # Arrange
    manager = VectorStoreManager()
    
    # Act & Assert
    with pytest.raises(ValueError, match="Query cannot be empty"):
        await manager.query_schema("")

@pytest.mark.asyncio
async def test_query_schema_uninitialized_store(mock_embeddings):
    # Arrange
    manager = VectorStoreManager(embeddings=mock_embeddings)
    manager.schema_vectordb = None  # Ensure vector store is not initialized
    
    # Act & Assert
    with pytest.raises(ValueError, match="Schema vector store not initialized"):
        await manager.query_schema("find users")

@pytest.mark.asyncio
async def test_schema_vectorizer_integration(mock_embeddings, sample_documents, monkeypatch):
    # Arrange
    db_url = "sqlite:///:memory:"
    vectorizer = SchemaVectorizer(db_url=db_url)
    vectorizer.vector_store_manager = VectorStoreManager(embeddings=mock_embeddings)
    
    # Mock schema extraction
    vectorizer.schema_extractor.extract_table_schema = AsyncMock(return_value={
        'users': {
            'columns': [
                {'name': 'id', 'type': 'INTEGER', 'description': 'Primary key'},
                {'name': 'username', 'type': 'VARCHAR', 'description': "User's login name"}
            ],
            'description': 'User account information'
        }
    })
    
    # Mock FAISS
    mock_faiss = MagicMock()
    mock_faiss.similarity_search = Mock(return_value=[sample_documents[0]])
    monkeypatch.setattr("backend.vector_store.vector_store.FAISS", Mock(return_value=mock_faiss))
    
    # Act
    schema_info = await vectorizer.schema_extractor.extract_table_schema()
    await vectorizer.initialize_vector_store(schema_info)
    
    # Set up the mock for query_schema
    vectorizer.vector_store_manager.schema_vectordb = mock_faiss
    result = await vectorizer.get_relevant_schema("find user information")
    
    # Assert
    assert "Table: users" in result
    assert "Description: User account information" in result 

@pytest.mark.asyncio
async def test_embeddings_stubbed_with_respx():
    # Arrange
    with respx.mock:
        # Mock the embeddings endpoint with more realistic values
        mock_embeddings = [0.025515518153991636] * 1536  # Use the actual value we're seeing
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=Response(200, json={"data": [{"embedding": mock_embeddings}]})
        )
        
        # Create an instance of OpenAIEmbeddings with a dummy key
        from langchain_community.embeddings import OpenAIEmbeddings
        emb = OpenAIEmbeddings(openai_api_key="dummy-key")
        
        # Act
        vec = emb.embed_query("hello")
        
        # Assert
        assert len(vec) == 1536  # Check length
        # Check that values are approximately what we expect
        assert all(abs(v - 0.025515518153991636) < 1e-10 for v in vec)  # Allow small floating point differences 
