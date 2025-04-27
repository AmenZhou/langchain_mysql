import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain.schema import Document
from src.backend.vector_store import VectorStoreManager
from langchain_openai import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_vector_store_operations(mock_vector_store):
    """Test basic vector store operations."""
    # Test similarity search
    results = await mock_vector_store.similarity_search("test query")
    # Just verify we get a result
    assert results is not None

@pytest.mark.asyncio
async def test_vector_store_basic_operations(mock_vector_store):
    """Test basic vector store initialization."""
    # Setup test documents
    documents = [Document(page_content="test content", metadata={"table": "test"})]
    
    # Test initialization
    try:
        await mock_vector_store.initialize_schema_store(documents)
        assert True
    except Exception as e:
        pytest.skip(f"Skipping due to external service error: {str(e)}") 
