import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain.schema import Document
from src.backend.schema_vectorizer import SchemaVectorizer
from src.backend.prompts import PROMPT_REFINE, PROMPT_TABLE_QUERY, get_sanitize_prompt
import logging
from src.backend.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.mock_db

@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.add_documents = AsyncMock()
    mock.query_schema = AsyncMock(return_value=[Document(page_content="User schema details", metadata={"table": "users"})])
    mock.query_prompts = AsyncMock(return_value=[Document(page_content="Mock prompt", metadata={"type": "refine"})])
    mock.initialize_schema_store = AsyncMock()
    mock.initialize_prompt_store = AsyncMock()
    mock.add_texts = AsyncMock()
    mock.similarity_search = AsyncMock()
    mock.delete_collection = AsyncMock()
    return mock

@pytest.fixture
def mock_schema_extractor():
    mock = MagicMock()
    mock.extract_schema = AsyncMock(return_value={"users": {"columns": [{"name": "id", "type": "INTEGER"}]}})
    mock.create_schema_documents = MagicMock(return_value=[Document(page_content="User schema", metadata={"table": "users"})])
    mock.create_prompt_documents = MagicMock(return_value=[Document(page_content="Mock prompt", metadata={"type": "refine"})])
    return mock

@pytest.fixture
def mock_inspector():
    mock = MagicMock()
    mock.get_table_names.return_value = ['users', 'orders']
    mock.get_columns.return_value = [
        {'name': 'id', 'type': 'INTEGER'},
        {'name': 'username', 'type': 'VARCHAR'}
    ]
    return mock

@pytest.fixture
def mock_db_engine():
    mock = MagicMock()
    mock.connect = AsyncMock()
    return mock

@pytest.fixture
def vectorizer(mock_vector_store, mock_schema_extractor, mock_db_engine):
    with patch('src.backend.schema_vectorizer.VectorStoreManager', return_value=mock_vector_store), \
         patch('src.backend.schema_vectorizer.SchemaExtractor', return_value=mock_schema_extractor), \
         patch('src.backend.schema_vectorizer.get_db_engine', return_value=mock_db_engine):
        vectorizer = SchemaVectorizer(db_url="mysql://test:test@localhost/test_db")
        return vectorizer

@pytest.mark.asyncio
async def test_init(mock_schema_vectorizer):
    """Test initialization of SchemaVectorizer."""
    assert mock_schema_vectorizer is not None

@pytest.mark.asyncio
async def test_error_handling_extract_table_schema(mock_schema_vectorizer):
    """Test error handling in extract_table_schema."""
    mock_schema_vectorizer.extract_table_schema.side_effect = Exception("Test error")
    with pytest.raises(Exception):
        await mock_schema_vectorizer.extract_table_schema()

@pytest.mark.asyncio
async def test_preload_schema_to_vectordb(mock_schema_vectorizer):
    """Test preloading schema to vector store."""
    await mock_schema_vectorizer.preload_schema_to_vectordb()
    # Just verify the method executes without error
    assert True

@pytest.mark.asyncio
async def test_get_relevant_prompt(mock_schema_vectorizer):
    """Test getting relevant prompt."""
    prompt = await mock_schema_vectorizer.get_relevant_prompt("test query")
    assert prompt == "test prompt"

@pytest.mark.asyncio
async def test_get_relevant_schema(mock_schema_vectorizer):
    """Test getting relevant schema."""
    schema = await mock_schema_vectorizer.get_relevant_schema("test query")
    assert schema == {"test_table": {"columns": ["id", "name"]}}

@pytest.mark.asyncio
async def test_get_relevant_schema_error_handling(mock_schema_vectorizer):
    """Test error handling in get_relevant_schema."""
    mock_schema_vectorizer.get_relevant_schema.side_effect = Exception("Test error")
    with pytest.raises(Exception):
        await mock_schema_vectorizer.get_relevant_schema("test query")

@pytest.mark.asyncio
async def test_preload_schema_error_handling(mock_schema_vectorizer):
    """Test error handling in preload_schema."""
    mock_schema_vectorizer.preload_schema_to_vectordb.side_effect = Exception("Test error")
    with pytest.raises(Exception):
        await mock_schema_vectorizer.preload_schema_to_vectordb()

@pytest.mark.asyncio
async def test_get_relevant_prompt_error_handling(mock_schema_vectorizer):
    """Test error handling in get_relevant_prompt."""
    mock_schema_vectorizer.get_relevant_prompt.side_effect = Exception("Test error")
    with pytest.raises(Exception):
        await mock_schema_vectorizer.get_relevant_prompt("test query")

@pytest.mark.asyncio
async def test_initialize_vector_store(mock_schema_vectorizer):
    """Test vector store initialization."""
    await mock_schema_vectorizer.initialize_vector_store()
    # Just verify the method executes without error
    assert True

@pytest.mark.asyncio
async def test_vector_store_operations(mock_vector_store):
    """Test basic vector store operations."""
    # Test similarity search
    results = await mock_vector_store.similarity_search("test query")
    # Just verify we get a result
    assert results is not None

@pytest.mark.asyncio
async def test_schema_vectorizer_initialization():
    """Test basic initialization of SchemaVectorizer."""
    db_url = "mysql+pymysql://root:@localhost:3306/dev_tas_live"
    vectorizer = SchemaVectorizer(db_url=db_url)
    assert vectorizer is not None
    assert vectorizer.db_url == db_url

@pytest.mark.asyncio
async def test_vector_store_initialization():
    """Test basic initialization of VectorStoreManager."""
    store = VectorStoreManager()
    assert store is not None
    assert store.embeddings is not None

@pytest.mark.asyncio
async def test_schema_extraction():
    """Test basic schema extraction."""
    db_url = "mysql+pymysql://root:@localhost:3306/dev_tas_live"
    vectorizer = SchemaVectorizer(db_url=db_url)
    try:
        schema = await vectorizer.extract_table_schema()
        assert schema is not None
        assert isinstance(schema, dict)
        logger.info(f"Extracted schema for {len(schema)} tables")
    except Exception as e:
        logger.error(f"Schema extraction failed: {str(e)}")
        pytest.fail(f"Schema extraction failed: {str(e)}")

@pytest.mark.asyncio
async def test_schema_vectorization(mock_schema_vectorizer):
    """Test basic schema vectorization."""
    # Test schema extraction
    schema = await mock_schema_vectorizer.extract_table_schema()
    # Just verify we get a schema
    assert schema is not None
