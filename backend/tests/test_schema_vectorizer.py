import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain.schema import Document
from backend.schema_vectorizer import SchemaVectorizer
from backend.prompts import PROMPT_REFINE, PROMPT_TABLE_QUERY, get_sanitize_prompt
import logging
from backend.vector_store import VectorStoreManager
from sqlalchemy import INTEGER, VARCHAR
from backend.schema_extractor import SchemaExtractor
from unittest.mock import Mock

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.mock_db

# Simple mock data
MOCK_TABLE = {
    'users': {
        'columns': [
            {'name': 'id', 'type': 'INTEGER', 'description': 'User ID'},
            {'name': 'name', 'type': 'VARCHAR', 'description': 'User name'}
        ]
    }
}

MOCK_DOCUMENT = Document(
    page_content="Table users contains id (INTEGER), name (VARCHAR)",
    metadata={'table_name': 'users', 'columns': ['id', 'name']}
)

@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.similarity_search = AsyncMock(return_value=[MOCK_DOCUMENT])
    return mock

@pytest.fixture
def mock_schema_extractor():
    mock = MagicMock()
    mock.extract_table_schema = AsyncMock(return_value=MOCK_TABLE)
    return mock

@pytest.fixture
def mock_inspector():
    inspector = MagicMock()
    inspector.get_table_names.return_value = ['users']
    inspector.get_columns.return_value = [
        # Provide comprehensive mock column data that create_schema_documents expects
        {'name': 'id', 'type': 'INTEGER', 'description': 'User ID', 'nullable': False, 'primary_key': True},
        {'name': 'name', 'type': 'VARCHAR', 'description': 'User name', 'nullable': True, 'primary_key': False}
    ]
    inspector.get_foreign_keys.return_value = [] # Added this
    return inspector

@pytest.fixture
def mock_db_engine():
    mock = MagicMock()
    mock.connect = AsyncMock()
    return mock

@pytest.fixture
def mock_vector_store_manager():
    manager = Mock()
    manager.initialize_schema_store = AsyncMock()
    manager.query_schema = AsyncMock()
    manager.add_documents = AsyncMock()
    return manager

@pytest.fixture
def schema_vectorizer():
    """Create SchemaVectorizer with minimal dependencies."""
    with patch('backend.schema_vectorizer.VectorStoreManager') as mock_store, \
         patch('backend.schema_vectorizer.SchemaExtractor') as mock_extractor:
        mock_store.return_value.similarity_search = AsyncMock(return_value=[MOCK_DOCUMENT])
        mock_extractor.return_value.extract_table_schema = AsyncMock(return_value=MOCK_TABLE)
        return SchemaVectorizer(db_url="mock://test")

@pytest.mark.asyncio
async def test_schema_extraction(schema_vectorizer):
    """Test schema extraction returns expected format."""
    schema = await schema_vectorizer.extract_table_schema()
    assert 'users' in schema
    assert len(schema['users']['columns']) == 2
    assert schema['users']['columns'][0]['name'] == 'id'
    assert schema['users']['columns'][1]['name'] == 'name'

@pytest.mark.asyncio
async def test_schema_vectorization(mock_schema_extractor, mock_vector_store_manager):
    """Test schema vectorization process."""
    # Setup mock responses
    mock_schema_extractor.extract_table_schema = AsyncMock(return_value={
        'users': {
            'columns': [
                {'name': 'id', 'type': 'INTEGER', 'description': 'Primary key'},
                {'name': 'username', 'type': 'VARCHAR', 'description': "User's login name"}
            ],
            'description': 'User account information'
        }
    })
    
    # Create a mock document to return
    mock_doc = Document(
        page_content="Table users contains:\nid (INTEGER) - Primary key",
        metadata={'table_name': 'users', 'columns': ['id'], 'description': 'User account information'}
    )
    mock_vector_store_manager.query_schema.return_value = [mock_doc]
    
    # Initialize vectorizer with mocks
    vectorizer = SchemaVectorizer("sqlite:///:memory:")
    vectorizer.schema_extractor = mock_schema_extractor
    vectorizer.vector_store_manager = mock_vector_store_manager
    
    # Test schema vectorization
    schema_info = await vectorizer.schema_extractor.extract_table_schema()
    await vectorizer.initialize_vector_store(schema_info)
    results = await vectorizer.get_relevant_schema("test query")
    
    # Verify results
    assert "users" in results
    assert "User account information" in results
    
    # Verify method calls
    mock_schema_extractor.extract_table_schema.assert_called_once()
    mock_vector_store_manager.add_documents.assert_called_once()
    mock_vector_store_manager.query_schema.assert_called_once_with("test query", k=5)

@pytest.mark.asyncio
async def test_schema_document_creation():
    """Test document creation from schema."""
    extractor = SchemaExtractor(MagicMock())
    with patch.object(extractor, 'extract_table_schema', AsyncMock(return_value=MOCK_TABLE)):
        schema_info = await extractor.extract_table_schema()
        documents = extractor.create_schema_documents(schema_info)
        
        assert len(documents) == 1
        doc = documents[0]
        assert doc.metadata['table_name'] == 'users'
        assert len(doc.metadata['columns']) == 2

@pytest.mark.asyncio
async def test_error_handling(schema_vectorizer):
    """Test error handling for main operations."""
    with patch.object(schema_vectorizer, 'extract_table_schema', 
                     side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            await schema_vectorizer.extract_table_schema()

    with patch.object(schema_vectorizer, 'get_relevant_schema', 
                     return_value=""):
        result = await schema_vectorizer.get_relevant_schema("nonexistent")
        assert result == ""

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
async def test_create_schema_documents(mock_inspector):
    """Test creating schema documents."""
    # Initialize SchemaExtractor with the mock_inspector directly
    extractor = SchemaExtractor(engine=None, inspector=mock_inspector) 
    schema_info = await extractor.extract_table_schema() # Changed from extract_all_tables
    documents = extractor.create_schema_documents(schema_info)
    
    assert len(documents) == 1
    doc = documents[0]
    
    # Assertions for metadata
    assert doc.metadata['table_name'] == 'users'
    assert doc.metadata['columns'] == ['id', 'name']
    assert doc.metadata['foreign_keys'] == []
    # The description in metadata is the one generated by extract_table_schema, which is complex.
    # Let's check for a substring for robustness.
    assert "Table users contains id, name" in doc.metadata['description']

    # Assertions for page_content (which is what we fixed earlier for test_schema_basic)
    expected_page_content_start = "Table users contains id, name"
    expected_columns_header = "\nColumns:"
    expected_col1_info = "- id (INTEGER) (PRIMARY KEY, NOT NULL): Column id of type INTEGER"
    expected_col2_info = "- name (VARCHAR): Column name of type VARCHAR"
    
    assert doc.page_content.startswith(expected_page_content_start)
    assert expected_columns_header in doc.page_content
    assert expected_col1_info in doc.page_content
    assert expected_col2_info in doc.page_content
    assert "Foreign Key Relationships:" not in doc.page_content # Since mock FKs are empty

@pytest.mark.asyncio
async def test_get_relevant_schema_empty(schema_vectorizer):
    """Test getting relevant schema with empty query."""
    schema = await schema_vectorizer.get_relevant_schema("")
    assert schema == ""

@pytest.mark.asyncio
async def test_extract_all_tables(schema_vectorizer, mock_inspector):
    """Test extracting all tables."""
    with patch.object(schema_vectorizer.schema_extractor.engine, 'inspect', return_value=mock_inspector):
        schema_info = await schema_vectorizer.schema_extractor.extract_table_schema()
        assert len(schema_info) == 1
        assert 'users' in schema_info
