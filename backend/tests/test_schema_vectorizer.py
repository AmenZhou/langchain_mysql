import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain.schema import Document
from src.backend.schema_vectorizer import SchemaVectorizer
from src.backend.prompts import PROMPT_REFINE, PROMPT_TABLE_QUERY, get_sanitize_prompt

@pytest.fixture
def mock_vector_store():
    mock = AsyncMock()
    mock.query_schema = AsyncMock()
    mock.query_prompts = AsyncMock()
    mock.initialize_schema_store = AsyncMock()
    mock.initialize_prompt_store = AsyncMock()
    mock.add_texts = AsyncMock()
    mock.similarity_search = AsyncMock()
    mock.delete_collection = AsyncMock()
    return mock

@pytest.fixture
def mock_schema_extractor():
    mock = MagicMock()
    mock.extract_table_schema = AsyncMock()
    mock.create_schema_documents = MagicMock()
    mock.create_prompt_documents = MagicMock()
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
def vectorizer(mock_vector_store, mock_schema_extractor):
    with patch('src.backend.schema_vectorizer.VectorStoreManager', return_value=mock_vector_store), \
         patch('src.backend.schema_vectorizer.SchemaExtractor', return_value=mock_schema_extractor):
        vectorizer = SchemaVectorizer(persist_directory="./chroma_db")
        return vectorizer

def test_init(vectorizer):
    """Test initialization of SchemaVectorizer."""
    assert vectorizer.persist_directory == "./chroma_db"
    assert vectorizer.vector_store is not None
    assert vectorizer.schema_extractor is not None

def test_get_all_tables(vectorizer, mock_inspector):
    """Test getting all table names."""
    # Set up the mock inspector with a return value
    mock_inspector.get_table_names.return_value = ['test_table1', 'test_table2']
    
    # Patch sqlalchemy.inspect to return our mock inspector
    with patch('src.backend.schema_vectorizer.inspect', return_value=mock_inspector):
        tables = vectorizer.get_all_tables()
        assert isinstance(tables, list)
        assert all(isinstance(table, str) for table in tables)
        mock_inspector.get_table_names.assert_called_once()

@pytest.mark.asyncio
async def test_extract_table_schema(vectorizer, mock_schema_extractor):
    """Test extracting schema for a specific table."""
    expected_schema = {
        'columns': ['id', 'username'],
        'description': 'User table'
    }
    mock_schema_extractor.extract_table_schema.return_value = expected_schema
    schema = await vectorizer.extract_table_schema('users')
    assert schema == expected_schema
    mock_schema_extractor.extract_table_schema.assert_called_once_with('users')

@pytest.mark.asyncio
async def test_error_handling_extract_table_schema(vectorizer, mock_schema_extractor):
    """Test error handling during schema extraction."""
    mock_schema_extractor.extract_table_schema.side_effect = Exception("Database error")
    result = await vectorizer.extract_table_schema('users')
    assert result == {"table_name": "users", "error": "Failed to extract schema"}

@pytest.mark.asyncio
async def test_create_schema_documents(vectorizer, mock_schema_extractor):
    """Test creating schema documents."""
    schema_info = {
        "users": {
            "columns": ["id", "username"],
            "description": "User table"
        }
    }
    expected_docs = [Document(page_content="User table schema", metadata={"table": "users"})]
    mock_schema_extractor.create_schema_documents.return_value = expected_docs
    
    documents = await vectorizer.create_schema_documents(schema_info)
    assert documents == expected_docs
    mock_schema_extractor.create_schema_documents.assert_called_once_with(schema_info)

def test_create_prompt_documents(vectorizer, mock_schema_extractor):
    """Test creating prompt documents."""
    expected_docs = [
        Document(page_content=PROMPT_REFINE, metadata={"type": "refine"}),
        Document(page_content=PROMPT_TABLE_QUERY, metadata={"type": "table"})
    ]
    mock_schema_extractor.create_prompt_documents.return_value = expected_docs
    
    documents = vectorizer.create_prompt_documents()
    assert documents == expected_docs
    mock_schema_extractor.create_prompt_documents.assert_called_once()

@pytest.mark.asyncio
async def test_preload_schema_to_vectordb(vectorizer, mock_vector_store, mock_schema_extractor):
    """Test preloading schema to vectordb."""
    schema_info = {"users": {"columns": ["id", "username"]}}
    schema_docs = [Document(page_content="User schema", metadata={"table": "users"})]
    prompt_docs = [Document(page_content=PROMPT_REFINE, metadata={"type": "refine"})]
    
    mock_schema_extractor.create_schema_documents.return_value = schema_docs
    mock_schema_extractor.create_prompt_documents.return_value = prompt_docs
    
    await vectorizer.preload_schema_to_vectordb(schema_info)
    
    mock_vector_store.initialize_schema_store.assert_called_once_with(schema_docs, None)
    mock_vector_store.initialize_prompt_store.assert_called_once_with(prompt_docs)

@pytest.mark.asyncio
async def test_get_relevant_prompt(vectorizer, mock_vector_store):
    """Test getting relevant prompt."""
    query = "test query"
    expected_prompt = "Custom prompt"
    mock_vector_store.query_prompts.return_value = [Document(page_content=expected_prompt)]
    
    result = await vectorizer.get_relevant_prompt(query, "refine")
    assert result == expected_prompt
    mock_vector_store.query_prompts.assert_called_once_with(query, "refine")

@pytest.mark.asyncio
async def test_get_relevant_prompt_fallback(vectorizer, mock_vector_store):
    """Test prompt fallback behavior."""
    mock_vector_store.query_prompts.return_value = []

    # Test refine prompt fallback
    result = await vectorizer.get_relevant_prompt("query", "refine")
    assert result == PROMPT_REFINE

    # Test table query prompt fallback
    result = await vectorizer.get_relevant_prompt("query", "table")
    assert result == PROMPT_TABLE_QUERY

    # Test sanitize prompt fallback with proper sql_result
    sql_result = "SELECT * FROM users"
    result = await vectorizer.get_relevant_prompt(sql_result, "sanitize")
    assert "sanitize" in result.lower()  # Just check if it's a sanitize-related prompt

@pytest.mark.asyncio
async def test_get_relevant_schema(vectorizer, mock_vector_store):
    """Test getting relevant schema with vector search."""
    query = "user details"
    expected_content = "User schema details"
    mock_vector_store.query_schema.return_value = [Document(page_content=expected_content)]
    
    result = await vectorizer.get_relevant_schema(query)
    assert result == expected_content
    mock_vector_store.query_schema.assert_called_once_with(query, k=1)

@pytest.mark.asyncio
async def test_get_relevant_schema_fallback(vectorizer, mock_vector_store, mock_schema_extractor):
    """Test fallback behavior when vector search fails."""
    mock_vector_store.query_schema.side_effect = Exception("Vector store error")
    mock_schema_extractor.extract_table_schema.return_value = {
        "users": {
            "columns": ["id", "username"],
            "description": "User table"
        }
    }
    
    result = await vectorizer.get_relevant_schema("test query")
    assert "Table: users" in result
    assert "Columns: id, username" in result
    assert "User table" in result

@pytest.mark.asyncio
async def test_get_relevant_schema_no_query(vectorizer):
    """Test getting relevant schema with no query."""
    result = await vectorizer.get_relevant_schema(None)
    assert result == ""
    
    result = await vectorizer.get_relevant_schema("")
    assert result == ""

@pytest.mark.asyncio
async def test_get_relevant_schema_error_handling(vectorizer, mock_vector_store, mock_schema_extractor):
    """Test error handling in get_relevant_schema when both vector store and fallback fail."""
    mock_vector_store.query_schema.side_effect = Exception("Vector store error")
    mock_schema_extractor.extract_table_schema.side_effect = Exception("Schema extraction error")
    
    result = await vectorizer.get_relevant_schema("test query")
    assert result == ""  # Should return empty string on complete failure

@pytest.mark.asyncio
async def test_preload_schema_error_handling(vectorizer, mock_vector_store, mock_schema_extractor):
    """Test error handling during schema preloading."""
    # Test case 1: Schema document creation fails
    mock_schema_extractor.create_schema_documents.side_effect = Exception("Document creation error")
    try:
        await vectorizer.preload_schema_to_vectordb({})
    except Exception as e:
        assert str(e) == "Document creation error"
    mock_vector_store.initialize_schema_store.assert_not_called()
    
    # Test case 2: Vector store initialization fails
    mock_schema_extractor.create_schema_documents.side_effect = None
    mock_schema_extractor.create_schema_documents.return_value = [Document(page_content="test")]
    mock_vector_store.initialize_schema_store.side_effect = Exception("Store initialization error")
    
    try:
        await vectorizer.preload_schema_to_vectordb({})
    except Exception as e:
        assert str(e) == "Store initialization error"
    mock_vector_store.initialize_schema_store.assert_called_once()

@pytest.mark.asyncio
async def test_get_relevant_prompt_error_handling(vectorizer, mock_vector_store):
    """Test error handling in get_relevant_prompt."""
    # Set up the mock to raise an exception
    mock_vector_store.query_prompts.side_effect = Exception("Query error")
    
    # Test that we get the default prompt when an error occurs
    result = await vectorizer.get_relevant_prompt("query", "refine")
    assert result == PROMPT_REFINE
    
    result = await vectorizer.get_relevant_prompt("query", "table")
    assert result == PROMPT_TABLE_QUERY
    
    # For sanitize prompt, we need to provide a sql_result
    sql_result = "SELECT * FROM users"
    result = await vectorizer.get_relevant_prompt(sql_result, "sanitize")
    assert "sanitize" in result.lower()  # Just check if it's a sanitize-related prompt
