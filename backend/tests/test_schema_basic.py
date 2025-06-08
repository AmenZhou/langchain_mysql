import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain.schema import Document
from schema_extractor import SchemaExtractor
from schema_vectorizer import SchemaVectorizer
from sqlalchemy import inspect, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector

# Common fixtures
@pytest.fixture
def mock_inspector():
    inspector = MagicMock(spec=Inspector)
    inspector.get_table_names.return_value = ['users']
    inspector.get_columns.return_value = [
        {'name': 'id', 'type': 'INTEGER'},
        {'name': 'name', 'type': 'VARCHAR'}
    ]
    inspector.get_foreign_keys.return_value = []
    return inspector

@pytest.fixture
def mock_engine():
    engine = MagicMock(spec=Engine)
    engine.dialect = MagicMock()
    engine.dialect.name = 'mysql'
    return engine

@pytest.fixture
def mock_schema_extractor():
    mock = MagicMock()
    mock.extract_table_schema = AsyncMock(return_value={
        'users': {
            'columns': [
                {'name': 'id', 'type': 'INTEGER'},
                {'name': 'name', 'type': 'VARCHAR'}
            ]
        }
    })
    return mock

@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.similarity_search = AsyncMock(return_value=[])
    return mock

# Simplified tests
@pytest.mark.asyncio
async def test_basic_schema_extraction(mock_inspector, mock_engine):
    """Basic test for schema extraction."""
    extractor = SchemaExtractor(engine=mock_engine, inspector=mock_inspector)
    
    schema_info = await extractor.extract_table_schema()
    assert isinstance(schema_info, dict)
    assert 'users' in schema_info
    assert len(schema_info['users']['columns']) == 2

@pytest.mark.asyncio
async def test_basic_document_creation(mock_schema_extractor):
    """Basic test for document creation."""
    schema_info = await mock_schema_extractor.extract_table_schema()
    extractor = SchemaExtractor(MagicMock())
    documents = extractor.create_schema_documents(schema_info)
    assert len(documents) == 1
    assert documents[0].page_content.startswith("Table users contains")

@pytest.mark.asyncio
async def test_basic_schema_vectorization(mock_schema_extractor, mock_vector_store):
    """Basic test for schema vectorization."""
    vectorizer = SchemaVectorizer(db_url="mock://test")
    vectorizer.schema_extractor = mock_schema_extractor
    vectorizer.vector_store = mock_vector_store
    
    schema = await vectorizer.get_relevant_schema("")
    assert schema == ""

@pytest.mark.asyncio
async def test_basic_table_extraction(mock_inspector, mock_engine):
    """Basic test for table extraction."""
    vectorizer = SchemaVectorizer(db_url="mock://test")
    vectorizer.schema_extractor = SchemaExtractor(engine=mock_engine, inspector=mock_inspector)
    schema_info = await vectorizer.schema_extractor.extract_table_schema()
    assert len(schema_info) == 1
    assert 'users' in schema_info 
