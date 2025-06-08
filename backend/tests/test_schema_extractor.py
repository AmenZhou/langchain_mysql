import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import inspect
from schema_extractor import SchemaExtractor

@pytest.fixture
def mock_inspector():
    inspector = MagicMock()
    inspector.get_table_names.return_value = ['users']
    inspector.get_columns.return_value = [
        {'name': 'id', 'type': 'INTEGER'},
        {'name': 'name', 'type': 'VARCHAR'}
    ]
    inspector.get_foreign_keys.return_value = []
    return inspector

@pytest.fixture
def schema_extractor(mock_inspector):
    engine = MagicMock()
    return SchemaExtractor(engine=engine, inspector=mock_inspector)

@pytest.mark.asyncio
async def test_extract_all_tables(schema_extractor):
    """Test extracting schema for all tables."""
    schema_info = await schema_extractor.extract_table_schema()
    assert isinstance(schema_info, dict)
    assert 'users' in schema_info
    assert 'columns' in schema_info['users']
    assert len(schema_info['users']['columns']) == 2

@pytest.mark.asyncio
async def test_extract_empty_database(schema_extractor, mock_inspector):
    """Test extracting schema from empty database."""
    mock_inspector.get_table_names.return_value = []
    schema_info = await schema_extractor.extract_table_schema()
    assert schema_info == {} 
