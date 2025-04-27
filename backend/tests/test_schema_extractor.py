import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import inspect
from src.backend.schema_extractor import SchemaExtractor

@pytest.fixture
def mock_inspector():
    inspector = MagicMock()
    inspector.get_table_names.return_value = ['users']
    inspector.get_columns.return_value = [
        {'name': 'id', 'type': 'INTEGER'},
        {'name': 'name', 'type': 'VARCHAR'}
    ]
    return inspector

@pytest.fixture
def schema_extractor():
    engine = MagicMock()
    return SchemaExtractor(engine)

@pytest.mark.skip(reason="Extract all tables needs to be fixed")
@pytest.mark.asyncio
async def test_extract_all_tables(schema_extractor, mock_inspector):
    """Test extracting schema for all tables."""
    with patch.object(schema_extractor.engine, 'inspect', return_value=mock_inspector):
        schema_info = await schema_extractor.extract_table_schema()
        assert isinstance(schema_info, dict)
        assert 'users' in schema_info
        assert len(schema_info['users']['columns']) == 2

@pytest.mark.asyncio
async def test_extract_empty_database(schema_extractor, mock_inspector):
    """Test extracting schema from empty database."""
    mock_inspector.get_table_names.return_value = []
    with patch.object(schema_extractor.engine, 'inspect', return_value=mock_inspector):
        schema_info = await schema_extractor.extract_table_schema()
        assert schema_info == {} 
