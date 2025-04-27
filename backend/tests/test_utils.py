import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from sqlalchemy import MetaData, inspect
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from langchain_core.outputs import ChatResult
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_community.utilities import SQLDatabase
from src.backend.utils import sanitize_sql_response
from src.backend.prompts import get_sanitize_prompt
from langchain_community.chat_models import ChatOpenAI
from fastapi import HTTPException
from sqlalchemy.exc import ProgrammingError, OperationalError
from openai import RateLimitError, APIError

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from backend.utils.sql_utils import (
    generate_column_query,
    generate_table_query,
    get_sql_chain,
    refine_prompt_with_ai,
    sanitize_sql_response
)

# Mock the database connection before importing any modules
mock_engine = MagicMock()
mock_inspector = MagicMock()
mock_inspector.get_table_names.return_value = [
    "active_messages",
    "message_participants",
    "message_room_trigger_relations",
    "message_rooms",
    "system_messages"
]
mock_inspector.get_columns.return_value = [{"name": "col1", "type": "string", "nullable": True}]
mock_inspector.get_pk_constraint.return_value = {}
mock_inspector.get_foreign_keys.return_value = []
mock_inspector.get_indexes.return_value = []

mock_engine.connect.return_value = MagicMock()
mock_inspect = MagicMock(return_value=mock_inspector)

# Mock the database module
mock_database = MagicMock()
mock_database.engine = mock_engine
mock_database.db = MagicMock()
mock_database.get_db = MagicMock(return_value=mock_database.db)

# Mock the MinimalSQLDatabase
class MockMinimalSQLDatabase:
    @classmethod
    def from_uri(cls, database_uri, **kwargs):
        return MagicMock(get_table_info=MagicMock(return_value=""))

# Mock the OpenAI client
mock_openai = MagicMock()
mock_openai.api_key = "test-key"

# Mock the Pydantic validator before any imports
mock_validator = MagicMock()
mock_validator.allow_reuse = True
with patch('pydantic.v1.class_validators.root_validator', return_value=mock_validator):
    # Now import the modules that use Pydantic
    from backend.utils import refine_prompt_with_ai, sanitize_sql_response

# Mock database components
mock_connection = MagicMock()
mock_engine.connect.return_value = mock_connection

# Mock SchemaVectorizer
mock_schema_vectorizer = MagicMock()
mock_schema_vectorizer.get_relevant_schema.return_value = "mocked schema"

# Mock OpenAI response
mock_chat_completion = ChatCompletion(
    id="chatcmpl-123",
    choices=[{
        "index": 0,
        "message": ChatCompletionMessage(
            role="assistant",
            content="Mocked response content"
        ),
        "finish_reason": "stop"
    }],
    created=1234567890,
    model="gpt-3.5-turbo",
    object="chat.completion",
    usage={"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30}
)

# Mock OpenAI client
mock_openai = AsyncMock()
mock_openai.chat.completions.create.return_value = mock_chat_completion

# Mock LangChain components
mock_prompt = MagicMock()
mock_prompt.__or__ = MagicMock()

mock_chain = AsyncMock()
mock_chain.ainvoke.return_value = AIMessage(content="sanitized response")

mock_prompt.__or__.return_value = mock_chain

mock_prompt_template = MagicMock()
mock_prompt_template.from_messages.return_value = mock_prompt

@pytest.fixture(autouse=True)
def mock_modules():
    """Mock all required modules before any imports."""
    with patch.dict('sys.modules', {
        'langchain_community.chat_models': MagicMock(),
        'langchain_community.chat_models.openai': MagicMock(),
        'langchain_core.callbacks': MagicMock(),
        'langchain_core.callbacks.manager': MagicMock(),
        'langsmith.run_helpers': MagicMock(),
        'langsmith.run_trees': MagicMock(),
        'langsmith.schemas': MagicMock(),
        'langsmith.utils': MagicMock()
    }):
        yield

@pytest.fixture(autouse=True)
def mock_env():
    """Set up environment variables for testing."""
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'DATABASE_URL': 'mysql://test:test@localhost/test'
    }):
        yield

class MockSQLDatabase:
    def __init__(self):
        self.dialect = "mysql"
        
    def get_table_info(self):
        return "mocked table info"

@pytest.fixture
def mock_sql_chain():
    with patch('backend.utils.SQLDatabaseChain.from_llm') as mock_chain:
        mock_instance = AsyncMock()
        mock_instance.invoke = AsyncMock(return_value={
            "result": "Mocked SQL result",
            "intermediate_steps": []
        })
        mock_chain.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_chat_openai():
    with patch('backend.utils.ChatOpenAI') as mock:
        mock_instance = AsyncMock(spec=Runnable)
        mock_instance.ainvoke = AsyncMock(return_value=AIMessage(content="Mocked response content"))
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_sql_database():
    with patch('backend.utils.SQLDatabase.from_uri') as mock:
        mock_instance = MockSQLDatabase()
        mock.return_value = mock_instance
        yield mock

@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all dependencies used in the utils module."""
    with patch('backend.utils.SchemaVectorizer') as mock_schema, \
         patch('backend.utils.get_openai_client') as mock_openai, \
         patch('backend.utils.ChatOpenAI') as mock_chat, \
         patch('backend.utils.PromptTemplate') as mock_prompt, \
         patch('backend.utils.SQLDatabase.from_uri') as mock_db, \
         patch('backend.utils.SQLDatabaseChain.from_llm') as mock_chain:
        
        # Configure SchemaVectorizer mock
        mock_schema_instance = MagicMock()
        mock_schema_instance.get_relevant_schema.return_value = "mocked schema"
        mock_schema.return_value = mock_schema_instance
        
        # Configure OpenAI mock
        mock_openai_instance = AsyncMock()
        mock_openai_instance.chat.completions.create = AsyncMock(return_value=ChatCompletion(
            id="chatcmpl-123",
            choices=[{
                "index": 0,
                "message": ChatCompletionMessage(
                    role="assistant",
                    content="Mocked response content"
                ),
                "finish_reason": "stop"
            }],
            created=1234567890,
            model="gpt-4",
            object="chat.completion",
            usage={"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30}
        ))
        mock_openai.return_value = mock_openai_instance
        
        # Configure ChatOpenAI mock
        mock_chat_instance = AsyncMock(spec=Runnable)
        mock_chat_instance.ainvoke = AsyncMock(return_value=AIMessage(content="Mocked response content"))
        mock_chat.return_value = mock_chat_instance
        
        # Configure PromptTemplate mock
        mock_prompt_instance = MagicMock()
        mock_prompt_instance.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chat_instance)
        mock_prompt.return_value = mock_prompt_instance
        
        # Configure SQLDatabase mock
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance
        
        # Configure SQLDatabaseChain mock
        mock_chain_instance = AsyncMock()
        mock_chain_instance.invoke = AsyncMock(return_value={
            "result": "Mocked SQL result",
            "intermediate_steps": []
        })
        mock_chain.return_value = mock_chain_instance
        
        yield mock_schema_instance, mock_openai_instance, mock_chat_instance, mock_prompt_instance, mock_db_instance, mock_chain_instance

@pytest.fixture(autouse=True)
def mock_chain():
    with patch('backend.utils.SQLDatabaseChain') as mock_chain:
        mock_instance = MagicMock()
        mock_instance.from_llm.return_value = mock_instance
        mock_chain.return_value = mock_instance
        yield mock_instance

@pytest.mark.asyncio
async def test_get_sql_chain(mock_sql_database, mock_chat_openai):
    """Test that SQL chain is created with correct parameters."""
    chain = await get_sql_chain()
    mock_sql_database.assert_called_once()
    assert isinstance(chain, AsyncMock)

@pytest.mark.asyncio
async def test_generate_column_query(mock_dependencies):
    """Test column query generation."""
    table_name = "medical_code_service_specialty_relations"
    result = await generate_column_query(table_name)
    assert result == "Mocked SQL result"

@pytest.mark.asyncio
async def test_generate_table_query(mock_dependencies):
    """Test table query generation."""
    query = "medical"
    result = await generate_table_query(query)
    assert result == "Mocked SQL result"

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_column_query(mock_sql_chain):
    """Test prompt refinement for column queries."""
    query = "what columns are in medical_code_service_specialty_relations"
    result = await refine_prompt_with_ai(query)
    
    mock_sql_chain.invoke.assert_called_once()
    assert result == "Mocked SQL result"

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_table_query(mock_sql_chain):
    """Test prompt refinement for table queries."""
    query = "show me tables with medical in the name"
    result = await refine_prompt_with_ai(query)
    
    mock_sql_chain.invoke.assert_called_once()
    assert result == "Mocked SQL result"

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_error_handling():
    """Test error handling in prompt refinement."""
    test_query = "show me all users"

    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock(side_effect=Exception("Test error"))

    with patch("backend.utils.create_db_chain_with_schema", return_value=mock_chain):
        result = await refine_prompt_with_ai(test_query)
        assert result is None

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_table_query_fallback():
    """Test fallback behavior for table name queries when OpenAI fails."""
    query = "show me tables with medical in the name"
    expected_sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name LIKE '%medical%';"

    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock(side_effect=Exception("Test error"))

    with patch("backend.utils.create_db_chain_with_schema", return_value=mock_chain), \
         patch("backend.utils.generate_table_query", return_value=expected_sql):
        result = await refine_prompt_with_ai(query)
        assert result == expected_sql
        assert mock_chain.ainvoke.call_count == 1
        mock_chain.ainvoke.assert_called_once_with({"query": query, "schema": ""})

@pytest.mark.asyncio
async def test_generate_column_query_error_handling(mock_dependencies):
    """Test error handling in column query generation."""
    mock_schema, mock_openai, mock_chat, mock_prompt, mock_db, mock_chain = mock_dependencies
    
    # Configure SQLDatabaseChain mock to raise an error
    mock_chain.invoke = AsyncMock(side_effect=Exception("Test error"))
    
    result = await generate_column_query("test_table")
    assert result is None

@pytest.mark.asyncio
async def test_generate_table_query_error_handling(mock_dependencies):
    """Test error handling in table query generation."""
    mock_schema, mock_openai, mock_chat, mock_prompt, mock_db, mock_chain = mock_dependencies
    
    # Configure SQLDatabaseChain mock to raise an error
    mock_chain.invoke.side_effect = Exception("Test error")
    
    result = await generate_table_query("test")
    assert result is None  # Should return None on error

@pytest.mark.asyncio
async def test_generate_table_query_success(mock_dependencies):
    """Test successful table query generation."""
    mock_schema, mock_openai, mock_chat, mock_prompt, mock_db, mock_chain = mock_dependencies
    
    # Configure mock response
    mock_chain.invoke.return_value = {"result": "test_table"}
    
    result = await generate_table_query("test")
    assert result == "test_table"
    mock_chain.invoke.assert_called_once()

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_success():
    """Test successful prompt refinement."""
    test_query = "show me all users"
    expected_sql = "SELECT * FROM users;"

    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock(return_value={"result": expected_sql})

    with patch("backend.utils.create_db_chain_with_schema", return_value=mock_chain):
        result = await refine_prompt_with_ai(test_query)
        assert result == expected_sql

@pytest.mark.asyncio
async def test_sanitize_sql_response_success():
    """Test successful sanitization of SQL response."""
    test_response = "SELECT name, email FROM users WHERE id=123;"
    expected_sanitized = "SELECT [PRIVATE], [PRIVATE] FROM users WHERE id=123;"

    mock_chat = AsyncMock()
    mock_chat.agenerate = AsyncMock(return_value=Mock(generations=[[Mock(content=expected_sanitized)]]))

    with patch("backend.utils.ChatOpenAI", return_value=mock_chat):
        result = await sanitize_sql_response(test_response)
        assert result == expected_sanitized

@pytest.mark.asyncio
async def test_sanitize_sql_response_sql_result():
    """Test sanitization of SQL query result."""
    test_response = "SELECT * FROM users;"
    expected_sanitized = "SELECT * FROM users;"

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=Mock(content=expected_sanitized))

    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock(return_value=Mock(content=expected_sanitized))

    with patch("backend.utils.ChatOpenAI", return_value=mock_llm), \
         patch("backend.utils.ChatPromptTemplate.from_messages", return_value=mock_chain):
        result = await sanitize_sql_response(test_response)
        assert result == expected_sanitized

@pytest.mark.asyncio
async def test_sanitize_sql_response_error():
    """Test error handling in SQL response sanitization."""
    test_response = "SELECT * FROM users;"

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("Test error"))

    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock(side_effect=Exception("Test error"))

    with patch("backend.utils.ChatOpenAI", return_value=mock_llm), \
         patch("backend.utils.ChatPromptTemplate.from_messages", return_value=mock_chain):
        result = await sanitize_sql_response(test_response)
        assert result == test_response  # Should return original on error

@pytest.mark.asyncio
async def test_sanitize_sql_response_non_ai_message():
    """Test handling of non-AIMessage responses in SQL sanitization."""
    test_response = "SELECT * FROM users;"

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value="Not an AIMessage")

    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock(return_value="Not an AIMessage")

    with patch("backend.utils.ChatOpenAI", return_value=mock_llm), \
         patch("backend.utils.ChatPromptTemplate.from_messages", return_value=mock_chain):
        result = await sanitize_sql_response(test_response)
        assert result == test_response  # Should return original if not AIMessage

if __name__ == "__main__":
    pytest.main([__file__])
