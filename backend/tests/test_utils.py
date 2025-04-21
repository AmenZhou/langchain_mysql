import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy import MetaData, inspect
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from langchain_core.outputs import ChatResult
from langchain_core.messages import AIMessage

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

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
    os.environ["OPENAI_API_KEY"] = "test-key"
    yield
    del os.environ["OPENAI_API_KEY"]

@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all dependencies used in the utils module."""
    with patch('backend.utils.SchemaVectorizer', return_value=mock_schema_vectorizer), \
         patch('backend.utils.get_openai_client', return_value=mock_openai), \
         patch('backend.utils.ChatOpenAI', return_value=mock_chain), \
         patch('backend.utils.PromptTemplate', mock_prompt_template):
        yield

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_success():
    """Test successful prompt refinement."""
    result = await refine_prompt_with_ai("test prompt")
    
    assert result == "Mocked response content"
    mock_schema_vectorizer.get_relevant_schema.assert_called_once()
    mock_openai.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_sanitize_sql_response_success():
    """Test successful SQL response sanitization."""
    result = await sanitize_sql_response("test response")
    
    assert result == "sanitized response"
    mock_prompt_template.from_messages.assert_called_once()
    mock_chain.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_schema_error():
    """Test schema error in prompt refinement."""
    mock_schema_vectorizer.get_relevant_schema.side_effect = Exception("Schema error")
    
    result = await refine_prompt_with_ai("test query")
    assert result is None
    
    mock_schema_vectorizer.get_relevant_schema.side_effect = None  # Reset for other tests

@pytest.mark.asyncio
async def test_refine_prompt_with_ai_openai_error():
    """Test OpenAI error in prompt refinement."""
    mock_openai.chat.completions.create.side_effect = Exception("OpenAI error")
    
    result = await refine_prompt_with_ai("test query")
    assert result is None
    
    mock_openai.chat.completions.create.side_effect = None  # Reset for other tests

@pytest.mark.asyncio
async def test_sanitize_sql_response_error():
    """Test error in SQL response sanitization."""
    mock_chain.ainvoke.side_effect = Exception("LLM error")
    
    result = await sanitize_sql_response("test response")
    assert result == "test response"
    
    mock_chain.ainvoke.side_effect = None  # Reset for other tests 
