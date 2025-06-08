import pytest
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_config import (
    create_db_chain_with_schema,
    get_table_query_prompt,
    get_column_query_prompt,
    get_sanitize_prompt,
    get_relevant_prompt,
    backoff_with_jitter,
)
from prompts import PROMPT_REFINE, PROMPT_TABLE_QUERY

@pytest.fixture
def mock_chat_model():
    """Create a mock ChatOpenAI instance."""
    # Create a real ChatOpenAI instance with a dummy key for mocking
    from langchain_config import CachedChatOpenAI
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy-key'}):
        mock_llm = CachedChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        yield mock_llm

@pytest.fixture
def mock_embeddings():
    """Create a mock OpenAIEmbeddings instance."""
    with patch('vector_store.vector_store.OpenAIEmbeddings') as mock:
        mock_instance = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_faiss():
    """Create a mock FAISS instance."""
    with patch('langchain_community.vectorstores.FAISS') as mock:
        mock_instance = AsyncMock()
        mock.from_texts.return_value = mock_instance
        mock.return_value = mock_instance
        yield mock_instance

@pytest.mark.asyncio
async def test_create_db_chain_with_schema(mock_chat_model, mock_embeddings, mock_faiss):
    """Test creating a database chain with schema."""
    schema_info = "Test schema information"
    memory = ConversationBufferMemory()
    # Just test that the chain is created successfully without mocking internal methods
    chain = await create_db_chain_with_schema(schema_info, memory, llm=mock_chat_model)
    assert chain is not None
    assert hasattr(chain, 'llm')
    assert hasattr(chain, 'memory')

def test_get_table_query_prompt():
    """Test getting table query prompt."""
    prompt = get_table_query_prompt("find user tables")
    assert prompt is not None
    assert "find user tables" in prompt

def test_get_column_query_prompt():
    """Test getting column query prompt."""
    prompt = get_column_query_prompt("show columns in users")
    assert prompt is not None
    assert "show columns in users" in prompt

def test_get_sanitize_prompt():
    """Test getting sanitize prompt."""
    test_sql = "SELECT * FROM users"
    prompt = get_sanitize_prompt(test_sql)
    assert "sanitize" in prompt.lower()
    assert test_sql in prompt

@pytest.mark.asyncio
async def test_create_db_chain_with_schema_error_handling(mock_chat_model):
    """Test error handling in create_db_chain_with_schema."""
    memory = ConversationBufferMemory()
    with patch('langchain_config.PromptTemplate') as mock_prompt:
        mock_prompt.side_effect = Exception("Chain creation error")
        with pytest.raises(Exception, match="Failed to create database chain"):
            await create_db_chain_with_schema("test schema", memory, llm=mock_chat_model)

def test_get_table_query_prompt_error_handling():
    """Test error handling in get_table_query_prompt."""
    with patch('langchain_config.get_relevant_prompt', side_effect=Exception("Vector search error")):
        prompt = get_table_query_prompt("find tables")
        assert prompt is not None
        assert "find tables" in prompt

def test_get_column_query_prompt_error_handling():
    """Test error handling in get_column_query_prompt."""
    with patch('langchain_config.get_relevant_prompt', side_effect=Exception("Vector search error")):
        prompt = get_column_query_prompt("show columns")
        assert prompt is not None
        assert "show columns" in prompt

def test_get_sanitize_prompt_error_handling():
    """Test error handling in get_sanitize_prompt."""
    test_sql = "SELECT * FROM users"
    with patch('langchain_config.get_relevant_prompt', side_effect=Exception("Vector search error")):
        prompt = get_sanitize_prompt(test_sql)
        assert test_sql in prompt
