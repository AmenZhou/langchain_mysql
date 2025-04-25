import pytest
import os
from dotenv import load_dotenv

pytest_plugins = ["pytest_asyncio"]

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Set test-specific environment variables
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    
    yield
    
    # Clean up after tests
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"] 
