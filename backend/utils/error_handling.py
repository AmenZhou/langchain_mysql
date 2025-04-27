from typing import Any, Awaitable
from openai import OpenAIError

async def handle_openai_error(coroutine: Awaitable[Any]) -> Any:
    """
    Handle OpenAI API errors by propagating them with appropriate error messages.
    
    Args:
        coroutine: The OpenAI API coroutine to execute
        
    Returns:
        The result of the coroutine if successful
        
    Raises:
        OpenAIError: Propagates any OpenAI-related errors with appropriate context
    """
    try:
        return await coroutine
    except OpenAIError as e:
        # Log the error here if needed
        raise  # Re-raise the original error to maintain the error context 
