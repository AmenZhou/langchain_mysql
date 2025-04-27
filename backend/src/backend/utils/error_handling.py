from fastapi import HTTPException
from sqlalchemy.exc import ProgrammingError, OperationalError
from openai import RateLimitError, APIError

def handle_sql_error(error: Exception) -> HTTPException:
    """Handle SQL-related errors and return appropriate HTTP exceptions."""
    if isinstance(error, ProgrammingError):
        if "relation" in str(error) and "does not exist" in str(error):
            return HTTPException(
                status_code=422,
                detail=f"Table not found: {str(error)}"
            )
        else:
            return HTTPException(
                status_code=422,
                detail=f"SQL syntax error: {str(error)}"
            )
    elif isinstance(error, OperationalError):
        return HTTPException(
            status_code=422,
            detail=f"Database operation error: {str(error)}"
        )
    else:
        return HTTPException(
            status_code=500,
            detail=f"Unexpected database error: {str(error)}"
        )

def handle_error(error: Exception) -> HTTPException:
    """Handle general application errors and return appropriate HTTP exceptions."""
    if isinstance(error, RateLimitError):
        return HTTPException(
            status_code=500,
            detail="OpenAI API rate limit exceeded. Please try again later."
        )
    elif isinstance(error, ValueError):
        return HTTPException(
            status_code=422,
            detail=str(error)
        )
    else:
        return HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(error)}"
        )

def validate_query(query: str) -> None:
    """Validate the query string."""
    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty"
        )
    if not isinstance(query, str):
        raise HTTPException(
            status_code=422,
            detail="Query must be a string"
        )

async def handle_openai_error(coro):
    """Handle OpenAI API errors in async functions.
    
    Args:
        coro: The coroutine to execute
        
    Returns:
        The result of the coroutine if successful
        
    Raises:
        HTTPException: If an OpenAI API error occurs
    """
    try:
        return await coro
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="OpenAI API rate limit exceeded. Please try again later."
        )
    except APIError as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling OpenAI API: {str(e)}"
        ) 
