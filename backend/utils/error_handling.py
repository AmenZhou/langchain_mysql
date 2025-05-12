from fastapi import HTTPException
from sqlalchemy.exc import ProgrammingError, OperationalError
from openai import RateLimitError, APIError, APIStatusError
from typing import Union, Optional

def handle_sql_error(error: Exception) -> HTTPException:
    """Handle SQL-related errors and return appropriate HTTP exceptions."""
    error_msg = str(error).lower()
    
    if isinstance(error, ProgrammingError):
        if "relation" in error_msg and "does not exist" in error_msg:
            return HTTPException(
                status_code=422,
                detail="Table does not exist"
            )
        elif "permission" in error_msg or "access denied" in error_msg:
            return HTTPException(
                status_code=403,
                detail="Permission denied"
            )
        else:
            return HTTPException(
                status_code=422,
                detail="Invalid SQL syntax"
            )
    elif isinstance(error, OperationalError):
        if "permission" in error_msg or "access denied" in error_msg:
            return HTTPException(
                status_code=403,
                detail="Permission denied"
            )
        return HTTPException(
            status_code=422,
            detail="Database error"
        )
    else:
        return HTTPException(
            status_code=422,
            detail="Database error"
        )

def handle_error(error: Exception) -> HTTPException:
    """Handle general application errors and return appropriate HTTP exceptions."""
    if isinstance(error, HTTPException):
        return error
    elif isinstance(error, RateLimitError):
        return HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )
    elif isinstance(error, (APIError, APIStatusError)):
        if getattr(error, 'status_code', 500) == 429:
            return HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        return HTTPException(
            status_code=422,
            detail=f"OpenAI API error: {str(error)}"
        )
    elif isinstance(error, ValueError):
        return HTTPException(
            status_code=422,
            detail=str(error)
        )
    else:
        return HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(error)}"
        )

def validate_query(query: Optional[str]) -> None:
    """Validate the query string."""
    if not query or not isinstance(query, str) or not query.strip():
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty"
        )

async def handle_openai_error(coro) -> Union[HTTPException, any]:
    """Handle OpenAI API errors in async functions."""
    if coro is None:
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty"
        )
        
    try:
        return await coro
    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )
    except (APIError, APIStatusError) as e:
        if getattr(e, 'status_code', 500) == 429:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        raise HTTPException(
            status_code=422,
            detail=f"OpenAI API error: {str(e)}"
        )
    except ProgrammingError as e:
        error_msg = str(e).lower()
        if "relation" in error_msg and "does not exist" in error_msg:
            raise HTTPException(
                status_code=422,
                detail="Table does not exist"
            )
        elif "permission" in error_msg or "access denied" in error_msg:
            raise HTTPException(
                status_code=403,
                detail="Permission denied"
            )
        else:
            raise HTTPException(
                status_code=422,
                detail="Invalid SQL syntax"
            )
    except OperationalError as e:
        error_msg = str(e).lower()
        if "permission" in error_msg or "access denied" in error_msg:
            raise HTTPException(
                status_code=403,
                detail="Permission denied"
            )
        elif "connection" in error_msg or "database" in error_msg:
            raise HTTPException(
                status_code=500,
                detail="Database connection error"
            )
        else:
            raise HTTPException(
                status_code=422,
                detail="Database error"
            )
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        ) 
