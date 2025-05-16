from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal
from enum import Enum

class ResponseType(str, Enum):
    """Enum for response types."""
    SQL = "sql"
    DATA = "data"
    NATURAL_LANGUAGE = "natural_language"
    ALL = "all"

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    session_id: Optional[str] = Field(default=None, description="Optional session ID for maintaining conversation history")
    prompt_type: Optional[str] = None
    response_type: ResponseType = Field(
        default=ResponseType.ALL,
        description="Type of response to return: sql (just the SQL query), data (execute SQL and return data), natural_language (return natural language explanation), or all (return everything)"
    )

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    result: Any = Field(description="Primary result based on requested response_type")
    sql: Optional[str] = Field(default=None, description="Generated SQL query")
    data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Data returned from executing the SQL query")
    explanation: Optional[str] = Field(default=None, description="Natural language explanation of the result")
    response_type: ResponseType = Field(default=ResponseType.ALL, description="Type of response that was returned")

class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    status_code: int = 500

class SchemaInfo(BaseModel):
    """Schema information model."""
    tables: List[str]
    relationships: List[Dict[str, str]]
