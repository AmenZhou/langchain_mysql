from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    prompt_type: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    result: Any
    sql: Optional[str] = None
    explanation: Optional[str] = None

class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    status_code: int = 500

class SchemaInfo(BaseModel):
    """Schema information model."""
    tables: List[str]
    relationships: List[Dict[str, str]]
