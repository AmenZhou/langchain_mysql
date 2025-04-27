from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .langchain_mysql import LangChainMySQL, lifespan
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    prompt_type: Optional[str] = None

# Create LangChain MySQL instance
langchain_mysql = LangChainMySQL()

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a natural language query and return SQL results."""
    return await langchain_mysql.process_query(request.query, request.prompt_type)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
