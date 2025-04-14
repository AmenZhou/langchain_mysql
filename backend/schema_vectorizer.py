from sqlalchemy import inspect, MetaData
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import json
import os
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import List, Dict, Any, Optional

from backend.database import engine, db
from backend.included_tables import INCLUDED_TABLES

# Configure logging
logger = logging.getLogger(__name__)

# Set up the embedding model
def get_embeddings():
    """Initialize and return the OpenAI embeddings model."""
    return OpenAIEmbeddings()

def get_chroma_db(persist_directory: str = "./chroma_db"):
    """Initialize and return the Chroma vector database."""
    embeddings = get_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def extract_table_schema(table_name: str) -> Dict[str, Any]:
    """Extract schema information for a specific table."""
    try:
        inspector = inspect(engine)
        
        # Get columns
        columns = inspector.get_columns(table_name)
        column_info = []
        for column in columns:
            column_info.append({
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": column.get("nullable", True),
                "default": str(column.get("default", "None")),
                "primary_key": column.get("primary_key", False)
            })
        
        # Get primary key
        pk = inspector.get_pk_constraint(table_name)
        
        # Get foreign keys
        fks = inspector.get_foreign_keys(table_name)
        
        # Get indexes
        indexes = inspector.get_indexes(table_name)
        
        # Combine all schema information
        schema_info = {
            "table_name": table_name,
            "columns": column_info,
            "primary_key": pk,
            "foreign_keys": fks,
            "indexes": indexes
        }
        
        return schema_info
    
    except SQLAlchemyError as e:
        logger.error(f"Error extracting schema for table {table_name}: {e}")
        return {"table_name": table_name, "error": str(e)}

def create_schema_documents(tables: List[str] = None) -> List[Document]:
    """Create Document objects for each table schema to be embedded."""
    if tables is None:
        tables = INCLUDED_TABLES
    
    documents = []
    
    for table_name in tables:
        schema_info = extract_table_schema(table_name)
        
        # Create an extremely minimal description
        description = f"{table_name}:"
        
        # Add only column names, skip types and other details
        columns = [col.get('name') for col in schema_info.get("columns", [])]
        
        # Join columns with spaces to save even more space
        description += " ".join(columns)
        
        # Create metadata with minimal schema information
        metadata = {
            "table_name": table_name,
            "columns": [col.get("name") for col in schema_info.get("columns", [])]
        }
        
        # Create the document
        doc = Document(page_content=description, metadata=metadata)
        documents.append(doc)
    
    return documents

def preload_schema_to_vectordb(persist_directory: str = "./chroma_db") -> None:
    """Extract database schema, create embeddings, and store in vector database."""
    try:
        # Create documents from schema
        schema_documents = create_schema_documents()
        
        # Check if we have documents to embed
        if not schema_documents:
            logger.warning("No schema documents created. Vector database not updated.")
            return
        
        # Create vector store with schema documents
        embeddings = get_embeddings()
        vectordb = Chroma.from_documents(
            documents=schema_documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Persist to disk
        vectordb.persist()
        
        logger.info(f"Successfully preloaded schema for {len(schema_documents)} tables to vector database")
        return vectordb
    
    except Exception as e:
        logger.error(f"Error preloading schema to vector database: {e}")
        raise

def query_schema_vectordb(query: str, persist_directory: str = "./chroma_db", k: int = 3) -> List[Document]:
    """Query the vector database for schema information."""
    try:
        # Load the vector database
        vectordb = get_chroma_db(persist_directory)
        
        # Perform similarity search
        results = vectordb.similarity_search(query, k=k)
        
        return results
    
    except Exception as e:
        logger.error(f"Error querying schema vector database: {e}")
        raise

def get_schema_as_text(query: Optional[str] = None, k: int = 1) -> str:
    """Get schema information as formatted text, optionally filtered by a query."""
    try:
        if query:
            # If query is provided, search for relevant schema info
            results = query_schema_vectordb(query, k=k)
            # Only include essential information for each table
            schema_texts = []
            for doc in results:
                # Use the minimal format directly
                schema_texts.append(doc.page_content)
            schema_text = "\n".join(schema_texts)
        else:
            # If no query, get all schema info but limit to essential information
            schema_documents = create_schema_documents()
            schema_texts = []
            for doc in schema_documents:
                schema_texts.append(doc.page_content)
            schema_text = "\n".join(schema_texts)
        
        return schema_text
    
    except Exception as e:
        logger.error(f"Error getting schema as text: {e}")
        return f"Error retrieving schema information: {e}"

if __name__ == "__main__":
    # When run directly, preload the schema
    preload_schema_to_vectordb() 
