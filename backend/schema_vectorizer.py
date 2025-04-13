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
        
        # Create a human-readable description
        description = f"Table: {table_name}\n\n"
        
        # Add column information
        description += "Columns:\n"
        for col in schema_info.get("columns", []):
            pk_indicator = "[PK]" if col.get("primary_key") else ""
            nullable = "NULL" if col.get("nullable") else "NOT NULL"
            default = f"DEFAULT {col.get('default')}" if col.get("default") != "None" else ""
            description += f"- {col.get('name')} {pk_indicator}: {col.get('type')} {nullable} {default}\n"
        
        # Add foreign key information
        fks = schema_info.get("foreign_keys", [])
        if fks:
            description += "\nForeign Keys:\n"
            for fk in fks:
                ref_table = fk.get("referred_table")
                ref_cols = fk.get("referred_columns", [])
                constrained_cols = fk.get("constrained_columns", [])
                ref_cols_str = ", ".join(ref_cols)
                constrained_cols_str = ", ".join(constrained_cols)
                description += f"- {constrained_cols_str} REFERENCES {ref_table}({ref_cols_str})\n"
        
        # Create metadata with the raw schema information
        metadata = {
            "table_name": table_name,
            "schema": json.dumps(schema_info)
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

def get_schema_as_text(query: Optional[str] = None, k: int = 5) -> str:
    """Get schema information as formatted text, optionally filtered by a query."""
    try:
        if query:
            # If query is provided, search for relevant schema info
            results = query_schema_vectordb(query, k=k)
            schema_text = "\n\n".join([doc.page_content for doc in results])
        else:
            # If no query, get all schema info
            schema_documents = create_schema_documents()
            schema_text = "\n\n".join([doc.page_content for doc in schema_documents])
        
        return schema_text
    
    except Exception as e:
        logger.error(f"Error getting schema as text: {e}")
        return f"Error retrieving schema information: {e}"

if __name__ == "__main__":
    # When run directly, preload the schema
    preload_schema_to_vectordb() 
