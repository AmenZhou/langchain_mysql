from sqlalchemy import inspect, MetaData
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
import json
import os
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .database import engine, db

# Configure logging
logger = logging.getLogger(__name__)

class SchemaVectorizer:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = self._get_embeddings()
        self.vectordb = None
        
    def _get_embeddings(self):
        """Initialize and return the OpenAI embeddings model."""
        return OpenAIEmbeddings()

    def _get_chroma_db(self):
        """Initialize and return the Chroma vector database."""
        if not self.vectordb:
            self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        return self.vectordb

    def get_all_tables(self) -> List[str]:
        """Get all table names from the database."""
        try:
            inspector = inspect(engine)
            return inspector.get_table_names()
        except SQLAlchemyError as e:
            logger.error(f"Error getting table names: {e}")
            return []

    def extract_table_schema(self, table_name: str) -> Dict[str, Any]:
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

    def create_schema_documents(self) -> List[Document]:
        """Create Document objects for each table schema to be embedded."""
        tables = self.get_all_tables()
        if not tables:
            logger.warning("No tables found in the database.")
            return []
        
        documents = []
        
        for table_name in tables:
            schema_info = self.extract_table_schema(table_name)
            
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

    def preload_schema_to_vectordb(self) -> None:
        """Extract database schema, create embeddings, and store in vector database."""
        try:
            # Create documents from schema
            schema_documents = self.create_schema_documents()
            
            # Check if we have documents to embed
            if not schema_documents:
                logger.warning("No schema documents created. Vector database not updated.")
                return
            
            try:
                # Try to create and persist vector store
                self.vectordb = Chroma.from_documents(
                    documents=schema_documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                self.vectordb.persist()
                logger.info(f"Successfully preloaded schema for {len(schema_documents)} tables to vector database")
            except Exception as e:
                logger.warning(f"Failed to create vector store: {e}. Will use fallback mechanism.")
            
            return schema_documents
        
        except Exception as e:
            logger.error(f"Error preloading schema to vector database: {e}")
            raise

    def query_schema_vectordb(self, query: str, k: int = 3) -> List[Document]:
        """Query the vector database for schema information."""
        try:
            # Load the vector database
            vectordb = self._get_chroma_db()
            
            # Perform similarity search
            results = vectordb.similarity_search(query, k=k)
            
            return results
        
        except Exception as e:
            logger.error(f"Error querying schema vector database: {e}")
            raise

    def get_relevant_schema(self, query: Optional[str] = None, k: int = 1) -> str:
        """Get schema information as formatted text, optionally filtered by a query."""
        try:
            # Create schema documents directly
            schema_documents = self.create_schema_documents()
            if not schema_documents:
                logger.warning("No schema documents created.")
                return ""

            # If no query, return all schema info
            if not query:
                schema_texts = []
                for doc in schema_documents:
                    schema_texts.append(doc.page_content)
                return "\n".join(schema_texts)

            # If query provided, try vector search first
            try:
                results = self.query_schema_vectordb(query, k=k)
                schema_texts = [doc.page_content for doc in results]
                return "\n".join(schema_texts)
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to full schema: {e}")
                # Simple keyword matching as fallback
                matched_docs = []
                query_terms = query.lower().split()
                for doc in schema_documents:
                    content = doc.page_content.lower()
                    if any(term in content for term in query_terms):
                        matched_docs.append(doc)
                if matched_docs:
                    return "\n".join(doc.page_content for doc in matched_docs[:k])
                # If no matches, return all schema info
                return "\n".join(doc.page_content for doc in schema_documents)
        
        except Exception as e:
            logger.error(f"Error getting schema as text: {e}")
            return f"Error retrieving schema information: {e}"

if __name__ == "__main__":
    # When run directly, preload the schema
    vectorizer = SchemaVectorizer()
    vectorizer.preload_schema_to_vectordb() 
