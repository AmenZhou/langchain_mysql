import os
from typing import Dict, List, Optional, Any
import logging
import asyncio
import sys

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from sqlalchemy.exc import SQLAlchemyError

from .utils.error_handling import handle_openai_error
from .database import get_db_engine
from .schema_extractor import SchemaExtractor
from .vector_store import VectorStoreManager

from sqlalchemy import inspect, MetaData
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from .prompts import PROMPT_REFINE, PROMPT_TABLE_QUERY, get_sanitize_prompt

# Configure logging
logger = logging.getLogger(__name__)

class SchemaVectorizer:
    def __init__(self, db_url: str, openai_api_key: Optional[str] = None):
        """Initialize SchemaVectorizer with database URL and OpenAI API key.

        Args:
            db_url (str): Database connection URL
            openai_api_key (Optional[str]): OpenAI API key. If not provided, will try to get from environment.

        Raises:
            ValueError: If OpenAI API key is not provided and not found in environment
            SQLAlchemyError: If database connection fails
            Exception: For other initialization errors
        """
        try:
            self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided either directly or via OPENAI_API_KEY environment variable")

            self.db_url = db_url
            self.engine = get_db_engine(db_url)
            self.schema_extractor = SchemaExtractor(self.engine)
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            self.vector_store_manager = VectorStoreManager(self.embeddings)
            
            logger.info("SchemaVectorizer initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error initializing SchemaVectorizer: {str(e)}")
            raise

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
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except SQLAlchemyError as e:
            logger.error(f"Error getting table names: {e}")
            return []

    async def extract_table_schema(self) -> Dict[str, Any]:
        """Extract schema information from the database.

        Returns:
            Dict[str, Any]: Dictionary containing table schema information

        Raises:
            SQLAlchemyError: If database schema extraction fails
            Exception: For other unexpected errors
        """
        try:
            schema_info = await self.schema_extractor.extract_schema()
            logger.info("Successfully extracted database schema")
            return schema_info
        except SQLAlchemyError as e:
            logger.error(f"Error extracting database schema: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during schema extraction: {str(e)}")
            raise

    def create_schema_documents(self, schema_info: Dict[str, Any]) -> List[Document]:
        """Create Document objects from schema information.

        Args:
            schema_info (Dict[str, Any]): Dictionary containing table schema information

        Returns:
            List[Document]: List of Document objects representing schema information

        Raises:
            ValueError: If schema_info is invalid
            Exception: For other document creation errors
        """
        try:
            if not schema_info:
                raise ValueError("Schema information cannot be empty")

            documents = []
            for table_name, table_info in schema_info.items():
                doc = Document(
                    page_content=str(table_info),
                    metadata={"table_name": table_name}
                )
                documents.append(doc)

            logger.info(f"Created {len(documents)} schema documents")
            return documents
        except Exception as e:
            logger.error(f"Error creating schema documents: {str(e)}")
            raise

    def create_prompt_documents(self) -> List[Document]:
        """Create Document objects for prompts to be embedded."""
        return self.schema_extractor.create_prompt_documents()

    async def initialize_vector_store(self) -> None:
        """Initialize vector store with schema information.

        Raises:
            SQLAlchemyError: If database schema extraction fails
            ValueError: If document creation fails
            Exception: For other vector store initialization errors
        """
        try:
            schema_info = await self.extract_table_schema()
            documents = self.create_schema_documents(schema_info)
            
            await self.vector_store_manager.add_documents(documents)
            logger.info("Successfully initialized vector store with schema information")
        except (SQLAlchemyError, ValueError) as e:
            logger.error(f"Error during vector store initialization: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing vector store: {str(e)}")
            raise

    async def get_relevant_prompt(self, query: str, prompt_type: Optional[str] = None) -> str:
        """Retrieve the most relevant prompt from the vector database.

        Args:
            query (str): The query to find relevant prompts for
            prompt_type (Optional[str]): Type of prompt to retrieve (e.g., 'refine', 'table', 'sanitize')

        Returns:
            str: The most relevant prompt text

        Raises:
            ValueError: If no prompt is found for the given type
            Exception: For other unexpected errors
        """
        try:
            results = await handle_openai_error(self.vector_store_manager.query_prompts(query, prompt_type))
            
            if not results:
                # Fallback to default prompts if no match found
                if prompt_type == "refine":
                    return PROMPT_REFINE
                elif prompt_type == "table":
                    return PROMPT_TABLE_QUERY
                elif prompt_type == "sanitize":
                    return get_sanitize_prompt(query)
                else:
                    raise ValueError(f"No prompt found for type: {prompt_type}")
                    
            return results[0].page_content
        except Exception as e:
            logger.error(f"Error retrieving relevant prompt: {str(e)}")
            # If any error occurs, fall back to default prompts
            if prompt_type == "refine":
                return PROMPT_REFINE
            elif prompt_type == "table":
                return PROMPT_TABLE_QUERY
            elif prompt_type == "sanitize":
                return get_sanitize_prompt(query)
            else:
                raise ValueError(f"No prompt found for type: {prompt_type}")

    async def get_relevant_schema(self, query: Optional[str] = None, k: int = 1) -> str:
        """Get schema information as formatted text, optionally filtered by a query.

        Args:
            query (Optional[str]): Query to filter schema information
            k (int): Number of most relevant schema entries to return

        Returns:
            str: Formatted schema information text

        Raises:
            Exception: For unexpected errors during schema retrieval
        """
        try:
            if not query:
                return ""

            # Try vector search first
            try:
                results = await handle_openai_error(self.vector_store_manager.query_schema(query, k=k))
                if results:
                    return results[0].page_content
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to full schema: {str(e)}")
                
            # Fallback to full schema if vector search fails
            schema_info = await self.extract_table_schema()
            if not schema_info:
                return ""
                
            # Format schema info as text
            schema_text = []
            for table_name, info in schema_info.items():
                schema_text.append(f"Table: {table_name}")
                if 'columns' in info:
                    schema_text.append("Columns: " + ", ".join(info['columns']))
                if 'description' in info:
                    schema_text.append(info['description'])
                schema_text.append("")  # Add blank line between tables
                
            return "\n".join(schema_text)
            
        except Exception as e:
            logger.error(f"Error getting relevant schema: {str(e)}")
            return ""

if __name__ == "__main__":
    # When run directly, initialize the vector store
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL environment variable must be set")
            
        vectorizer = SchemaVectorizer(db_url=db_url)
        asyncio.run(vectorizer.initialize_vector_store())
        logger.info("Successfully initialized vector store")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        sys.exit(1) 
