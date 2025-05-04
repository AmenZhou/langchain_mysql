import os
from typing import Dict, List, Optional, Any
import logging
import asyncio
import sys

from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
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
    def __init__(self, db_url: str, openai_api_key: Optional[str] = None, persist_directory: str = "./chroma_db"):
        """Initialize SchemaVectorizer with database URL and OpenAI API key.

        Args:
            db_url (str): Database connection URL
            openai_api_key (Optional[str]): OpenAI API key. If not provided, will try to get from environment.
            persist_directory (str): Directory to persist vector store data

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
            os.environ["DATABASE_URL"] = db_url  # Set the environment variable for get_db_engine
            self.engine = get_db_engine()  # Call without parameters
            self.schema_extractor = SchemaExtractor(self.engine)
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            self.vector_store_manager = VectorStoreManager(
                persist_directory=persist_directory,
                embeddings=self.embeddings
            )
            
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
            schema_info = await self.schema_extractor.extract_table_schema()
            logger.info("Successfully extracted database schema")
            return schema_info
        except SQLAlchemyError as e:
            logger.error(f"Error extracting database schema: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during schema extraction: {str(e)}")
            raise

    def create_schema_documents(self, schema_info: Dict[str, Dict]) -> List[Document]:
        """Create documents from schema information."""
        documents = []
        for table_name, table_info in schema_info.items():
            columns = table_info.get('columns', [])
            description = table_info.get('description', '')
            foreign_keys = table_info.get('foreign_keys', [])
            
            # Create a detailed description of the table
            column_descriptions = []
            for col in columns:
                col_name = col.get('name', '')
                col_type = col.get('type', '')
                col_desc = col.get('description', '')
                column_descriptions.append(f"{col_name} ({col_type}) - {col_desc}")
            
            # Add foreign key relationships to the content - MAKE THIS MORE PROMINENT
            fk_section = ""
            if foreign_keys:
                fk_descriptions = []
                for fk in foreign_keys:
                    fk_descriptions.append(f"{fk['column']} references {fk['references']}")
                fk_section = f"\n\nFOREIGN KEY RELATIONSHIPS IN {table_name}:\n" + "\n".join(fk_descriptions)
            
            content = f"Table {table_name} contains:\n" + "\n".join(column_descriptions) + fk_section
            
            # Add additional foreign key context for semantic similarity
            if foreign_keys:
                foreign_key_context = "\n\nThis table has relationships with other tables through foreign keys."
                for fk in foreign_keys:
                    referred_table = fk['references'].split('.')[0]
                    foreign_key_context += f"\nThe column {fk['column']} in table {table_name} is linked to table {referred_table}."
                content += foreign_key_context
            
            documents.append(Document(
                page_content=content,
                metadata={
                    'table_name': table_name,
                    'columns': [col.get('name', '') for col in columns],
                    'foreign_keys': foreign_keys,
                    'description': description
                }
            ))
        return documents

    def create_prompt_documents(self) -> List[Document]:
        """Create Document objects for prompts to be embedded."""
        return self.schema_extractor.create_prompt_documents()

    async def initialize_vector_store(self, schema_info: Dict[str, Any]) -> None:
        """Initialize vector store with schema information.

        Args:
            schema_info (Dict[str, Any]): Schema information to initialize vector store with

        Raises:
            SQLAlchemyError: If database schema extraction fails
            ValueError: If document creation fails
            Exception: For other vector store initialization errors
        """
        try:
            logger.info("Starting vector store initialization")
            
            if not schema_info:
                logger.error("No schema information provided for initialization")
                raise ValueError("Schema information cannot be empty")
                
            logger.info(f"Schema info contains {len(schema_info)} tables: {list(schema_info.keys())}")
            logger.info(f"Creating documents from schema info")
            documents = self.create_schema_documents(schema_info)
            logger.info(f"Created {len(documents)} documents")
            
            if not documents:
                logger.error("No documents created from schema info")
                raise ValueError("Failed to create documents from schema information")
                
            logger.info("Adding documents to vector store")
            await self.vector_store_manager.add_documents(documents)
            logger.info("Successfully initialized vector store with schema information")
            
            # Verify the vector store was initialized
            if not self.vector_store_manager.schema_vectordb:
                logger.error("Vector store was not properly initialized")
                raise ValueError("Vector store was not properly initialized")
                
            logger.info("Vector store initialization complete")
        except (SQLAlchemyError, ValueError) as e:
            logger.error(f"Error during vector store initialization: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing vector store: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            raise
            
    async def preload_schema_to_vectordb(self, schema_info: Dict[str, Any]) -> None:
        """Preload schema information to vector database.
        
        Args:
            schema_info (Dict[str, Any]): Schema information to preload
            
        Raises:
            Exception: If an error occurs during preloading
        """
        try:
            if not schema_info:
                raise ValueError("No schema information provided for preloading")
                
            logger.info("Starting schema preloading")
            
            # Create documents from schema info
            schema_docs = self.create_schema_documents(schema_info)
            if not schema_docs:
                raise ValueError("Failed to create schema documents")
                
            # Create prompt documents
            prompt_docs = self.create_prompt_documents()
            if not prompt_docs:
                raise ValueError("Failed to create prompt documents")
            
            # Initialize vector stores asynchronously
            await self.vector_store_manager.initialize_schema_store(schema_docs)
            await self.vector_store_manager.initialize_prompt_store(prompt_docs)
            
            logger.info("Successfully preloaded schema to vector database")
        except Exception as e:
            logger.error(f"Error preloading schema to vector database: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            raise

    async def get_relevant_prompt(self, query: str, prompt_type: str) -> str:
        """Get the most relevant prompt for a given query and prompt type.

        Args:
            query (str): The query to find relevant prompt for
            prompt_type (str): The type of prompt to retrieve

        Returns:
            str: The most relevant prompt

        Raises:
            ValueError: If no prompt is found for the given type
        """
        try:
            results = await handle_openai_error(self.vector_store_manager.query_prompt(query, prompt_type))
            
            if not results:
                logger.warning(f"No relevant prompt found for type: {prompt_type}")
                raise ValueError(f"No prompt found for type: {prompt_type}")
                
            return results[0].page_content
        except Exception as e:
            logger.error(f"Error retrieving relevant prompt: {str(e)}")
            raise ValueError(f"No prompt found for type: {prompt_type}")

    async def get_relevant_schema(self, query: str) -> str:
        """Get the most relevant schema for a given query."""
        if not query:
            logger.warning("Empty query provided")
            return ""
        
        try:
            logger.info(f"Processing query: {query}")
            
            # Get relevant documents from vector store
            results = await self.vector_store_manager.query_schema(query, k=5)
            
            if not results:
                logger.warning(f"No relevant schema found for query: {query}")
                return ""
            
            # Log the top matches
            logger.info(f"Found {len(results)} relevant schema matches:")
            for i, result in enumerate(results, 1):
                logger.info(f"Match {i}: {result.metadata.get('table_name')} - Score: {result.metadata.get('score', 'N/A')}")
            
            # Combine the most relevant schemas
            relevant_schemas = []
            for result in results:
                table_name = result.metadata.get('table_name')
                description = result.metadata.get('description', '')
                relevant_schemas.append(f"Table: {table_name}\nDescription: {description}")
            
            return "\n\n".join(relevant_schemas)
            
        except Exception as e:
            logger.error(f"Error getting relevant schema: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            return ""

    async def preload_schema_error_handling(self) -> None:
        """Preload schema information and initialize vector store.

        Raises:
            Exception: If schema extraction fails
        """
        try:
            schema_info = await self.schema_extractor.extract_table_schema()
            await self.vector_store_manager.initialize(schema_info)
        except Exception as e:
            logger.error(f"Error preloading schema: {str(e)}")
            raise Exception("Failed to extract schema information")

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
