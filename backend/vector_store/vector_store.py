from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document
from typing import List, Optional
import logging
import asyncio
import os

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db", embeddings: Optional[OpenAIEmbeddings] = None):
        self.persist_directory = persist_directory
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.schema_vectordb = None
        self.prompt_vectordb = None
        self._load_schema_store()

    def _load_schema_store(self) -> None:
        """Load the schema vector store from disk if it exists."""
        try:
            if os.path.exists(self.persist_directory):
                logger.info(f"Loading schema store from {self.persist_directory}")
                self.schema_vectordb = FAISS.load_local(
                    self.persist_directory, 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # We trust our own files
                )
                logger.info("Successfully loaded schema store from disk")
        except Exception as e:
            logger.error(f"Error loading schema store from disk: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")

    async def initialize_schema_store(self, documents: List[Document], persist_directory: Optional[str] = None) -> None:
        """Initialize or update the schema vector database."""
        if persist_directory:
            self.persist_directory = persist_directory

        try:
            if not documents:
                raise ValueError("No documents provided for schema store initialization")
                
            logger.info(f"Initializing schema store with {len(documents)} documents")
            texts = [doc.page_content for doc in documents]
            
            self.schema_vectordb = await asyncio.to_thread(
                lambda: FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=[doc.metadata for doc in documents]
                )
            )
            
            if self.persist_directory:
                await asyncio.to_thread(lambda: self.schema_vectordb.save_local(self.persist_directory))
                logger.info(f"Saved schema store to {self.persist_directory}")
                
            logger.info("Schema store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing schema vector store: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            raise

    def initialize_prompt_store(self, documents: List[Document]) -> None:
        """Initialize or update the prompt vector database."""
        try:
            if self.persist_directory:
                self.prompt_vectordb = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=f"{self.persist_directory}_prompts"
                )
            else:
                self.prompt_vectordb = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
        except Exception as e:
            logger.error(f"Error initializing prompt vector store: {e}")
            raise

    async def query_schema(self, query: str, k: int = 3) -> List[Document]:
        """Query the schema vector database."""
        try:
            if not query:
                logger.error("Empty query received in query_schema")
                raise ValueError("Query cannot be empty")
                
            if not self.schema_vectordb:
                logger.error("Schema vector store not initialized")
                raise ValueError("Schema vector store not initialized")
                
            logger.info(f"Querying schema vector store with query: {query}")
            logger.info("Performing similarity search")
            results = await asyncio.to_thread(lambda: self.schema_vectordb.similarity_search(query, k=k))
            logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error querying schema vector store: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            raise

    def query_prompts(self, query: str, prompt_type: Optional[str] = None, k: int = 1) -> List[Document]:
        """Query the prompt vector database."""
        if not self.prompt_vectordb:
            raise ValueError("Prompt vector store not initialized")
        
        filter_dict = {"prompt_type": prompt_type} if prompt_type else None
        return self.prompt_vectordb.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the schema vector database.
        
        Args:
            documents (List[Document]): Documents to add to the vector store
            
        Raises:
            ValueError: If no documents provided
            Exception: For other errors during document addition
        """
        if not documents:
            raise ValueError("No documents provided to add to vector store")
            
        try:
            await self.initialize_schema_store(documents)
            logger.info(f"Added {len(documents)} documents to schema vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise 