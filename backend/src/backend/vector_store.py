from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document
from typing import List, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db", embeddings: Optional[OpenAIEmbeddings] = None):
        self.persist_directory = persist_directory
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.schema_vectordb = None
        self.prompt_vectordb = None

    async def initialize_schema_store(self, documents: List[Document], persist_directory: Optional[str] = None) -> None:
        """Initialize or update the schema vector database."""
        if persist_directory:
            self.persist_directory = persist_directory

        try:
            if not documents:
                raise ValueError("No documents provided for schema store initialization")
                
            logger.info(f"Initializing schema store with {len(documents)} documents")
            
            # Create embeddings asynchronously
            embeddings = await asyncio.to_thread(
                lambda: self.embeddings.embed_documents([doc.page_content for doc in documents])
            )
            
            if self.persist_directory:
                self.schema_vectordb = await asyncio.to_thread(
                    lambda: FAISS.from_embeddings(
                        text_embeddings=zip([doc.page_content for doc in documents], embeddings),
                        embedding=self.embeddings,
                        metadatas=[doc.metadata for doc in documents]
                    )
                )
                await asyncio.to_thread(lambda: self.schema_vectordb.save_local(self.persist_directory))
            else:
                self.schema_vectordb = await asyncio.to_thread(
                    lambda: FAISS.from_embeddings(
                        text_embeddings=zip([doc.page_content for doc in documents], embeddings),
                        embedding=self.embeddings,
                        metadatas=[doc.metadata for doc in documents]
                    )
                )
                
            logger.info("Schema store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing schema vector store: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            raise

    async def initialize_prompt_store(self, documents: List[Document]) -> None:
        """Initialize or update the prompt vector database."""
        try:
            if not documents:
                raise ValueError("No documents provided for prompt store initialization")
                
            logger.info(f"Initializing prompt store with {len(documents)} documents")
            
            if self.persist_directory:
                self.prompt_vectordb = await asyncio.to_thread(
                    lambda: Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        persist_directory=f"{self.persist_directory}_prompts"
                    )
                )
            else:
                self.prompt_vectordb = await asyncio.to_thread(
                    lambda: Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings
                    )
                )
                
            logger.info("Prompt store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing prompt vector store: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            raise

    async def query_schema(self, query: str, k: int = 3) -> List[Document]:
        """Query the schema vector database."""
        try:
            logger.info(f"Querying schema vector store with query: {query}")
            
            if not self.schema_vectordb:
                logger.error("Schema vector store not initialized")
                raise ValueError("Schema vector store not initialized")
                
            if not query:
                logger.error("Empty query received in query_schema")
                raise ValueError("Query cannot be empty")
                
            logger.info("Performing similarity search")
            results = await asyncio.to_thread(lambda: self.schema_vectordb.similarity_search(query, k=k))
            logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error querying schema vector store: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            raise

    async def query_prompts(self, query: str, prompt_type: Optional[str] = None, k: int = 1) -> List[Document]:
        """Query the prompt vector database."""
        try:
            if not self.prompt_vectordb:
                raise ValueError("Prompt vector store not initialized")
            
            if not query:
                raise ValueError("Query cannot be empty")
                
            filter_dict = {"prompt_type": prompt_type} if prompt_type else None
            return await asyncio.to_thread(
                lambda: self.prompt_vectordb.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            )
        except Exception as e:
            logger.error(f"Error querying prompt vector store: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            raise
        
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the schema vector database."""
        if not documents:
            raise ValueError("No documents provided to add to vector store")
            
        try:
            await self.initialize_schema_store(documents)
            logger.info(f"Added {len(documents)} documents to schema vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            raise 
