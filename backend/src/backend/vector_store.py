from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db", embeddings: Optional[OpenAIEmbeddings] = None):
        self.persist_directory = persist_directory
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.schema_vectordb = None
        self.prompt_vectordb = None

    def initialize_schema_store(self, documents: List[Document], persist_directory: Optional[str] = None) -> None:
        """Initialize or update the schema vector database."""
        if persist_directory:
            self.persist_directory = persist_directory

        try:
            if self.persist_directory:
                self.schema_vectordb = FAISS.from_documents(documents, self.embeddings)
                self.schema_vectordb.save_local(self.persist_directory)
            else:
                self.schema_vectordb = FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            logger.error(f"Error initializing schema vector store: {e}")
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

    def query_schema(self, query: str, k: int = 3) -> List[Document]:
        """Query the schema vector database."""
        if not self.schema_vectordb:
            raise ValueError("Schema vector store not initialized")
        return self.schema_vectordb.similarity_search(query, k=k)

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
            self.initialize_schema_store(documents)
            logger.info(f"Added {len(documents)} documents to schema vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise 
