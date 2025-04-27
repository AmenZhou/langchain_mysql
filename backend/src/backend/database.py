from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_community.utilities import SQLDatabase
import os

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "mysql://user:password@localhost:3306/database")

# Initialize engine and session as None
_engine = None
_Session = None

def _get_engine():
    """Get or create the SQLAlchemy engine instance."""
    global _engine, _Session
    
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
        _Session = sessionmaker(bind=_engine)
    
    return _engine

def get_db_engine():
    """Get the SQLAlchemy engine instance."""
    return _get_engine()

def get_db():
    """Get a database session."""
    if _Session is None:
        _get_engine()
    return _Session()

def get_langchain_db():
    """Get a LangChain SQLDatabase instance."""
    return SQLDatabase.from_uri(DATABASE_URL)

__all__ = ['get_db', 'get_db_engine', 'get_langchain_db']
