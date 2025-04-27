from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost:3306/dev_tas_live")

# Initialize engine and session as None
_engine = None
_Session = None

def get_db_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine

def get_db():
    """Get a database session."""
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_db_engine())
    return _Session()

def get_langchain_db():
    """Get a database connection for LangChain."""
    return get_db_engine()

__all__ = ['get_db', 'get_db_engine', 'get_langchain_db', 'DATABASE_URL'] 
