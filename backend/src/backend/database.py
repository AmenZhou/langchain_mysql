from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker
from langchain_community.utilities import SQLDatabase
import os
import pymysql
from typing import Optional, TYPE_CHECKING
from fastapi import HTTPException, status
import logging

if TYPE_CHECKING:
    from .langchain_mysql import LangChainMySQL

# Configure logging
logger = logging.getLogger(__name__)

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://user:password@localhost:3306/database")

# Initialize engine and session as None
_engine = None
_Session = None

# Global variables
engine: Optional[Engine] = None
langchain_mysql: Optional['LangChainMySQL'] = None

def get_db_engine():
    """Get or create the database engine."""
    global engine
    if engine is None:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return engine

def get_db():
    """Get a database session."""
    if _Session is None:
        _Session = sessionmaker(bind=get_db_engine())
    return _Session()

def get_langchain_db():
    """Get a database connection for LangChain."""
    engine = get_db_engine()
    return engine

async def get_langchain_mysql():
    """Get or create the LangChainMySQL instance."""
    global langchain_mysql
    if langchain_mysql is None:
        from .langchain_mysql import LangChainMySQL
        langchain_mysql = LangChainMySQL()
        try:
            await langchain_mysql.initialize()
        except Exception as e:
            logger.error(f"Error initializing LangChain MySQL: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize LangChain MySQL: {str(e)}"
            )
    return langchain_mysql

__all__ = ['get_db', 'get_db_engine', 'get_langchain_db', 'get_langchain_mysql']
