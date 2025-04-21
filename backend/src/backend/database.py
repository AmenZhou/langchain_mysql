from sqlalchemy import create_engine, MetaData
from langchain_community.utilities import SQLDatabase
import os

# ✅ Secure API Key Handling (Only if you need it here, otherwise keep in main app)
# openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Correct Database Connection (Ensure password is set!)
DB_URI = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost:3306/dev_tas_live")

# Initialize these as None and provide functions to get/initialize them
engine = None
db = None

def get_db_engine():
    """Get the SQLAlchemy engine instance."""
    global engine
    if engine is None:
        engine = create_engine(DB_URI)
    return engine

def get_db():
    """Get the SQLDatabase instance."""
    global db
    if db is None:
        metadata = MetaData()  # Do not auto-reflect large foreign key relationships
        db = SQLDatabase(
            get_db_engine(), 
            metadata=metadata,
            sample_rows_in_table_info=2  # Reduce the number of sample rows to minimize token usage
        )
    return db

__all__ = ['get_db', 'get_db_engine']
