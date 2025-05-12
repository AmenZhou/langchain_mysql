from sqlalchemy import inspect, create_engine
import os
from .db_utils import get_database_url

# Database connection
DB_URI = get_database_url()
engine = create_engine(DB_URI)

def list_all_tables():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print("\nAll tables in the database:")
    print("-" * 50)
    for table in sorted(tables):
        print(table)

if __name__ == "__main__":
    list_all_tables() 
