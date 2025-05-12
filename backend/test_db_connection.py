from sqlalchemy import create_engine, text
import os
from .db_utils import get_database_url

# Database connection parameters from docker-compose.yml
DB_HOST = os.getenv("DB_HOST", "mysql")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rootpassword")
DB_NAME = os.getenv("DB_NAME", "sakila")
DB_PORT = os.getenv("DB_PORT", "3306")

# Construct database URL
DATABASE_URL = get_database_url()

def test_connection():
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Test connection
        with engine.connect() as connection:
            # Try to execute a simple query
            result = connection.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            
            # List all tables
            result = connection.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            print("\nTables in the database:")
            for table in tables:
                print(f"- {table}")
                
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_connection() 