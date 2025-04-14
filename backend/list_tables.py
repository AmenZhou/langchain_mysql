from sqlalchemy import inspect, create_engine
import os

# Database connection
DB_URI = "mysql+pymysql://root:@localhost:3306/dev_tas_live"
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
