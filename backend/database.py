from sqlalchemy import create_engine, MetaData
from langchain_community.utilities import SQLDatabase
import os

# ✅ Secure API Key Handling (Only if you need it here, otherwise keep in main app)
# openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Correct Database Connection (Ensure password is set!)
DB_URI = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost:3306/dev_tas_live")
engine = create_engine(DB_URI)

# ✅ Solution 1: Minimize or disable reflection for SQLDatabase
metadata = MetaData()  # Do not auto-reflect large foreign key relationships

# ✅ Initialize SQLDatabase with minimal configuration
db = SQLDatabase(
    engine, 
    metadata=metadata,
    sample_rows_in_table_info=2  # Reduce the number of sample rows to minimize token usage
)
