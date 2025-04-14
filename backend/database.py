from sqlalchemy import create_engine, MetaData
from langchain_community.utilities import SQLDatabase
from backend.included_tables import INCLUDED_TABLES
import os

# ✅ Secure API Key Handling (Only if you need it here, otherwise keep in main app)
# openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Correct Database Connection (Ensure password is set!)
DB_URI = "mysql+pymysql://root:@mysql:3306/dev_tas_live"
engine = create_engine(DB_URI)

# ✅ Solution 1: Minimize or disable reflection for SQLDatabase
metadata = MetaData()  # Do not auto-reflect large foreign key relationships

# ✅ Include only necessary tables, skipping reflection
# Note: We disabled foreign key resolution by setting a limited list of tables
db = SQLDatabase(
    engine, 
    metadata=metadata, 
    include_tables=INCLUDED_TABLES,
    sample_rows_in_table_info=2  # Reduce the number of sample rows to minimize token usage
)
