from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .langchain_mysql import LangChainMySQL, lifespan
from .database import get_db_engine
from .schema_vectorizer import SchemaVectorizer
from pydantic import BaseModel

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db_engine = get_db_engine()
schema_vectorizer = SchemaVectorizer()
langchain_mysql = LangChainMySQL(db_engine, schema_vectorizer)

class Query(BaseModel):
    question: str

@app.post("/query")
async def query_database(query: Query):
    try:
        result = await langchain_mysql.run_query_with_retry(query.question)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_memory")
async def reset_memory():
    return await langchain_mysql.reset_memory()

@app.post("/preload_schema")
async def preload_schema():
    return await langchain_mysql.preload_schema()

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 
