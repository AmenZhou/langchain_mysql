from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from routers import query, auth
from security import setup_security_middleware, limiter
from config import Settings
from contextlib import asynccontextmanager
from schema_vectorizer import SchemaVectorizer
from db_utils import get_database_url
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()

# Define persistent directory for vector store
VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_store")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Initialize schema vectorizer
schema_vectorizer = SchemaVectorizer(
    db_url=get_database_url(),
    persist_directory=VECTOR_STORE_DIR
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    try:
        logger.info("Initializing schema vectorizer...")
        schema_info = await schema_vectorizer.extract_table_schema()
        logger.info("Extracted schema information")
        await schema_vectorizer.initialize_vector_store(schema_info)
        logger.info("Schema vectorizer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing schema vectorizer: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
    yield

app = FastAPI(
    title="LangChain MySQL API",
    description="API for interacting with MySQL using LangChain",
    version="1.0.0",
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url="/redoc" if settings.environment == "development" else None,
    lifespan=lifespan,
)

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup security middleware
setup_security_middleware(app)

# Include routers
app.include_router(query.router)
app.include_router(auth.router)

# Health check endpoint
@app.get("/health")
@limiter.limit("5/minute")
async def health_check(request: Request):
    return {"status": "healthy"}
