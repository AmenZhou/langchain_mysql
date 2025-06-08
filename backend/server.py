from fastapi import FastAPI, HTTPException, status
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from db_utils import get_database_url
from api import queries_router
try:
    from api import charts_router
except ImportError:
    charts_router = None

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global database engine
engine = None

def get_db_engine():
    """Get or create the database engine."""
    global engine
    if engine is None:
        db_url = os.getenv("DATABASE_URL", get_database_url())
        engine = create_engine(db_url)
    return engine

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LangChain MySQL API",
        description="API for interacting with MySQL using LangChain",
        version="1.0.0"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app

def register_routers(app: FastAPI) -> None:
    """Register all routers with the application."""
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            engine = get_db_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "ok", "database": "connected"}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Health check failed: {str(e)}"
            )
    
    # Include routers for different feature sets
    app.include_router(queries_router)
    
    # Include charts router only if available
    if charts_router is not None:
        app.include_router(charts_router)
    else:
        logger.warning("Charts functionality not available - missing dependencies")

# Create the FastAPI application instance
app = create_app()
register_routers(app) 
