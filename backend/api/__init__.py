"""
API Package - FastAPI Route Modules

Organized API endpoints for the LangChain MySQL application.
Separates concerns between different feature sets.
"""

from .queries import router as queries_router

# Charts router is optional - only import if dependencies are available
try:
    from .charts import router as charts_router
    __all__ = ['charts_router', 'queries_router']
except ImportError:
    charts_router = None
    __all__ = ['queries_router'] 