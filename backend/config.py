from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv
from db_utils import get_database_url

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = get_database_url()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Other shared configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

class Settings(BaseSettings):
    # Database settings
    database_url: str = DATABASE_URL
    
    # OpenAI settings
    openai_api_key: str = OPENAI_API_KEY
    
    # Security settings
    allowed_origins: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:4000").split(",")
    allowed_hosts: List[str] = os.getenv("ALLOWED_HOSTS", "*").split(",")
    session_secret: str = os.getenv("SESSION_SECRET", "your-secret-key")
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    # Rate limiting
    rate_limit: str = os.getenv("RATE_LIMIT", "5/minute")
    
    class Config:
        env_file = ".env"
        case_sensitive = True 
