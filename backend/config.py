"""
Global Configuration Settings

Centralized configuration for the LangChain MySQL application.
Controls features like PII filtering, chart generation, and other settings.
"""

import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AppConfig:
    """Global application configuration."""
    
    def __init__(self):
        # PII/PHI Filtering Configuration
        self.ENABLE_PII_FILTERING = self._get_bool_env("ENABLE_PII_FILTERING", default=True)
        
        # Chart Generation Configuration
        self.ENABLE_CHART_GENERATION = self._get_bool_env("ENABLE_CHART_GENERATION", default=True)
        
        # Database Configuration
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        
        # OpenAI Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Performance Configuration
        self.MAX_QUERY_RETRIES = int(os.getenv("MAX_QUERY_RETRIES", "3"))
        self.QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "30"))
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Development/Debug Configuration
        self.DEBUG_MODE = self._get_bool_env("DEBUG_MODE", default=False)
        self.ENABLE_CORS = self._get_bool_env("ENABLE_CORS", default=True)
        
        # Log the current configuration
        self._log_configuration()
    
    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
        else:
            return default
    
    def _log_configuration(self):
        """Log the current configuration settings."""
        logger.info("🔧 Application Configuration:")
        logger.info(f"   PII Filtering: {'✅ ENABLED' if self.ENABLE_PII_FILTERING else '❌ DISABLED'}")
        logger.info(f"   Chart Generation: {'✅ ENABLED' if self.ENABLE_CHART_GENERATION else '❌ DISABLED'}")
        logger.info(f"   Debug Mode: {'✅ ENABLED' if self.DEBUG_MODE else '❌ DISABLED'}")
        logger.info(f"   OpenAI Model: {self.OPENAI_MODEL}")
        logger.info(f"   Max Retries: {self.MAX_QUERY_RETRIES}")
        logger.info(f"   Log Level: {self.LOG_LEVEL}")
    
    def enable_pii_filtering(self):
        """Enable PII filtering at runtime."""
        self.ENABLE_PII_FILTERING = True
        logger.info("🔒 PII filtering ENABLED")
    
    def disable_pii_filtering(self):
        """Disable PII filtering at runtime."""
        self.ENABLE_PII_FILTERING = False
        logger.warning("⚠️  PII filtering DISABLED")
    
    def toggle_pii_filtering(self) -> bool:
        """Toggle PII filtering and return new state."""
        self.ENABLE_PII_FILTERING = not self.ENABLE_PII_FILTERING
        status = "ENABLED" if self.ENABLE_PII_FILTERING else "DISABLED"
        logger.info(f"🔄 PII filtering toggled to: {status}")
        return self.ENABLE_PII_FILTERING
    
    def enable_chart_generation(self):
        """Enable chart generation at runtime."""
        self.ENABLE_CHART_GENERATION = True
        logger.info("📊 Chart generation ENABLED")
    
    def disable_chart_generation(self):
        """Disable chart generation at runtime."""
        self.ENABLE_CHART_GENERATION = False
        logger.info("📊 Chart generation DISABLED")
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return self.DEBUG_MODE or os.getenv("ENVIRONMENT", "").lower() in ("dev", "development", "local")
    
    def get_summary(self) -> dict:
        """Get a summary of current configuration."""
        return {
            "pii_filtering_enabled": self.ENABLE_PII_FILTERING,
            "chart_generation_enabled": self.ENABLE_CHART_GENERATION,
            "debug_mode": self.DEBUG_MODE,
            "openai_model": self.OPENAI_MODEL,
            "max_retries": self.MAX_QUERY_RETRIES,
            "log_level": self.LOG_LEVEL,
            "development_mode": self.is_development_mode()
        }

# Global configuration instance
config = AppConfig()

# Convenience functions for easy access
def is_pii_filtering_enabled() -> bool:
    """Check if PII filtering is enabled."""
    return config.ENABLE_PII_FILTERING

def is_chart_generation_enabled() -> bool:
    """Check if chart generation is enabled."""
    return config.ENABLE_CHART_GENERATION

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config

# Environment variable documentation
ENV_VARS_HELP = """
Environment Variables for Configuration:

🔒 PII/PHI Filtering:
   ENABLE_PII_FILTERING=true|false    # Enable/disable PII filtering (default: true)

📊 Chart Generation:
   ENABLE_CHART_GENERATION=true|false # Enable/disable chart generation (default: true)

🗄️  Database:
   DATABASE_URL=mysql://...           # MySQL connection string

🤖 OpenAI:
   OPENAI_API_KEY=sk-...              # OpenAI API key
   OPENAI_MODEL=gpt-3.5-turbo         # OpenAI model to use

⚡ Performance:
   MAX_QUERY_RETRIES=3                # Maximum query retry attempts
   QUERY_TIMEOUT=30                   # Query timeout in seconds

🐛 Development:
   DEBUG_MODE=true|false              # Enable debug mode (default: false)
   LOG_LEVEL=INFO|DEBUG|WARNING       # Logging level (default: INFO)
   ENABLE_CORS=true|false             # Enable CORS (default: true)

Example .env file:
   ENABLE_PII_FILTERING=false
   ENABLE_CHART_GENERATION=true
   DEBUG_MODE=true
   LOG_LEVEL=DEBUG
"""

if __name__ == "__main__":
    print("🔧 LangChain MySQL Configuration")
    print("=" * 50)
    print(f"PII Filtering: {'✅ ENABLED' if config.ENABLE_PII_FILTERING else '❌ DISABLED'}")
    print(f"Chart Generation: {'✅ ENABLED' if config.ENABLE_CHART_GENERATION else '❌ DISABLED'}")
    print(f"Debug Mode: {'✅ ENABLED' if config.DEBUG_MODE else '❌ DISABLED'}")
    print("\n" + ENV_VARS_HELP)
