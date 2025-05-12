"""Custom exceptions for the backend module."""

class OpenAIRateLimitError(Exception):
    """Raised when OpenAI API rate limit is exceeded."""
    pass

class OpenAIAPIError(Exception):
    """Raised when OpenAI API returns an error."""
    pass

class DatabaseError(Exception):
    """Raised when a database operation fails."""
    pass

class SchemaError(Exception):
    """Raised when there is an error with the database schema."""
    pass

class ConfigurationError(Exception):
    """Raised when there is a configuration error."""
    pass 
