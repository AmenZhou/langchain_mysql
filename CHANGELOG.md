# Changelog

## [Unreleased] - 2025-04-27

### Added
- Enhanced schema understanding and query processing capabilities
- Improved testing infrastructure with new configurations and patterns
- Enhanced backend architecture with better error handling
- Updated configuration and documentation
- Improved frontend-backend integration

### Changed
- Refactored schema extraction to include comprehensive table information
- Enhanced schema vectorization with detailed document representations
- Improved vector store management with persistent storage
- Updated error handling and logging throughout the system
- Aligned frontend query parameter naming with backend conventions

### Fixed
- Improved foreign key relationship detection
- Enhanced SQL query generation with better table joins
- Fixed error handling in database operations
- Improved logging for better debugging
- Enhanced test coverage across multiple modules

## Detailed Changes

### Schema Understanding and Query Processing Improvements

1. **Enhanced Schema Extraction**
   - The `SchemaExtractor` class now captures comprehensive table information:
     - Column details including name, type, nullability, and default values
     - Primary key constraints
     - Foreign key relationships with detailed reference information
     - Index information for performance optimization
   - Improved error handling with detailed logging for schema extraction failures
   - Better table descriptions that include relationship information

2. **Improved Schema Vectorization**
   - The `SchemaVectorizer` class now creates more detailed document representations:
     - Each table's schema is converted into a rich document with:
       - Column descriptions including types and constraints
       - Foreign key relationships explicitly stated
       - Table relationships clearly documented
     - Documents include metadata for better context during query processing
   - Enhanced error handling during vector store initialization
   - Better logging for tracking schema processing

3. **Advanced Vector Store Management**
   - The `VectorStoreManager` class provides:
     - Persistent storage of schema information using FAISS
     - Efficient similarity search for relevant schema information
     - Separate storage for prompts and schema information
     - Asynchronous operations for better performance
   - Improved error handling and logging throughout the process

4. **Query Processing Improvements**
   - Better understanding of table relationships through:
     - Explicit foreign key relationship storage
     - Detailed column type information
     - Table relationship descriptions
   - Enhanced error handling for:
     - Empty queries
     - Uninitialized vector stores
     - Invalid schema information
   - Improved logging for debugging and monitoring

These improvements have resulted in:
- More accurate SQL query generation
- Better understanding of table relationships
- Improved error handling and debugging capabilities
- More efficient schema processing and storage
- Better performance through asynchronous operations 
