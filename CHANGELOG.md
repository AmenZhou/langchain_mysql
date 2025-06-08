# Changelog

## [Unreleased] - 2025-01-08

### Added
- **Comprehensive Chart Generation System**: Full implementation of intelligent chart generation with support for bar charts, line charts, pie charts, scatter plots, histograms, and time series
  - Advanced chart type detection based on data characteristics
  - Beautiful Plotly visualizations with interactive features
  - Smart confidence scoring and data analysis
  - Detailed chart metadata and recommendations
- **Configurable PII/PHI Data Protection**: Enterprise-grade privacy protection system
  - Global configuration system via environment variables (`ENABLE_PII_FILTERING`)
  - Runtime configuration methods for dynamic control
  - Performance-optimized filtering (bypasses LLM calls when disabled)
  - Multi-layer protection: post-processing, SQL response, and prompt-based filtering
  - Production-ready with secure defaults (filtering enabled by default)
- **Interactive Chart Demo**: Professional HTML demo showcasing chart generation capabilities
  - Real-time chart generation from natural language queries
  - Multiple example queries demonstrating different chart types
  - Chart eligibility analysis and confidence scoring display
  - Integration with PII filtering system for secure demonstrations

### Changed
- **Enhanced Docker Configuration**: Updated container setup with complete chart dependencies
  - Added pandas, plotly, matplotlib, seaborn, numpy, and kaleido to setup.py
  - Fixed container startup issues with proper dependency management
  - Improved docker-compose.yml with PII filtering environment variables

### Technical Implementation Details

#### PII/PHI Data Protection System

The new PII filtering system provides enterprise-grade data protection with the following components:

1. **Global Configuration (`backend/config.py`)**
   ```python
   class AppConfig:
       ENABLE_PII_FILTERING: bool = True  # Secure by default
       
       @classmethod
       def is_pii_filtering_enabled(cls) -> bool
       def enable_pii_filtering(cls) -> None
       def disable_pii_filtering(cls) -> None
       def toggle_pii_filtering(cls) -> None
   ```

2. **Environment Variable Control**
   - `ENABLE_PII_FILTERING=true|false` controls the global setting
   - Docker containers can be configured via docker-compose.yml
   - Production defaults to enabled for security

3. **Performance Optimization**
   - When disabled: Complete bypass of all filtering logic (no LLM calls)
   - When enabled: Intelligent filtering only when needed
   - Eliminates unnecessary API costs and latency when filtering is off

4. **Multi-Layer Protection**
   - `sanitize_sql_response()`: Analyzes SQL query results
   - `sanitize_query_data()`: Post-processes final data
   - `get_sanitize_prompt()`: LLM prompt-based filtering
   - All functions check global config before processing

#### Chart Generation System Architecture

The chart system is built with a modular architecture:

1. **Chart Orchestrator (`charts/orchestrator.py`)**
   - Central coordinator for chart generation workflow
   - Manages chart eligibility, type detection, and generation
   - Provides confidence scoring and recommendations

2. **Chart Type Detector (`charts/detector.py`)**
   - Intelligent analysis of data characteristics
   - Supports 6 chart types: bar, line, pie, scatter, histogram, time series
   - Confidence scoring based on data suitability

3. **Chart Generator (`charts/generator.py`)**
   - Creates beautiful Plotly visualizations
   - Interactive features with hover information
   - Professional color palettes and styling

4. **Chart Analyzer (`charts/analyzer.py`)**
   - Data quality assessment and statistics
   - Diversity and concentration metrics
   - Category distribution analysis

#### Demo Integration

The chart demo (`frontend/chart_demo.html`) showcases:
- Real-time chart generation from natural language
- PII filtering toggle for demonstration purposes
- Professional UI with confidence indicators
- Multiple chart type examples

### Migration Guide

To enable PII filtering in production:

1. **Environment Variable (Recommended)**
   ```bash
   export ENABLE_PII_FILTERING=true
   ```

2. **Docker Compose**
   ```yaml
   environment:
     ENABLE_PII_FILTERING: true
   ```

3. **Runtime Control**
   ```python
   from backend.config import AppConfig
   AppConfig.enable_pii_filtering()
   ```

### Breaking Changes

None. PII filtering is enabled by default, maintaining backward compatibility while enhancing security.

## [Previous] - 2025-04-27

### Added
- Enhanced schema understanding and query processing capabilities
- Improved testing infrastructure with new configurations and patterns
- Enhanced backend architecture with better error handling
- Updated configuration and documentation
- Improved frontend-backend integration
- Added support for multiple response types including SQL queries, data, and natural language answers

### Changed
- Refactored schema extraction to include comprehensive table information
- Enhanced schema vectorization with detailed document representations
- Improved vector store management with persistent storage
- Updated error handling and logging throughout the system
- Aligned frontend query parameter naming with backend conventions
- Enhanced query processing to execute SQL and generate natural language explanations

### Fixed
- Improved foreign key relationship detection
- Enhanced SQL query generation with better table joins
- Fixed error handling in database operations
- Improved logging for better debugging
- Enhanced test coverage across multiple modules
- Fixed foreign key vectorization to ensure foreign key relationships are properly included in the vector database

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

### Testing Infrastructure Improvements

1. **Enhanced Test Configuration**
   - Updated pytest configuration with:
     - Clear test path organization
     - Custom markers for different test types
     - Improved test output formatting
     - Better Python path configuration
   - Added comprehensive test fixtures for:
     - Mock database inspectors
     - Engine configurations
     - Schema extractors
     - Vector stores

2. **New Test Categories**
   - Core functionality tests
   - API endpoint tests
   - Integration tests
   - Mock database tests
   - Schema extraction tests
   - Vectorization tests

3. **Test Coverage Improvements**
   - Added tests for schema extraction:
     - Basic schema extraction
     - Document creation
     - Schema vectorization
     - Table extraction
   - Enhanced error handling tests
   - Improved integration test coverage
   - Better mock database testing

4. **Test Infrastructure**
   - Added automated test execution script
   - Improved test environment setup
   - Better test isolation
   - Enhanced test reporting
   - Streamlined test execution process

These testing improvements have resulted in:
- More reliable test execution
- Better test organization
- Improved test coverage
- Easier test maintenance
- Better debugging capabilities

### FAISS Implementation Details

1. **Core Functionality**
   - FAISS (Facebook AI Similarity Search) is used for efficient similarity search and clustering
   - Enables fast retrieval of relevant schema information for natural language queries
   - Provides persistent storage of vectorized schema data

2. **Implementation Features**
   - Efficient storage and retrieval of schema embeddings
   - Fast nearest neighbor search capabilities
   - Support for high-dimensional vectors
   - Persistent storage with local file system integration

3. **Technical Integration**
   ```python
   # Vector store initialization
   self.schema_vectordb = FAISS.load_local(
       self.persist_directory, 
       self.embeddings,
       allow_dangerous_deserialization=True
   )
   ```
   - Seamless integration with our schema vectorization system
   - Efficient similarity search for schema information
   - Support for both CPU and GPU implementations

4. **Benefits**
   - Fast retrieval of relevant schema information
   - Efficient storage of vectorized schema data
   - Scalable solution for large database schemas
   - Improved query processing performance

5. **Performance Considerations**
   - Optimized for high-dimensional vector spaces
   - Efficient indexing methods for similarity search
   - Support for various indexing options
   - Memory-efficient storage of schema information

### Future Test Improvements

1. **Test Architecture Improvements**
   - Clear separation between core, integration, and mock tests
   - Layered test structure for better organization
   - Improved test fixture management
   - Better test isolation patterns

2. **Comprehensive Test Coverage**
   - Edge case testing
   - Error scenario coverage
   - Performance boundary testing
   - Data validation testing
   - Integration point testing

3. **Test Documentation and Maintenance**
   - Clear test descriptions
   - Documented test dependencies
   - Detailed test scenarios
   - Automated test generation
   - Test maintenance tools

4. **Infrastructure Enhancements**
   - Improved test reporting
   - Test metrics implementation
   - Test dashboard creation
   - Automated test runs
   - Coverage checks

5. **Performance and Error Handling**
   - Benchmark testing
   - Error recovery testing
   - Error message validation
   - Performance monitoring
   - Resource usage tracking

These future improvements aim to:
- Reduce test maintenance overhead
- Improve test reliability
- Enhance test understanding
- Provide better coverage
- Simplify debugging
- Accelerate development cycles

### Multiple Response Types Support

1. **Response Type Options**
   - Added support for multiple response types:
     - `sql`: Return only the generated SQL query (default)
     - `data`: Execute the SQL query and return the results
     - `natural_language`: Generate natural language explanation of the query and results
     - `all`: Return SQL, data, and natural language explanation combined

2. **Enhanced Query Processing**
   - Updated the `process_query` method to handle different response types
   - Added execution of SQL queries when data is requested
   - Implemented natural language explanation generation for query results
   - Created structured data return format for better usability

3. **Improved Response Structure**
   - Updated the `QueryResponse` model to include:
     - `result`: The primary result based on the requested response type
     - `sql`: The generated SQL query
     - `data`: The data returned from executing the SQL query
     - `explanation`: Natural language explanation of the results
     - `response_type`: The type of response that was returned

4. **Error Handling**
   - Added specific error handling for execution and explanation failures
   - Implemented graceful degradation when certain response types fail
   - Maintained backward compatibility for existing integrations

These enhancements provide more flexibility and value to the API consumers, allowing them to choose the most appropriate response format for their use case.

## 05/13/2025

### Changed
- Refactored `SchemaExtractor` to accept an optional `inspector` object in its constructor. This allows for more direct injection of mock inspectors during testing, improving test stability and simplifying test setup by removing the need for complex patching of `sqlalchemy.inspect`.
- Internal methods of `SchemaExtractor` (e.g., `get_all_tables`, `extract_table_schema`) now use a new `self.inspector` property, which either returns the injected inspector or lazily creates one from the engine.
- Updated unit tests for `SchemaExtractor` (`test_schema_basic.py`, `test_schema_extractor.py`, `test_schema_vectorizer.py`) to align with the `SchemaExtractor` refactor, by passing a `mock_inspector` directly during `SchemaExtractor` instantiation.
- Corrected document content generation in `SchemaExtractor.create_schema_documents` to match test expectations, ensuring consistent output for schema representations.
- Fixed import path errors in `test_server.py` related to patching `AsyncOpenAI` by using an absolute module path.
- Stabilized `test_query_endpoint_exists` in `test_server.py` by using FastAPI's `app.dependency_overrides` for the `get_langchain_mysql` dependency, providing a more robust mocking mechanism for endpoint tests.
- Successfully resolved all previously failing and skipped tests, resulting in a fully passing test suite.

## 05/10/2024

### 05/15/2025
- Implemented chat history feature to provide conversational context to the LLM.
  - Stores conversation history per session in memory.
  - Injects history into prompts for improved contextual understanding.
  - Allows the chatbot to remember previous turns in a dialogue.

// ... existing code ... 
