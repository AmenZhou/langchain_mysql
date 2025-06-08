# LangChain and FastAPI Integration with MySQL

This repository demonstrates how to integrate [LangChain](https://github.com/hwchase17/langchain) with a MySQL database using a [FastAPI](https://fastapi.tiangolo.com/) backend, accompanied by a React frontend for user interaction. Everything is set up to run within Docker containers.

## Features

- **Dockerized FastAPI Backend**: Runs the FastAPI application using Docker.
- **Dockerized React Frontend**: Provides a user interface built with React, running in a Docker container.
- **LangChain Integration**: Connects to a MySQL database and interacts with it using LangChain's SQL capabilities.
- **API Endpoint**: The FastAPI backend exposes an API endpoint for the frontend to send natural language queries.
- **Schema Vectorization**: Preloads database schema into a vector database for improved query accuracy.

## Prerequisites

- [Docker](https://www.docker.com/get-started): Ensure Docker is installed and running on your system.
- [Docker Compose](https://docs.docker.com/compose/install/): Ensure Docker Compose is installed on your system.
- **MySQL Server**: A running MySQL server instance that the backend application can connect to. You will need to configure the connection details in your environment variables.
- [OpenAI API Key](https://platform.openai.com/account/api-keys): Necessary for LangChain's LLM functionalities.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone [https://github.com/AmenZhou/langchain_mysql.git](https://github.com/AmenZhou/langchain_mysql.git)
   cd langchain_mysql
   ```

2. **Set Up Environment Variables**:

   Create a `.env` file in the root directory with the following content. **Make sure to configure the MySQL connection details to match your running MySQL server.**

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=mysql+pymysql://<user>:<password>@<host>:<port>/<database>
   # Add any other backend specific environment variables here
   ```

   Replace the placeholders with your actual OpenAI API key and the connection details for your MySQL server.

3. **Build and Run Docker Containers**:

   Use Docker Compose to build and start all the application containers (backend and frontend):

   ```bash
   docker-compose up -d --build
   ```

   This command will:

   - Build the Docker images as defined in the `Dockerfile` and any frontend Dockerfile (if present).
   - Start the containers for the FastAPI backend and the React frontend, along with any other services defined in your `docker-compose.yml`.

## How to Use

1.  **Access the Frontend**: Once the Docker containers are running, the React frontend application should be accessible in your web browser. Based on the `package.json` you provided, it's likely running on port `4000`. So, navigate to `http://localhost:4000` in your browser.

2.  **Interact with the Backend**: The frontend application will provide an interface where you can enter your natural language questions. These questions will be sent to the FastAPI backend (likely running on `http://localhost:8000` if you're using the default configuration) at the `/query` endpoint.

3.  **Receive the Response**: The backend will process your query using LangChain, generate the SQL, interact with the MySQL database, and return a JSON response to the frontend. This response will typically contain the answer to your question and the generated SQL query.

4.  **Preload Schema (Optional)**: The database schema is automatically preloaded into a vector database on startup. If you want to manually preload it, you can use the following API endpoint:

    ```bash
    curl -X POST http://localhost:8000/preload_schema
    ```

    Alternatively, you can run the preload script directly:

    ```bash
    python -m backend.preload_schema --persist-dir ./chroma_db
    ```

## Schema Vectorization

This application uses a vector database to store and retrieve database schema information. This significantly improves query accuracy by providing relevant schema context to the LLM. Key benefits include:

- **Improved SQL Generation**: The LLM has access to accurate table and column information
- **Better Relationship Understanding**: Foreign key relationships are properly understood
- **Semantic Search**: Natural language queries are matched to the most relevant schema elements
- **Reduced Hallucinations**: The LLM is less likely to generate SQL with non-existent tables or columns

The schema information is automatically preloaded when the application starts.

## PII/PHI Data Protection

This application includes comprehensive PII (Personally Identifiable Information) and PHI (Protected Health Information) filtering capabilities to ensure sensitive data protection in query responses.

### PII Filtering Configuration

The PII filtering system can be controlled via environment variables and runtime configuration:

#### Environment Variable Control

**Enable PII Filtering (Default - Recommended for Production):**
```bash
# In .env file or environment
ENABLE_PII_FILTERING=true

# Or in docker-compose.yml
environment:
  ENABLE_PII_FILTERING: true
```

**Disable PII Filtering (For Development/Demo):**
```bash
# In .env file or environment
ENABLE_PII_FILTERING=false

# Or in docker-compose.yml
environment:
  ENABLE_PII_FILTERING: false
```

#### Runtime Configuration

You can also control PII filtering programmatically:

```python
from backend.config import AppConfig

# Check current status
print(f"PII Filtering enabled: {AppConfig.is_pii_filtering_enabled()}")

# Enable PII filtering
AppConfig.enable_pii_filtering()

# Disable PII filtering  
AppConfig.disable_pii_filtering()

# Toggle PII filtering
AppConfig.toggle_pii_filtering()
```

### How PII Filtering Works

When enabled, the system applies multiple layers of protection:

1. **Post-processing Filter**: Sanitizes query results using an LLM to identify and replace sensitive data with `[PRIVATE]`
2. **SQL Response Filter**: Analyzes SQL query results for potential PII/PHI data
3. **Prompt-based Protection**: Uses specialized prompts to instruct the LLM to identify and mask sensitive information

### Performance Optimization

The system is optimized for performance:
- **When PII filtering is disabled**: All filtering functions are bypassed entirely, eliminating unnecessary LLM calls
- **When PII filtering is enabled**: Only processes data that potentially contains sensitive information

### Security Considerations

**Production Deployment:**
- Always enable PII filtering in production environments
- Review filtered results to ensure sensitive data protection
- Consider additional encryption for data at rest and in transit

**Development/Testing:**
- PII filtering can be disabled for better development experience with real data visibility
- Ensure test databases don't contain actual sensitive information

### Chart Generation Impact

PII filtering affects chart generation:
- **Enabled**: Charts may show `[PRIVATE]` labels instead of actual values
- **Disabled**: Charts display actual data values for better visualization

For demonstration purposes, you may want to temporarily disable PII filtering to showcase full functionality with real data.

## Additional Resources

For more information on the technologies used in this project, consider exploring the following resources:

- [LangChain Documentation](https://python.langchain.com/docs/get_started)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Chat With a MySQL Database Using Python and LangChain](https://alejandro-ao.com/chat-with-mysql-using-python-and-langchain/)
- [LangChain SQL Database Chain Example](https://github.com/sugarforever/LangChain-SQL-Chain)
- [LangChain Documentation on SQL Database Agents](https://python.langchain.com/docs/integrations/sql_database_agents)
  
## Diagram
```mermaid
graph TD
    %% User interaction
    User((User)) -->|"Interacts via Browser"| Frontend

    %% Docker environment laid out left‑to‑right
    subgraph Docker_Environment
        direction LR

        %% Core containers and services
        Frontend["Frontend App<br/>(React/Node.js)"]
        Backend["LangChain Backend<br/>(Python)"]
        RefineLLM["LLM<br/>(Prompt&nbsp;Refiner)"]
        FAISS["FAISS Vector Store"]
        MainLLM["LLM<br/>(NL‑to‑SQL&nbsp;+&nbsp;Executor)"]
        DB["MySQL DB Container"]
        FilterLLM["LLM<br/>(PII&nbsp;Filter)"]

        %% 1  User → Frontend → Backend
        Frontend -- "① HTTP Request<br/>(Raw NL Query)" --> Backend

        %% 2–3 Prompt refinement
        Backend  -- "② Raw NL Query" --> RefineLLM
        RefineLLM -- "③ Refined Prompt" --> Backend

        %% 4–5 Schema retrieval (now immediately to the right)
        Backend -- "④ Retrieve Schema Chunks" --> FAISS
        FAISS   -- "⑤ Schema Context" --> Backend

        %% 6–9 Main LLM + DB
        Backend  -- "⑥ Refined Prompt + Schema Context" --> MainLLM
        MainLLM  -- "⑦ SQL Query" --> DB
        DB       -- "⑧ Raw DB Result" --> MainLLM
        MainLLM  -- "⑨ Raw DB Result" --> Backend

        %% 10–11 Result sanitisation
        Backend    -- "⑩ Raw DB Result + Filter Prompt" --> FilterLLM
        FilterLLM  -- "⑪ Sanitised Result" --> Backend

        %% 12 Backend → Frontend
        Backend -- "⑫ Sanitised API Response" --> Frontend
    end

    %% Styling
    style Frontend   fill:#e0f7ea,stroke:#333,stroke-width:2px
    style Backend    fill:#f5e0f7,stroke:#333,stroke-width:2px
    style DB         fill:#e0ecf7,stroke:#333,stroke-width:2px
    style FAISS      fill:#d0eaff,stroke:#339,stroke-width:2px
    style RefineLLM  fill:#d4f7d4,stroke:#696,stroke-width:1px,stroke-dasharray:3 3
    style MainLLM    fill:#fff4d4,stroke:#996,stroke-width:1px,stroke-dasharray:3 3
    style FilterLLM  fill:#f7d4d4,stroke:#966,stroke-width:1px,stroke-dasharray:3 3
```
### Resolve Token Rate Limit Exceeded Problem

```mermaid
---
config:
  layout: fixed
---
flowchart TD
 subgraph subGraph0["Problem: Using Full Schema"]
    direction LR
        B1["LLM Prompt (Exceeds Limit)"]
        A1[("Full DB Schema")]
        C1("LLM")
        D1{{"Error: Token Limit Exceeded"}}
  end
 subgraph subGraph1["Workaround: Using Subset Schema"]
    direction LR
        S[("Subset Schema")]
        Filter@{ label: "Subset Selection / `top_k=1`" }
        A2[("Full DB Schema")]
        B2["LLM Prompt (Within Limit)"]
        U["User Query"]
        C2("LLM")
        D2{{"OK: Query Processed"}}
  end
    A1 -- Schema Info (Very Large) --> B1
    B1 --> C1
    C1 --> D1
    A2 --> Filter
    Filter --> S
    U --> B2
    S -- Schema Info (Small) --> B2
    B2 --> C2
    C2 --> D2
    Filter@{ shape: diamond}
    style A1 fill:#ccc,stroke:#666
    style C1 fill:#ffc,stroke:#996
    style D1 fill:#f99,stroke:#f00,stroke-width:2px
    style S fill:#ccf,stroke:#33f
    style Filter fill:#ddd, stroke:#555 %% Style the filter process node
    style A2 fill:#ccc,stroke:#666
    style U fill:#eee, stroke:#555 %% Style the User Query node
    style C2 fill:#ffc,stroke:#996
    style D2 fill:#9f9,stroke:#0f0,stroke-width:2px
```
### PHI/PII Filter LLM
```mermaid
graph TD

    subgraph Docker_Environment
        %% Define container nodes directly within this subgraph

        Backend["LangChain Backend (Python Container)"]


       
        %% 4. Backend uses Filter LLM to sanitize the received result
        Backend --  Raw DB Result + Filter Prompt --> FilterLLM["LLM (PII Filter)"]
        FilterLLM -- Sanitized Result --> Backend

    end

    %% --- Styling (applied after nodes are implicitly defined/used) ---

    style Backend fill:#f9f,stroke:#333,stroke-width:2px

    style FilterLLM fill:#fcc,stroke:#966,stroke-width:1px,stroke-dasharray: 3 3
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Note: Ensure you have the necessary permissions and have reviewed the code before running scripts, especially in a production environment.*
