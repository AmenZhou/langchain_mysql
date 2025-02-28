
# LangChain MySQL Integration

This repository demonstrates how to integrate [LangChain](https://github.com/hwchase17/langchain) with a MySQL database. It provides a setup to run a MySQL instance using Docker and interact with it using LangChain.

## Features

- **Dockerized MySQL**: Easily set up and run a MySQL database instance using Docker.
- **LangChain Integration**: Connect and interact with the MySQL database using LangChain's SQL capabilities.
- **Sample Queries**: Execute sample queries to demonstrate the interaction between LangChain and MySQL.

## Prerequisites

- [Docker](https://www.docker.com/get-started): Ensure Docker is installed and running on your system.
- [Python 3.9+](https://www.python.org/downloads/): Required for running the Python scripts.
- [OpenAI API Key](https://platform.openai.com/account/api-keys): Necessary for LangChain's LLM functionalities.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/AmenZhou/langchain_mysql.git
   cd langchain_mysql
   ```

2. **Set Up Environment Variables**:

   Create a `.env` file in the root directory with the following content:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key.

3. **Build and Run Docker Containers**:

   Use Docker Compose to build and start the MySQL container:

   ```bash
   docker-compose up -d
   ```

   This command will:

   - Build the Docker image as defined in the `Dockerfile`.
   - Start a MySQL container with the configuration specified in `docker-compose.yml`.

4. **Install Python Dependencies**:

   It's recommended to use a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

5. **Initialize the MySQL Database**:

   After the MySQL container is running, initialize the database with sample data:

   ```bash
   ./load.sh
   ```

6. **Run the LangChain Script**:

   Execute the Python script to interact with the MySQL database:

   ```bash
   python langchain_mysql.py
   ```

   This script will:

   - Connect to the configured database.
   - Execute sample queries using LangChain.
   - Display the results.

## Additional Resources

For more information on integrating LangChain with SQL databases, consider exploring the following resources:

- [Chat With a MySQL Database Using Python and LangChain](https://alejandro-ao.com/chat-with-mysql-using-python-and-langchain/)
- [LangChain SQL Database Chain Example](https://github.com/sugarforever/LangChain-SQL-Chain)
- [LangChain Documentation on SQL Database Agents](https://python.langchain.com/docs/integrations/sql_database_agents)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Note: Ensure you have the necessary permissions and have reviewed the code before running scripts, especially in a production environment.*
```

This `README.md` provides a comprehensive overview of the repository, including setup instructions, prerequisites, and additional resources for users to reference. 
