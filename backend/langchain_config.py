from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from backend.database import db  # Import the database object
from backend.schema_vectorizer import get_schema_as_text  # Import the schema function
import os

# ✅ Secure API Key Handling
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Use GPT-3.5-Turbo for Faster Processing
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)

# ✅ Summarized Memory to Reduce Token Usage
memory = ConversationSummaryMemory(llm=llm, memory_key="history", max_token_limit=500, output_key="result")

# ✅ Define a prompt template with memory integration and schema information
prompt = PromptTemplate(
    input_variables=["history", "query", "schema_info"],
    template="""Chat History: {history}
User: {query}

Database Schema Information:
{schema_info}

Pay close attention to the details of the previous turn and the database schema information provided above, especially the table relationships and column names.
When writing SQL, make sure to reference the correct table and column names as shown in the schema.
"""
)

# Function to get relevant schema information based on the query
def get_relevant_schema_info(query):
    try:
        # Extract relevant schema information based on the query
        schema_info = get_schema_as_text(query=query, k=5)
        return schema_info
    except Exception as e:
        # If there's an error, return empty string to avoid breaking the chain
        print(f"Error getting schema info: {e}")
        return ""

# Custom chain creation with schema info
def create_db_chain_with_schema(query):
    # Get relevant schema info for this query
    schema_info = get_relevant_schema_info(query)
    
    # Create a custom prompt with the schema information
    custom_prompt = prompt.format(
        history=memory.buffer,
        query=query,
        schema_info=schema_info
    )
    
    # Create and return the chain
    chain = SQLDatabaseChain.from_llm(
        llm,
        db,
        verbose=True,
        return_intermediate_steps=True,
        memory=memory,
        top_k=1
    )
    
    return chain, custom_prompt

# ✅ Solution 2: top_k=1 to reduce query complexity
# The original db_chain is kept for backwards compatibility
db_chain = SQLDatabaseChain.from_llm(
    llm,
    db,
    verbose=True,
    return_intermediate_steps=True,
    memory=memory,
    top_k=1
)
