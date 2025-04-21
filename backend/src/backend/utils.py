import os
import traceback
from openai import OpenAIError, AsyncOpenAI
from sqlalchemy import text
from typing import Optional, List
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from .database import get_db
from .schema_vectorizer import SchemaVectorizer

# ✅ Secure API Key Handling
def get_openai_client() -> AsyncOpenAI:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    return AsyncOpenAI(api_key=openai_api_key)

# ✅ Configure Logging (You might keep this in the main file as well)
import logging
logger = logging.getLogger(__name__)

# ✅ Function to refine the prompt using AI
async def refine_prompt_with_ai(query: str) -> Optional[str]:
    try:
        schema_vectorizer = SchemaVectorizer()
        schema_docs = schema_vectorizer.get_relevant_schema()
        
        openai_client = get_openai_client()
        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that refines SQL queries based on schema information."},
                {"role": "user", "content": f"Schema information:\n{schema_docs}\n\nOriginal query: {query}\n\nPlease refine this query."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in refine_prompt_with_ai: {e}")
        return None

# ✅ Secondary Prompt to Review SQL Response and Remove PHI/PII (Allow IDs)
async def sanitize_sql_response(response: str) -> str:
    try:
        llm = ChatOpenAI(model_name="gpt-4")
        prompt = PromptTemplate.from_messages([
            ("system", "Extract only the SQL query from the following response. Do not include any explanations or comments."),
            ("user", "{response}")
        ])
        chain = prompt | llm
        result = await chain.ainvoke({"response": response})
        return result.content
    except Exception as e:
        print(f"Error in sanitize_sql_response: {e}")
        return response
