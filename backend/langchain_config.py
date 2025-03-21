from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from backend.database import db  # Import the database object
import os

# ✅ Secure API Key Handling
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Use GPT-3.5-Turbo for Faster Processing
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500)

# ✅ Summarized Memory to Reduce Token Usage
memory = ConversationSummaryMemory(llm=llm, memory_key="history", max_token_limit=500, output_key="result")

# ✅ Define a prompt template with memory integration
prompt = PromptTemplate(
    input_variables=["history", "query"],
    template="Chat History: {history}\nUser: {query}\n\nPay close attention to the details of the previous turn, especially the number of rows in any tables printed."
)

# ✅ Solution 2: top_k=1 to reduce query complexity
db_chain = SQLDatabaseChain.from_llm(
    llm,
    db,
    verbose=True,
    return_intermediate_steps=True,
    memory=memory,
    top_k=1
)
