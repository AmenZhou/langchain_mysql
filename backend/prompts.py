# ✅ Prompt for the Refinement AI
PROMPT_REFINE = """You are an expert prompt engineer. Your task is to take a user's natural language query and refine it so that it will reliably generate a raw SQL query using a language model connected to a SQL database via Langchain. The goal is to get only the SQL query without any extra explanations or execution results.

Here is the user's query: '{user_query}'

Please rewrite the user's query to be more explicit and direct in asking for the raw SQL query. Ensure the refined query clearly specifies the tables and columns involved and asks for the SQL to be returned without execution or additional text."""

# ✅ Secondary Prompt to Review SQL Response and Remove PHI/PII (Allow IDs)
def get_sanitize_prompt(response: str) -> str:
    return f"""
    You are a data privacy filter. Your job is to review the SQL response below and redact any Protected Health Information (PHI) or Personally Identifiable Information (PII), including names, addresses, medical records, and other sensitive details.

    SQL Response:
    {response}

    Please return the sanitized response with all PHI/PII redacted, but allow numeric IDs (e.g., user IDs, document IDs) to remain.
    """
