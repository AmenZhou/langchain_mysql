# ✅ Prompt for the Refinement AI
PROMPT_REFINE = """You are an expert prompt engineer. Your task is to take a user's natural language query and refine it to reliably generate a raw SQL query using a language model connected to a SQL database via Langchain. The goal is to get ONLY the SQL query, without any extra explanations, execution results, or natural language.

{database_schema_info}

Here is the user's query: '{user_query}'

Please rewrite the user's query to be very explicit and direct in asking for the raw SQL query. Ensure the refined query:
- Clearly specifies the table(s) involved.
- Clearly specifies the columns to retrieve (ideally all columns using '*').
- Clearly specifies the condition for filtering (e.g., 'where the consultation_id is equal to 1').
- Ends with a request for the raw SQL query only.
- Avoids any ambiguity or phrasing that might lead to explanations or execution.
"""

# ✅ Prompt for Table Name Queries
PROMPT_TABLE_QUERY = """You are an expert SQL assistant. Your task is to help users find tables in the database based on their descriptions.

{database_schema_info}

Here is the user's query about table names: '{user_query}'

Generate a SQL query that finds tables matching the description. The query must:
1. Use the information_schema.tables table
2. Filter for the current database using table_schema = DATABASE()
3. Use LIKE conditions to match table names
4. Return only the table_name column
5. Order results alphabetically

For queries about multiple words (like "medical and service"), use this exact format:
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = DATABASE() 
AND table_name LIKE '%medical%' 
AND table_name LIKE '%service%'
ORDER BY table_name;

IMPORTANT:
- Only return the SQL query, no explanations
- Use exact SQL syntax
- Do not include any extra text
- Use proper LIKE conditions with % wildcards
- Always include ORDER BY table_name
"""

# ✅ Secondary Prompt to Review SQL Response and Remove PHI/PII (Allow IDs)
def get_sanitize_prompt(response: str) -> str:
    return f"""
    You are a data privacy filter. Your job is to review the SQL response below and redact any Protected Health Information (PHI) or Personally Identifiable Information (PII), including names, addresses, medical records, and other sensitive details.

    However, you should **allow numeric identifiers** related to the context of the query to remain unredacted. Specifically:
    - **Allow numeric values that appear to be Member IDs.**
    - **Allow numeric values that appear to be Consultation IDs.**
    - Generally, allow any purely numeric IDs that are likely foreign keys or identifiers within the database.

    SQL Response:
    {response}

    Please return the sanitized response with all PHI/PII redacted, but ensure that relevant numeric IDs (like Member IDs and Consultation IDs) are not redacted.
    """
