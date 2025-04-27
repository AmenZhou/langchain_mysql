# ✅ Prompt for the Refinement AI
PROMPT_REFINE = """You are an expert prompt engineer. Your task is to take a user's natural language query and refine it to reliably generate a raw SQL query using a language model connected to a SQL database via Langchain. The goal is to get ONLY the SQL query, without any extra explanations, execution results, or natural language.

{schema}

Here is the user's query: '{query}'

Please rewrite the user's query to be very explicit and direct in asking for the raw SQL query. Ensure the refined query:
- Clearly specifies the table(s) involved.
- Clearly specifies the columns to retrieve (ideally all columns using '*').
- Clearly specifies the condition for filtering (e.g., 'where the consultation_id is equal to 1').
- Ends with a request for the raw SQL query only.
- Avoids any ambiguity or phrasing that might lead to explanations or execution.
"""

# ✅ Prompt for Table Name Queries
PROMPT_TABLE_QUERY = """You are an expert SQL assistant. Your task is to help users find tables in the database based on their descriptions.

{schema}

Here is the user's query about table names: '{query}'

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
def get_sanitize_prompt(sql_result: str) -> str:
    """Get the prompt for sanitizing SQL responses."""
    return f"""Please clean up and sanitize the following SQL result to ensure no sensitive data is exposed:

{sql_result}

Rules for sanitization:
1. Replace sensitive column values with [PRIVATE]
2. Keep table names and column names visible
3. Keep SQL keywords and syntax visible
4. Keep non-sensitive values (like IDs) visible
5. Return only the sanitized SQL, no explanations
"""
