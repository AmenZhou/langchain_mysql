import os
import traceback
from openai import OpenAIError, OpenAI
from sqlalchemy import text

# ✅ Secure API Key Handling
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

# ✅ Configure Logging (You might keep this in the main file as well)
import logging
logger = logging.getLogger(__name__)

# ✅ Function to refine the prompt using AI
async def refine_prompt_with_ai(user_query: str) -> str | None:
    try:
        from backend.prompts import PROMPT_REFINE
        refine_prompt = PROMPT_REFINE.format(user_query=user_query)
        refinement_response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can choose a different model if you prefer
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=0.2,  # Keep it relatively low for more consistent refinement
        )
        refined_query = refinement_response.choices[0].message.content.strip()
        logger.info(f"Refined query: {refined_query}")
        return refined_query
    except OpenAIError as e:
        logger.error(f"Error during prompt refinement: {e}")
        logger.error(traceback.format_exc())
        return None
    except Exception as e:
        logger.error(f"Unexpected error during prompt refinement: {e}")
        logger.error(traceback.format_exc())
        return None

# ✅ Secondary Prompt to Review SQL Response and Remove PHI/PII (Allow IDs)
async def sanitize_sql_response(response: str, llm) -> str:
    try:
        from backend.prompts import get_sanitize_prompt
        review_prompt = get_sanitize_prompt(response)
        sanitized_response = llm.predict(review_prompt)
        return sanitized_response
    except OpenAIError as e:
        logger.error(f"Error during sanitization: {e}")
        return response  # Return the original response if sanitization fails
