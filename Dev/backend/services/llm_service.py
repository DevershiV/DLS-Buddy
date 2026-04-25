from groq import Groq
from core.config import MODEL_NAME, GROQ_API_KEY
import logging
import os

logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

def generate_response(messages: str) -> str:
    logger.info("Calling Groq LLM")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Groq LLM failed: {str(e)}")
        raise