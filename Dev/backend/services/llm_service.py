from groq import Groq
import logging
import os

logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_response(message: str) -> str:
    logger.info("Calling Groq LLM")

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Keep it concise"},
                {"role": "user", "content": message}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Groq LLM failed: {str(e)}")
        raise