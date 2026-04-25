import os
import tomllib

with open("core/config.toml", "rb") as f:
    config = tomllib.load(f)

# safe config
MAX_HISTORY = config["app"]["max_history"]
MODEL_NAME = config["llm"]["model"]
DATABASE_URL = config["database"]["default_url"]

# env-based
GROQ_API_KEY = os.getenv("GROQ_API_KEY")