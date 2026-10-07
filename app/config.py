import os

from dotenv import load_dotenv


load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)


if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set"
    )


if not TAVILY_API_KEY:
    raise RuntimeError(
        "TAVILY_API_KEY environment variable is not set"
    )