import os
from pathlib import Path

from dotenv import load_dotenv


# The project folder is the parent of the app folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load values from RAG/.env when the application starts.
load_dotenv(PROJECT_ROOT / ".env")


# Document and database locations.
DOCUMENTS_DIR = PROJECT_ROOT / "docs"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# Chroma and embedding settings.
CHROMA_COLLECTION_NAME = "documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Document processing settings.
MAX_DOCUMENTS = 5
MAX_UPLOAD_SIZE_MB = 50
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# xAI settings. Keep the real key in .env, never in source code.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "xai")
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_MODEL_NAME = os.getenv("XAI_MODEL_NAME", "grok-3-mini")
XAI_BASE_URL = "https://api.x.ai/v1"

# Groq settings. Keep the real key in .env, never in source code.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv(
	"GROQ_MODEL_NAME",
	"openai/gpt-oss-20b",
)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_xai_api_key() -> str:
	"""Return the xAI key or raise a helpful error when it is missing."""
	if not XAI_API_KEY:
		raise ValueError(
			"XAI_API_KEY is missing. Add it to a .env file in the project root."
		)
	return XAI_API_KEY


def get_groq_api_key() -> str:
	"""Return the Groq key or raise a helpful error when it is missing."""
	if not GROQ_API_KEY:
		raise ValueError(
			"GROQ_API_KEY is missing. Add it to a .env file in the project root."
		)
	return GROQ_API_KEY
