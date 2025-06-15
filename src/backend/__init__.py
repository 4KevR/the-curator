import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv(".env")
load_dotenv(".env.db")

required_vars = [
    "LECTURE_TRANSLATOR_TOKEN",
    "LECTURE_TRANSLATOR_URL",
    "ANKI_COLLECTION_PATH",
    "LLM_URL",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
]
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.llm = HuggingFaceInferenceAPI(
    model=os.getenv("LLM_URL"),
    task="text-generation",
    num_output=64,
    temperature=0.1,
)
# Settings.tokenizer = AutoTokenizer.from_pretrained(
#     "meta-llama/Llama-3.1-8B-Instruct", token=""
# )
