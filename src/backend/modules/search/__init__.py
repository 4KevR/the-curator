import logging
import os

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.lmstudio import LMStudio as LlamaLMStudio
from transformers import AutoTokenizer

from src.backend.modules.helpers import check_for_environment_variables

os.environ["TOKENIZERS_PARALLELISM"] = "false"

required_vars = [
    "LLM_URL",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
]

check_for_environment_variables(required_vars)

logger = logging.getLogger(__name__)

__USE_LOCAL_MODEL = True

if __USE_LOCAL_MODEL:
    logger.info("Initializing LlamaIndex with LM Studio LLM...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", cache_folder="./model_cache")
    Settings.llm = LlamaLMStudio(
        model_name="Meta-Llama-3.1-8B-Instruct",
        base_url="http://localhost:1234/v1",
        temperature=0.1,
    )
    logger.info("LlamaIndex initialized successfully with LM Studio.")

else:
    logger.info("Initializing LlamaIndex with Hugging Face Embedding model and KIT LLM...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", cache_folder="./model_cache")
    Settings.llm = HuggingFaceInferenceAPI(
        model=os.getenv("LLM_URL"),
        task="text-generation",
        num_output=64,
        temperature=0.1,
    )
    if os.getenv("HUGGING_FACE_TOKEN"):
        logging.info("Using Hugging Face Tokenizer for Llama-3.1-8B-Instruct")
        Settings.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", token=os.getenv("HUGGING_FACE_TOKEN"), cache_dir="./model_cache"
        )
    else:
        # Add HUGGING_FACE_TOKEN to .env and request access for https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
        logging.warning("Hugging Face Token not set. Using default tokenizer.")

    logger.info("LlamaIndex initialized successfully.")
