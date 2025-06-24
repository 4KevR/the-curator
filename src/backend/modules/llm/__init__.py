__all__ = ["KitLLM", "LMStudioLLM"]

from src.backend.modules.helpers import check_for_environment_variables

from .kit_llm import KitLLM
from .lm_studio_llm import LMStudioLLM

required_vars = [
    "LLM_URL",
]

check_for_environment_variables(required_vars)
