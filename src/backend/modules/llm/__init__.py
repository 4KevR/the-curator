from src.backend.modules.helpers import check_for_environment_variables

required_vars = [
    "LLM_URL",
]

check_for_environment_variables(required_vars)
