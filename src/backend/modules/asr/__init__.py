from src.backend.modules.helpers import check_for_environment_variables

required_vars = [
    "LECTURE_TRANSLATOR_TOKEN",
    "LECTURE_TRANSLATOR_URL",
]

check_for_environment_variables(required_vars)
