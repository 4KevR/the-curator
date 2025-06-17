from src.backend.modules.helpers import check_for_environment_variables

required_vars = [
    "ANKI_COLLECTION_PATH",
]

check_for_environment_variables(required_vars)
