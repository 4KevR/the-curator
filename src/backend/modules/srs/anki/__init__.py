__all__ = ["AnkiSRS"]

from src.backend.modules.helpers import check_for_environment_variables

from .anki import AnkiSRS

required_vars = [
    "ANKI_COLLECTION_PATH",
]

check_for_environment_variables(required_vars)
