import os


def check_for_environment_variables(required_vars: list[str]) -> None:
    """
    Check if all required environment variables are set.

    :param required_vars: List of environment variable names to check.
    :raises EnvironmentError: If any required environment variable is missing.
    """
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
