def replace_many(s: str, replacements: dict[str, str]) -> str:
    """
    Replace multiple substrings in a string. There is no guarantee on the order of replacements.
    """
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def find_substring_in_llm_response(
    response: str, token_for_true: str, token_for_false: str, ignore_case: bool = True
) -> bool:
    """
    Determines if a specific substring is present in a response string from an LLM
    and identifies its position relative to another token. It uses optional case-
    insensitive search functionality.

    Args:
        response: A string indicating the response from the LLM.
        token_for_true: A string token that designates a "true" response.
        token_for_false: A string token that designates a "false" response.
        ignore_case: A boolean indicating whether to ignore case while searching
            for substrings (default is True).

    Returns:
        A boolean indicating if the "true" token appears later in the response than
        the "false" token.

    Raises:
        ValueError: If both "true" and "false" tokens coexist in the same response
            string.
    """
    if ignore_case:
        response = response.lower()
        token_for_true = token_for_true.lower()
        token_for_false = token_for_false.lower()

    return find_substring_in_llm_response(response, "true", "false")
