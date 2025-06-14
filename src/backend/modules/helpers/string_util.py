import re


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

    false_index = response.rfind(token_for_false)
    true_index = response.rfind(token_for_true)

    if false_index != -1 and true_index != -1:
        raise ValueError(f"Unexpected llm response: {response!r}")

    return true_index > false_index


def find_substring_in_llm_response_or_null(
    response: str, token_for_true: str, token_for_false: str, ignore_case: bool = True
) -> bool | None:
    """
    Determines if a specific substring is present in a response string from an LLM
    and identifies its position relative to another token. It uses optional case-
    insensitive search functionality.
    Returns None if none of the tokens are found.

    Args:
        response: A string indicating the response from the LLM.
        token_for_true: A string token that designates a "true" response.
        token_for_false: A string token that designates a "false" response.
        ignore_case: A boolean indicating whether to ignore case while searching
            for substrings (default is True).

    Returns:
        A boolean indicating if the "true" token appears later in the response than
        the "false" token.
    """
    if ignore_case:
        response = response.lower()
        token_for_true = token_for_true.lower()
        token_for_false = token_for_false.lower()

    false_index = response.rfind(token_for_false)
    true_index = response.rfind(token_for_true)

    if false_index != -1 and true_index != -1:
        return None

    return true_index > false_index


def remove_block(text: str, block_name: str, strip: bool = True) -> str:
    """
    Removes a specific block of text enclosed by given block tags from an input string.

    This function identifies and removes content within a block specified by its opening
    and closing tags, such as <block_name> and </block_name>. It also provides an optional
    argument to strip leading and trailing whitespaces around the resulting text.

    Parameters:
        text (str): The input string containing the block of text to remove.
        block_name (str): The tag name of the block to be removed.
        strip (bool): Whether to strip leading and trailing whitespaces in the
                      resulting string. Defaults to True.

    Returns:
        str: The modified string with the specified block removed.
    """
    pattern = f"<{block_name}>.*?</{block_name}>"
    res = re.sub(pattern, "", text, flags=re.DOTALL)
    if strip:
        res = res.strip()
    return res
