def replace_many(s: str, replacements: dict[str, str]) -> str:
    """
    Replace multiple substrings in a string. There is no guarantee on the order of replacements.
    """
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s
