def create_card_generation_prompt(max_cards: int, content: str) -> str:
    return f"""
    Please generate Anki flashcards.
    Requirements:
    1. Use a concise question-and-answer format; each card should include a clear 
    question and an accurate answer;
    2. Questions should be as specific as possible, avoiding vague or broad topics;
    3. Generate no more than {max_cards} cards;
    4. The output format should be as follows:
    Q: ...\nA: ...\n\nQ: ...\nA: ...
    Content:\n{content}
    """
