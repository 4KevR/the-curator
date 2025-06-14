from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.search.abstract_card_searcher import AbstractCardSearcher
from src.backend.modules.srs.abstract_srs import AbstractCard


class LLMSearchByContent(AbstractCardSearcher[AbstractCard]):
    """
    Searches for content in AbstractCards by asking an LLM if the card content fits the search prompt.

    The search may fail if the LLM returns an unfitting response.
    """

    def __init__(self, llm: AbstractLLM, search_prompt: str, search_in_question: bool, search_in_answer: bool):
        self.llm = llm
        self.search_prompt = search_prompt
        self.search_in_question = search_in_question
        self.search_in_answer = search_in_answer

    def search(self, card: AbstractCard) -> bool:
        if card.question is not None and card.answer is not None:
            prompt = f"""Please evaluate if the following flash card fits the search prompt.
Question: {card.question}
Answer: {card.answer}
Search prompt: {self.search_prompt}

Please return true if it fits, and else false."""
        elif card.question is not None and card.answer is None:
            prompt = f"""Please evaluate if the following question of a flash card fits the search prompt.
Question: {card.question}
Search prompt: {self.search_prompt}

Please return true if it fits, and else false."""
        elif card.answer is not None and card.question is None:
            prompt = f"""Please evaluate if the following answer of a flash card fits the search prompt.
Answer: {card.answer}
Search prompt: {self.search_prompt}

Please return true if it fits, and else false."""
        else:
            raise ValueError("At least one of question or answer must be specified.")

        messages = [
            {"role": "user", "content": prompt},
        ]
        response = self.llm.generate(messages).lower()

        false_index = response.rfind("false")
        true_index = response.rfind("true")

        if false_index != -1 and true_index != -1:
            raise ValueError(f"Unexpected llm response: {response!r}")

        return true_index > false_index
