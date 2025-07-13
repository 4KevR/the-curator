from src.backend.modules.helpers.string_util import find_substring_in_llm_response
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

    def _search(self, card: AbstractCard) -> bool:
        if card.question is not None and card.answer is not None:
            prompt = f"""Please evaluate if the following flash card fits the search prompt.
Question: {card.question}
Answer: {card.answer}
Search prompt: {self.search_prompt}

Please return true if it fits, and else false. **Do not respond anything else**"""
        elif card.question is not None and card.answer is None:
            prompt = f"""Please evaluate if the following question of a flash card fits the search prompt.
Question: {card.question}
Search prompt: {self.search_prompt}

Please return true if it fits, and else false. **Do not respond anything else**"""
        elif card.answer is not None and card.question is None:
            prompt = f"""Please evaluate if the following answer of a flash card fits the search prompt.
Answer: {card.answer}
Search prompt: {self.search_prompt}

Please return true if it fits, and else false. **Do not respond anything else**"""
        else:
            raise ValueError("At least one of question or answer must be specified.")

        response = self.llm.generate_single(prompt).lower()
        return find_substring_in_llm_response(response, "true", "false")
