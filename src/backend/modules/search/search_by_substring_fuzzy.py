import rapidfuzz
from src.backend.modules.search.abstract_card_searcher import AbstractCardSearcher
from src.backend.modules.srs.abstract_srs import AbstractCard


class SearchBySubstringFuzzy(AbstractCardSearcher[AbstractCard]):
    """
    Fuzzy search by substring. For details on the fuzzy search, see:
    rapidfuzz.fuzz.partial_ratio

    Usually, a fuzzy search value of 0.8 is useful.
    """

    def __init__(
            self, search_substring: str, search_in_question: bool,
            search_in_answer: bool, case_sensitive: bool, fuzzy: float
    ):
        if not (fuzzy is None or 0.0 <= fuzzy <= 1.0):
            raise ValueError("If fuzzy is set, it must be between 0 and 1.")
        self.search_substring = search_substring if not case_sensitive else search_substring.lower()
        self.search_in_question = search_in_question
        self.search_in_answer = search_in_answer
        self.case_sensitive = case_sensitive
        self.fuzzy = fuzzy

    def __fuzzy_search(self, text: str) -> bool:
        return rapidfuzz.fuzz.partial_ratio(self.search_substring, text) >= self.fuzzy * 100.0

    def search(self, card: AbstractCard) -> bool:
        if self.search_in_question:
            search_question = card.question if self.case_sensitive else card.question.lower()
            if self.__fuzzy_search(search_question): return True

        if self.search_in_answer:
            search_answer = card.answer if self.case_sensitive else card.answer.lower()
            if self.__fuzzy_search(search_answer): return True

        return False
