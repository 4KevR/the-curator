from src.backend.modules.search.abstract_card_searcher import AbstractCardSearcher
from src.backend.modules.srs.abstract_srs import AbstractCard


class SearchBySubstring(AbstractCardSearcher[AbstractCard]):
    """
    Searches for substrings in the question and answer.
    The substring must be fully contained in question or answer; it is not possible to contain half the substring in
    the question and half in the answer.
    """

    def __init__(
        self,
        search_substring: str,
        search_in_question: bool,
        search_in_answer: bool,
        case_sensitive: bool,
    ):
        self.search_substring = search_substring if not case_sensitive else search_substring.lower()
        self.search_in_question = search_in_question
        self.search_in_answer = search_in_answer
        self.case_sensitive = case_sensitive

    def _search(self, card: AbstractCard) -> bool:
        if self.search_in_question:
            search_question = card.question if self.case_sensitive else card.question.lower()
            if self.search_substring in search_question:
                return True

        if self.search_in_answer:
            search_answer = card.answer if self.case_sensitive else card.answer.lower()
            if self.search_substring in search_answer:
                return True

        return False
