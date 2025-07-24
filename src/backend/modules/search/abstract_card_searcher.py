from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.backend.modules.srs.abstract_srs import AbstractCard

C = TypeVar("C", bound=AbstractCard)


class AbstractCardSearcher(ABC, Generic[C]):
    """
    Abstract class for card searchers.
    The searcher consumes a list of cards and returns 'fitting' cards.
    """

    def search_all(self, cards: list[C]) -> list[C]:
        return [c for c in cards if self._search(c)]

    @abstractmethod
    def _search(self, card: C) -> bool:
        raise NotImplementedError

    @staticmethod
    def union_search_all(searchers: list["AbstractCardSearcher"], all_cards: list[AbstractCard]):
        """
        Returns all cards that are found by any of the searchers.
        Short-circuiting is used where possible. Searchers are used in order.
        """
        return [card for card in all_cards if any(searcher._search(card) for searcher in searchers)]
