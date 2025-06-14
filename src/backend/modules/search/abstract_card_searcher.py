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
        return [c for c in cards if self.search(c)]

    @abstractmethod
    def search(self, card: C) -> bool:
        raise NotImplementedError
