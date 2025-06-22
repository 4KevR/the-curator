from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Collection, Generic, TypeVar

from typeguard import typechecked

TCard = TypeVar("TCard", bound="AbstractCard")  # Must be a subtype of AbstractCard
TDeck = TypeVar("TDeck", bound="AbstractDeck")  # Must be a subtype of AbstractDeck
TTmpCol = TypeVar("TTmpCol", bound="AbstractTemporaryCollection")  # Must be a subtype of AbstractTemporaryCollection


@typechecked
@dataclass(frozen=True)
class DeckID:
    """Simple deck identifier based on an integer."""

    numeric_id: int

    def __int__(self) -> int:
        """Returns the integer value of the deck ID."""
        return self.numeric_id

    def __str__(self) -> str:
        """Returns a string representation of the deck ID."""
        return f"DeckID_{self.numeric_id}"


@typechecked
@dataclass(frozen=True)
class CardID:
    """Simple card identifier based on an integer."""

    numeric_id: int

    def __int__(self) -> int:
        """Returns the integer value of the card ID."""
        return self.numeric_id

    def __str__(self) -> str:
        """Returns a string representation of the card ID."""
        return f"CardID_{self.numeric_id}"


@typechecked
@dataclass(frozen=True)
class TmpCollectionID:
    """Simple temporary collection identifier based on an integer."""

    numeric_id: int

    def __int__(self) -> int:
        """Returns the integer value of the temporary collection ID."""
        return self.numeric_id

    def __str__(self) -> str:
        """Returns a string representation of the temporary collection ID."""
        return f"TmpCol_{self.numeric_id}"


class AbstractDeck(ABC):
    """A deck in a spaced repetition system."""

    id: DeckID
    name: str

    def __init__(self, deck_id: DeckID, name: str):
        self.id = deck_id
        self.name = name


class AbstractCard(ABC):
    """A card in a spaced repetition system"""

    id: CardID
    question: str
    answer: str

    def __init__(self, card_id: CardID, question: str, answer: str):
        self.id = card_id
        self.question = question
        self.answer = answer

    def to_hashable(self) -> Any:
        """Returns a hashable representation of the card's content. May **not** contain the card's id."""
        raise NotImplementedError


class AbstractTemporaryCollection(ABC):
    """A temporary collection of cards in a spaced repetition system, e.g. for a search result."""

    id: TmpCollectionID
    description: str

    def __init__(self, tmp_collection_id: TmpCollectionID, description: str):
        self.id = tmp_collection_id
        self.description = description


class AbstractSRS(Generic[TTmpCol, TCard, TDeck], ABC):
    """Abstract class for a spaced repetition system (SRS), such as Anki."""

    # Decks
    @abstractmethod
    def add_deck(self, deck_name: str) -> TDeck:
        """
        Create a new deck with the given name.
        Returns the newly created deck.
        Raises ValueError if a deck with the given name already exists.
        """

    @abstractmethod
    def deck_exists(self, deck: TDeck) -> bool:
        """Check if the given deck exists."""

    def get_deck_by_name(self, deck_name: str) -> TDeck:
        """
        Retrieve a deck by name.
        If the deck does not exist, an error is thrown.
        """
        deck = self.get_deck_by_name_or_none(deck_name)
        if deck is None:
            raise ValueError(f"Deck '{deck_name}' does not exist.")
        return deck

    @abstractmethod
    def get_deck_by_name_or_none(self, deck_name: str) -> TDeck | None:
        """
        Retrieve a deck by name.
        If the deck does not exist, return None.
        """

    def get_deck_by_id(self, deck_id: DeckID) -> TDeck:
        """
        Retrieve a deck by id.
        If the deck does not exist, an error is thrown.
        """
        deck = self.get_deck_by_id_or_none(deck_id)
        if deck is None:
            raise ValueError(f"Deck with id {deck_id} does not exist.")
        return deck

    @abstractmethod
    def get_deck_by_id_or_none(self, deck_id: DeckID) -> TDeck | None:
        """
        Retrieve a deck by id.
        If the deck does not exist, return None.
        """

    @abstractmethod
    def get_all_decks(self) -> list[TDeck]:
        """Retrieve all decks."""

    @abstractmethod
    def rename_deck(self, deck: TDeck, new_name: str) -> None:
        """Rename a deck."""

    @abstractmethod
    def delete_deck(self, deck: TDeck) -> None:
        """Delete the specified Deck and all cards in it"""

    # Cards
    @abstractmethod
    def add_card(self, deck: TDeck, question: str, answer: str) -> TCard:
        """Add a card with a given question and answer to the given deck."""

    @abstractmethod
    def card_exists(self, card: TCard) -> bool:
        """Check if a card exists."""

    def get_card(self, card_id: CardID) -> TCard:
        """Retrieve a card by its ID. Fails if the card is not present in any deck."""
        card = self.get_card_or_none(card_id)
        if card is None:
            raise ValueError(f"Card with id {card_id} does not exist.")
        return card

    @abstractmethod
    def get_card_or_none(self, card_id: CardID) -> TCard | None:
        """Retrieve a card by its ID. Returns None if the card is not present in any deck."""

    @abstractmethod
    def get_cards_in_deck(self, deck: TDeck) -> list[TCard]:
        """Retrieve all cards in a deck."""

    @abstractmethod
    def edit_card_question(self, card: TCard, new_question: str) -> TCard:
        """Edit the question of a card."""

    @abstractmethod
    def edit_card_answer(self, card: TCard, new_answer: str) -> TCard:
        """Edit the answer of a card."""

    @abstractmethod
    def get_deck_of_card(self, card: TCard) -> TDeck:
        """Retrieve the deck of a card."""

    @abstractmethod
    def change_deck_of_card(self, card: TCard, new_deck: TDeck) -> TCard:
        """Change the deck of a card, and return the updated card object."""

    @abstractmethod
    def copy_card_to(self, card: TCard, deck: TDeck) -> TCard:
        """
        Copy a card to a (usually different) deck, and return the new card object.
        Only copies the content of the card, not the metadata (e.g., id).
        """

    @abstractmethod
    def delete_card(self, card: TCard) -> None:
        """
        Delete a card.
        Raises a ValueError if the card is not present in any deck.
        """

    # ###########################################################
    # ################# Temporary Collections ###################
    # ###########################################################
    def create_temporary_collection(self, description: str, cards: list[TCard]) -> TTmpCol:
        """Create a new temporary collection with the given description and cards. Cards may be empty."""
        raise NotImplementedError

    def get_temporary_collections(self) -> list[TTmpCol]:
        """Retrieve all temporary collections."""
        raise NotImplementedError

    def get_temporary_collection(self, tmp_collection_id: TmpCollectionID) -> TTmpCol:
        """Retrieve a temporary collection by its ID. Fails if the temporary collection does not exist."""
        tmp_col = self.get_temporary_collection_or_none(tmp_collection_id)
        if tmp_col is None:
            raise ValueError(f"Temporary collection with id {tmp_collection_id} does not exist.")
        return tmp_col

    def get_temporary_collection_or_none(self, tmp_collection_id: TmpCollectionID) -> TTmpCol | None:
        """Retrieve a temporary collection by its ID. Returns None if the temporary collection does not exist."""
        raise NotImplementedError

    def delete_temporary_collection(self, tmp_collection: TTmpCol) -> None:
        """Delete a temporary collection. Raises a ValueError if the temporary collection does not exist."""
        raise NotImplementedError

    def add_cards_to_temporary_collection(self, tmp_collection: TTmpCol, cards: Collection[TCard]) -> None:
        raise NotImplementedError

    def remove_cards_from_temporary_collection(self, tmp_collection: TTmpCol, cards: Collection[TCard]) -> None:
        raise NotImplementedError
