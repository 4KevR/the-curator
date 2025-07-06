import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Generic, Optional, TypeVar

from overrides.final import final
from typeguard import typechecked

TCard = TypeVar("TCard", bound="AbstractCard")  # Must be a subtype of AbstractCard
TDeck = TypeVar("TDeck", bound="AbstractDeck")  # Must be a subtype of AbstractDeck


@typechecked
@dataclass(frozen=True)
class DeckID:
    """Represents a deck identifier that can be stored as both integer and hexadecimal format.

    The deck ID is stored internally as a 32-bit integer but can be represented as an 8-digit
    hexadecimal string prefixed with 'deck_'. For example:
    - Integer format: 65535
    - Hex format: deck_0000ffff

    The hex string format supports an optional underscore separator after the first 4 digits,
    e.g., 'deck_0000_ffff'.
    """

    numeric_id: int
    __DECK_ID_REGEX: ClassVar[re.Pattern] = re.compile(r"^deck_[0-9a-fA-F]{4}_?[0-9a-fA-F]{4}$")

    def hex_id(self):
        """Returns a hex string of the deck id (32 bits, 8 hex digits)."""
        return f"deck_{self.numeric_id:08x}"

    @staticmethod
    def from_hex_string(hex_str: str):
        """Returns a DeckID from a hex string."""
        if not DeckID.__DECK_ID_REGEX.match(hex_str):
            raise ValueError(f"Invalid deck ID format: {hex_str}")
        hex_nr = hex_str[5:].replace("_", "")
        return DeckID(int(hex_nr, 16))


@typechecked
@dataclass(frozen=True)
class CardID:
    """Represents a card identifier that can be stored as both integer and hexadecimal format.

    The card ID is stored internally as a 32-bit integer but can be represented as an 8-digit
    hexadecimal string prefixed with 'card_'. For example:
    - Integer format: 65535
    - Hex format: card_0000ffff

    The hex string format supports an optional underscore separator after the first 4 digits,
    e.g., 'card_0000_ffff'.
    """

    numeric_id: int
    __CARD_ID_REGEX: ClassVar[re.Pattern] = re.compile(r"^card_[0-9a-fA-F]{4}_?[0-9a-fA-F]{4}$")

    def hex_id(self):
        """Returns a hex string of the card id (32 bits, 8 hex digits)."""
        return f"card_{self.numeric_id:08x}"

    @staticmethod
    def from_hex_string(hex_str: str):
        """Returns a CardID from a hex string."""
        if not CardID.__CARD_ID_REGEX.match(hex_str):
            raise ValueError(f"Invalid card ID format: {hex_str}")
        hex_nr = hex_str[5:].replace("_", "")
        return CardID(int(hex_nr, 16))


@typechecked
class Flag(Enum):
    NONE = "none"
    RED = "red"
    ORANGE = "orange"
    GREEN = "green"
    BLUE = "blue"
    PINK = "pink"
    TURQUOISE = "turquoise"
    PURPLE = "purple"

    @staticmethod
    def from_str(s: str):
        s = s.lower()
        for flag in Flag:
            if flag.value == s:
                return flag
        raise ValueError(f"{s} is not a valid flag.")


@typechecked
class CardState(Enum):
    NEW = "new"
    LEARNING = "learning"
    REVIEW = "review"
    SUSPENDED = "suspended"
    BURIED = "buried"
    RELEARN = "relearn"

    @staticmethod
    def from_str(s: str):
        s = s.lower()
        for state in CardState:
            if state.value == s:
                return state
        raise ValueError(f"{s} is not a valid state.")


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
    flag: Flag
    state: CardState

    def __init__(self, card_id: CardID, question: str, answer: str, flag: Flag, state: CardState):
        self.id = card_id
        self.question = question
        self.answer = answer
        self.flag = flag
        self.state = state

    @abstractmethod
    def to_hashable(self) -> Any:
        """Returns a hashable representation of the card's content. May **not** contain the card's id."""
        raise NotImplementedError


class MissingDeckException(Exception):

    def __init__(self, deck_name: str | None = None, deck_id: DeckID | None = None, *args: object) -> None:
        self.deck_name = deck_name
        self.deck_id = deck_id
        if deck_name is None and deck_id is None:
            raise ValueError("At least one of deck_name or deck_id must be specified.")
        super().__init__(*args)

    def __str__(self):
        if self.deck_name is not None and self.deck_id is not None:
            return f"Deck '{self.deck_name}' with id {self.deck_id} does not exist."
        elif self.deck_name is not None:
            return f"Deck '{self.deck_name}' does not exist."
        elif self.deck_id is not None:
            return f"Deck with id {self.deck_id} does not exist."
        else:
            raise AssertionError("Unreachable.")


class AbstractSRS(Generic[TCard, TDeck], ABC):
    """Abstract class for a spaced repetition system (SRS), such as Anki."""

    study_mode: bool = False
    cards_to_be_learned: Optional[list[TCard]] = None  # ugly.. just initial ideas
    card_index_currently_being_learned: int = -1  # ugly.. just initial ideas

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

    @final
    def get_deck_by_name(self, deck_name: str) -> TDeck:
        """
        Retrieve a deck by name.
        If the deck does not exist, an MissingDeckException is thrown.
        """
        deck = self.get_deck_by_name_or_none(deck_name)
        if deck is None:
            raise MissingDeckException(deck_name=deck_name)
        return deck

    @abstractmethod
    def get_deck_by_name_or_none(self, deck_name: str) -> TDeck | None:
        """
        Retrieve a deck by name.
        If the deck does not exist, return None.
        """

    @final
    def get_deck_by_id(self, deck_id: DeckID) -> TDeck:
        """
        Retrieve a deck by id.
        If the deck does not exist, an MissingDeckException is thrown.
        """
        deck = self.get_deck_by_id_or_none(deck_id)
        if deck is None:
            raise MissingDeckException(deck_id=deck_id)
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
    def add_card(self, deck: TDeck, question: str, answer: str, flag: Flag, state: CardState) -> TCard:
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
    def edit_card_flag(self, card: TCard, new_flag: Flag) -> TCard:
        """Edit the flag of a card."""

    @abstractmethod
    def edit_card_state(self, card: TCard, new_state: CardState) -> TCard:
        """Edit the state of a card."""

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

    def set_memory_grade(self, card_id: CardID, ease: str) -> None:
        """
        Simulate user memory feedback. (no card scheduling.)
        """
        grade_map = {
            "again": 0,
            "hard": 1,
            "good": 2,
            "easy": 3,
        }
        if ease not in grade_map:
            raise ValueError("The memory level must be: again / hard / good / easy.")
