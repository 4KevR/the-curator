from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import ClassVar

from typeguard import typechecked

TCard = TypeVar('TCard', bound='AbstractCard')  # Must be a subtype of AbstractCard
TDeck = TypeVar('TDeck', bound='AbstractDeck')  # Must be a subtype of AbstractDeck


@typechecked
@dataclass
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

    def hexstr(self):
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
@dataclass
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

    def hexstr(self):
        """Returns a hex string of the card id (32 bits, 8 hex digits)."""
        return f"card_{self.numeric_id:08x}"

    @staticmethod
    def from_hex_string(hex_str: str):
        """Returns a CardID from a hex string."""
        if not CardID.__CARD_ID_REGEX.match(hex_str):
            raise ValueError(f"Invalid card ID format: {hex_str}")
        hex_nr = hex_str[5:].replace("_", "")
        return CardID(int(hex_nr, 16))


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


class AbstractSRS(Generic[TCard, TDeck], ABC):
    """Abstract class for a spaced repetition system (SRS), such as Anki."""

    # Decks
    @abstractmethod
    def add_deck(self, deck_name: str) -> TDeck:
        """
        Create a new deck with the given name.

        Returns the newly created deck.

        Raises ValueError if a deck with the given name already exists.
        """
        raise NotImplementedError

    @abstractmethod
    def deck_exists(self, deck: TDeck) -> bool:
        """Check if the given deck exists."""

    @abstractmethod
    def get_deck(self, deck_name: str) -> TDeck | None:
        """
        Retrieve a deck by name.
        If the deck does not exist, return None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_decks(self) -> list[TDeck]:
        """ Retrieve all decks."""
        raise NotImplementedError

    @abstractmethod
    def rename_deck(self, deck: TDeck, new_name: str) -> None:
        """Rename a deck."""
        raise NotImplementedError

    @abstractmethod
    def delete_deck(self, deck: TDeck) -> None:
        """Delete a deck. If the deck does not exist, raise a value error."""
        raise NotImplementedError

    # Cards
    @abstractmethod
    def add_card(self, deck: TDeck, question: str, answer: str) -> TCard:
        """Add a card with a given question and answer to the given deck."""
        raise NotImplementedError

    @abstractmethod
    def card_exists(self, card: TCard) -> bool:
        """Check if a card exists."""
        raise NotImplementedError

    @abstractmethod
    def get_card(self, card_id: CardID) -> TCard | None:
        """Retrieve a card by its ID."""
        raise NotImplementedError

    @abstractmethod
    def get_deck_of_card(self, card: TCard) -> TDeck | None:
        """Retrieve the deck of a card. Returns None if the card is not in a deck."""
        raise NotImplementedError

    @abstractmethod
    def get_cards_in_deck(self, deck: TDeck) -> list[TCard]:
        """Retrieve all cards in a deck."""
        raise NotImplementedError

    @abstractmethod
    def delete_card(self, card: TCard) -> bool:
        """
        Delete a card.
        Returns True if the card was deleted, False otherwise.
        """
        raise NotImplementedError
