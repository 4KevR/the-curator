from typing import TypeVar, Generic, Collection
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import ClassVar
from typeguard import typechecked

TCard = TypeVar('TCard', bound='AbstractCard')  # Must be a subtype of AbstractCard
TDeck = TypeVar('TDeck', bound='AbstractDeck')  # Must be a subtype of AbstractDeck
TTmpCol = TypeVar('TTmpCol', bound='AbstractTemporaryCollection')  # Must be a subtype of AbstractDeck


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
@dataclass(frozen=True)
class TmpCollectionID:
    """Represents a temporary collection identifier that can be stored as both integer and hexadecimal format.

    The tmp collection ID is stored internally as a 32-bit integer but can be represented as an 8-digit
    hexadecimal string prefixed with 'tmp_collection_'. For example:
    - Integer format: 65535
    - Hex format: tmp_collection_0000ffff

    The hex string format supports an optional underscore separator after the first 4 digits,
    e.g., 'tmp_collection_0000_ffff'.
    """
    numeric_id: int
    __TMP_COLLECTION_ID_REGEX: ClassVar[re.Pattern] = re.compile(r"^tmp_collection_[0-9a-fA-F]{4}_?[0-9a-fA-F]{4}$")

    def hex_id(self):
        """Returns a hex string of the deck id (32 bits, 8 hex digits)."""
        return f"tmp_collection_{self.numeric_id:08x}"

    @staticmethod
    def from_hex_string(hex_str: str):
        """Returns a DeckID from a hex string."""
        if not TmpCollectionID.__TMP_COLLECTION_ID_REGEX.match(hex_str):
            raise ValueError(f"Invalid temporary collection ID format: {hex_str}")
        hex_nr = hex_str[len("tmp_collection_"):].replace("_", "")
        return TmpCollectionID(int(hex_nr, 16))


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
        raise NotImplementedError

    @abstractmethod
    def deck_exists(self, deck: TDeck) -> bool:
        """Check if the given deck exists."""

    @abstractmethod
    def get_deck_by_name(self, deck_name: str) -> TDeck | None:
        """
        Retrieve a deck by name.
        If the deck does not exist, return None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_deck(self, deck_id: DeckID) -> TDeck | None:
        """
        Retrieve a deck by id.
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
    def change_deck_of_card(self, card: TCard, new_deck: TDeck) -> TCard:
        """Change the deck of a card, and return the updated card object."""
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

    # Temporary Collections
    @abstractmethod
    def create_temporary_collection(self, description: str, cards: list[TCard]) -> TTmpCol:
        """Create a new temporary collection with the given description and cards. Cards may be empty."""
        raise NotImplementedError

    @abstractmethod
    def get_temporary_collections(self) -> list[TTmpCol]:
        """Retrieve all temporary collections."""
        raise NotImplementedError

    @abstractmethod
    def get_temporary_collection(self, tmp_collection_id: TmpCollectionID) -> TTmpCol:
        """Retrieve a temporary collection by its ID."""
        raise NotImplementedError

    @abstractmethod
    def delete_temporary_collection(self, tmp_collection: TTmpCol):
        """Delete a temporary collection."""
        raise NotImplementedError

    @abstractmethod
    def add_cards_to_temporary_collection(self, tmp_collection: TTmpCol, cards: Collection[TCard]):
        raise NotImplementedError

    @abstractmethod
    def remove_cards_from_temporary_collection(self, tmp_collection: TTmpCol, cards: Collection[TCard]):
        raise NotImplementedError
