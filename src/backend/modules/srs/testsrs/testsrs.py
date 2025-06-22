import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Collection

from overrides import override
from typeguard import typechecked

from src.backend.modules.srs.abstract_srs import (
    AbstractCard,
    AbstractDeck,
    AbstractSRS,
    AbstractTemporaryCollection,
    CardID,
    DeckID,
    TmpCollectionID,
)


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

    @staticmethod
    def from_str(s: str):
        s = s.lower()
        for state in CardState:
            if state.value == s:
                return state
        raise ValueError(f"{s} is not a valid state.")


@dataclass(frozen=False)
@typechecked
class TestCard(AbstractCard):
    """
    A Card is a representation of a flashcard, containing a question and an answer.
    The card is uniquely identified by the id.

    Properties:
      id (str): The id uniquely identifies the card.
          It is represented as "card_xxxx_xxxx", with x being hexadecimal digits.
          The id is the only way to identify a card.
      question (str): The question (frontside) of the card.
      answer (str): The answer (frontside) of the card.
      flag (str): The flag of the card. **Must** be one of:
          none, red, orange, green, blue, pink, turquoise, purple
      cardState (str): The state of the card in the flashcard system. **Must** be one of:
          new, learning, review, suspended, buried
    """

    id: CardID
    deck: "TestDeck"
    question: str
    answer: str
    flag: Flag
    cardState: CardState
    fuzzymatch_question: bool = False
    fuzzymatch_answer: bool = False

    def __str__(self):
        return (
            f"Card  from the deck {self.deck.name} with flag {self.flag.value} and state {self.cardState.value}.\n\n"
            f"**Question**: {self.question}\n\n"
            f"**Answer**: {self.answer}"
        )

    @override
    def to_hashable(self) -> Any:
        return self.question, self.answer, self.flag, self.cardState


@dataclass(frozen=False)
@typechecked
class TestDeck(AbstractDeck):
    """
    A Deck represents a collection of flashcards.

    Properties:
       id (str): The id uniquely identifies the deck.
          It is represented as "deck_xxxx_xxxx", with x being hexadecimal digits.
          The id is the only way to identify a deck. It is assigned randomly, there is no way to guess it!
       name (str): The name of the deck. This is **not** the id, and is **not** sufficient to address decks.
       cards (List[Card]): The cards contained in the deck. The order has no meaning.
    """

    id: DeckID
    name: str
    cards: list[TestCard]

    def __str__(self):
        s = f"""Deck '{self.name}' containing {len(self.cards)} cards."""
        return s


@dataclass(frozen=False)
class TestTemporaryCollection(AbstractTemporaryCollection):
    """
    A Temporary Collection represents a collection of flashcards.
    However, the flashcards themselves are part of another deck;
    a temporary collection is a temporary collection of flashcards.
    Any changes to the cards in the temporary collection will also change the cards in their 'normal' deck.
    Temporary Collections are e.g. used to represent the result of search queries.
    Temporary Collections do not have names.

    Properties:
       id (str): The id uniquely identifies the temporary collection.
          It is represented as "tmp_collection_xxxx_xxxx", with x being hexadecimal digits.
          The id is the only way to identify a temporary collection.
          It is assigned randomly, there is no way to guess it!
       description (str): A description that may explain how this deck was created. Optional, may be left blank.
       cards (List[Card]): The cards contained in the temporary collection. The order has no meaning.
    """

    id: TmpCollectionID
    description: str
    cards: list[TestCard]

    def __str__(self):
        s = f"""Temporary Collection (id: {str(self.id)}) containing {len(self.cards)} cards."""
        if self.description.strip():
            s += "\nDescription: " + self.description
        return s


@typechecked
class TestFlashcardManager(AbstractSRS[TestTemporaryCollection, TestCard, TestDeck]):
    __cards_by_id: dict[CardID, TestCard]
    __decks_by_id: dict[DeckID, TestDeck]
    __decks_by_name: dict[str, TestDeck]
    __temp_collections_by_id: dict[TmpCollectionID, TestTemporaryCollection]
    _frozen: bool

    def __init__(self):
        super().__init__()
        self.__cards_by_id = {}
        self.__decks_by_id = {}
        self.__decks_by_name = {}
        self.__temp_collections_by_id = {}
        self._frozen = False

    # ################ ID Handling ######################
    # noinspection DuplicatedCode
    @staticmethod
    def __create_id(existing_ids: set[int]):
        attempt = 0
        while True:
            attempt += 1
            random_bytes = os.urandom(4)
            random_int = int.from_bytes(random_bytes, byteorder="big")
            if random_int not in existing_ids:
                return random_int
            if attempt >= 100:
                raise RuntimeError(f"{attempt} attempts of generating a new, unique id failed.")

    def __create_card_id(self) -> CardID:
        nr_id = self.__create_id({it.numeric_id for it in self.__cards_by_id})
        return CardID(nr_id)

    def __create_deck_id(self) -> DeckID:
        nr_id = self.__create_id({it.numeric_id for it in self.__decks_by_id})
        return DeckID(nr_id)

    def __create_temp_collection_id(self) -> TmpCollectionID:
        nr_id = self.__create_id({it.numeric_id for it in self.__temp_collections_by_id})
        return TmpCollectionID(nr_id)

    # ################ Freeze / Unfreeze ######################
    def freeze(self):
        self._frozen = True

    def is_frozen(self):
        return self._frozen

    def _check_frozen(self):
        if self._frozen:
            raise RuntimeError("The SRS is frozen. No changes can be made.")

    # ################ SRS Functions ######################
    @override
    def add_deck(self, deck_name: str) -> TestDeck:
        self._check_frozen()
        if deck_name in self.__decks_by_name:
            raise ValueError(f"Deck '{deck_name}' already exists.")

        deck = TestDeck(name=deck_name, id=self.__create_deck_id(), cards=[])
        self.__decks_by_id[deck.id] = deck
        self.__decks_by_name[deck.name] = deck
        return deck

    @override
    def deck_exists(self, deck: TestDeck) -> bool:
        return deck.id in self.__decks_by_id

    def _verify_deck_exists(self, deck: TestDeck):
        if deck.id not in self.__decks_by_id:
            raise ValueError(f"Deck {deck.id} not found.")

    @override
    def get_deck_by_name_or_none(self, deck_name: str) -> TestDeck | None:
        return self.__decks_by_name.get(deck_name, None)

    @override
    def get_deck_or_none(self, deck_id: DeckID) -> TestDeck | None:
        return self.__decks_by_id.get(deck_id, None)

    @override
    def get_all_decks(self) -> list[TestDeck]:
        return list(self.__decks_by_id.values())

    @override
    def rename_deck(self, deck: TestDeck, new_name: str) -> None:
        self._check_frozen()
        if deck.name == new_name:
            return
        if new_name in self.__decks_by_name:
            raise ValueError(f"Deck '{new_name}' already exists.")
        self.__decks_by_name[new_name] = deck
        self.__decks_by_name.pop(deck.name)
        deck.name = new_name

    @override
    def delete_deck(self, deck: TestDeck) -> None:
        self._check_frozen()
        self.__decks_by_name.pop(deck.name)
        self.__decks_by_id.pop(deck.id)
        for card in deck.cards:
            self.__cards_by_id.pop(card.id)

    @override
    def add_card(self, deck: TestDeck, question: str, answer: str) -> TestCard:
        self._check_frozen()
        self._verify_deck_exists(deck)
        card = TestCard(
            id=self.__create_card_id(),
            question=question,
            answer=answer,
            flag=Flag.NONE,
            cardState=CardState.NEW,
            deck=deck,
        )
        self.__cards_by_id[card.id] = card
        deck.cards.append(card)
        return card

    def add_full_card(
        self,
        deck: TestDeck,
        question: str,
        answer: str,
        flag: Flag,
        card_state: CardState,
        fuzzymatch_question: bool = False,
        fuzzymatch_answer: bool = False,
    ) -> TestCard:
        self._check_frozen()
        self._verify_deck_exists(deck)
        card = TestCard(
            id=self.__create_card_id(),
            question=question,
            answer=answer,
            flag=flag,
            cardState=card_state,
            deck=deck,
            fuzzymatch_question=fuzzymatch_question,
            fuzzymatch_answer=fuzzymatch_answer,
        )
        self.__cards_by_id[card.id] = card
        deck.cards.append(card)
        return card

    @override
    def card_exists(self, card: TestCard) -> bool:
        return card.id in self.__cards_by_id

    def _verify_card_exists(self, card: TestCard):
        if card.id not in self.__cards_by_id:
            raise ValueError(f"Card {card.id} not found.")

    @override
    def get_card_or_none(self, card_id: CardID) -> TestCard | None:
        return self.__cards_by_id.get(card_id, None)

    @override
    def get_deck_of_card(self, card: TestCard) -> TestDeck | None:
        self._verify_card_exists(card)
        return card.deck

    @override
    def change_deck_of_card(self, card: TestCard, new_deck: TestDeck) -> TestCard:
        self._check_frozen()
        self._verify_card_exists(card)
        self._verify_deck_exists(new_deck)
        old_deck = card.deck
        old_deck.cards.remove(card)
        new_deck.cards.append(card)
        card.deck = new_deck
        return card

    @override
    def copy_card_to(self, card: TestCard, deck: TestDeck) -> TestCard:
        """
        Copy a card to a (usually different) deck, and return the new card object.
        Only copies the content of the card, not the metadata (e.g., id).
        """
        self._check_frozen()
        self._verify_card_exists(card)
        self._verify_deck_exists(deck)
        new_card = TestCard(
            id=self.__create_card_id(),
            question=card.question,
            answer=card.answer,
            flag=card.flag,
            cardState=card.cardState,
            deck=deck,
        )
        deck.cards.append(new_card)
        return new_card

    @override
    def get_cards_in_deck(self, deck: TestDeck) -> list[TestCard]:
        self._verify_deck_exists(deck)
        return deck.cards

    @override
    def edit_card_question(self, card: TestCard, new_question: str) -> TestCard:
        self._check_frozen()
        self._verify_card_exists(card)
        card.question = new_question
        return card

    @override
    def edit_card_answer(self, card: TestCard, new_answer: str) -> TestCard:
        self._check_frozen()
        self._verify_card_exists(card)
        card.answer = new_answer
        return card

    def edit_card_flag(self, card: TestCard, new_flag: Flag) -> TestCard:
        self._check_frozen()
        self._verify_card_exists(card)
        card.flag = new_flag
        return card

    def edit_card_state(self, card: TestCard, new_state: CardState) -> TestCard:
        self._check_frozen()
        self._verify_card_exists(card)
        card.cardState = new_state
        return card

    @override
    def delete_card(self, card: TestCard) -> None:
        self._check_frozen()
        self._verify_card_exists(card)
        deck = card.deck
        deck.cards.remove(card)
        self.__cards_by_id.pop(card.id)

    @override
    def create_temporary_collection(self, description: str, cards: list[TestCard]) -> TestTemporaryCollection:
        tmp_collection_id = self.__create_temp_collection_id()
        tmp_collection = TestTemporaryCollection(id=tmp_collection_id, description=description, cards=cards)
        self.__temp_collections_by_id[tmp_collection_id] = tmp_collection
        return tmp_collection

    @override
    def get_temporary_collections(self) -> list[TestTemporaryCollection]:
        return list(self.__temp_collections_by_id.values())

    @override
    def get_temporary_collection_or_none(self, tmp_collection_id: TmpCollectionID) -> TestTemporaryCollection | None:
        return self.__temp_collections_by_id.get(tmp_collection_id, None)

    def temporary_collection_exists(self, tmp_collection: TestTemporaryCollection) -> bool:
        return tmp_collection.id in self.__temp_collections_by_id

    def _verify_temporary_collection_exists(self, tmp_collection: TestTemporaryCollection):
        if tmp_collection.id not in self.__temp_collections_by_id:
            raise ValueError(f"Temporary Collection {tmp_collection.id} not found.")

    @override
    def delete_temporary_collection(self, tmp_collection: TestTemporaryCollection):
        self.__temp_collections_by_id.pop(tmp_collection.id)

    @override
    def add_cards_to_temporary_collection(self, tmp_collection: TestTemporaryCollection, cards: Collection[TestCard]):
        self._verify_temporary_collection_exists(tmp_collection)
        for card in cards:  # fail early
            self._verify_card_exists(card)

        for card in cards:
            tmp_collection.cards.append(card)

    @override
    def remove_cards_from_temporary_collection(
        self, tmp_collection: TestTemporaryCollection, cards: Collection[TestCard]
    ):
        self._verify_temporary_collection_exists(tmp_collection)
        for card in cards:  # fail early
            self._verify_card_exists(card)

        for card in cards:
            tmp_collection.cards.remove(card)

    def copy(self):
        new_manager = TestFlashcardManager()
        for deck in self.get_all_decks():
            new_deck = new_manager.add_deck(deck.name)
            for card in deck.cards:
                new_manager.add_full_card(
                    deck=new_deck, question=card.question, answer=card.answer, flag=card.flag, card_state=card.cardState
                )
        return new_manager

    def __str__(self):
        if len(self.get_all_decks()) == 0:
            return "Empty Flashcard Manager."
        else:
            deck_str = "\n".join(["* " + str(deck) for deck in self.get_all_decks()])
            return f"Flashcard Manager with the following decks:\n{deck_str}\n"
