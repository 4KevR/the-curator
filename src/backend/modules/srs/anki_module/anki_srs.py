# isort: skip_file

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from anki.collection import Collection  # Must be placed at the top, or circular import will occur.
from anki.cards import Card, CardId
from anki.consts import (
    CardQueue,
    CardType,
    QUEUE_TYPE_MANUALLY_BURIED,
    QUEUE_TYPE_NEW,
    QUEUE_TYPE_LRN,
    QUEUE_TYPE_DAY_LEARN_RELEARN,
    QUEUE_TYPE_REV,
    QUEUE_TYPE_SUSPENDED,
    CARD_TYPE_NEW,
    CARD_TYPE_LRN,
    CARD_TYPE_REV,
    CARD_TYPE_RELEARNING,
    QUEUE_TYPE_SIBLING_BURIED,
    QUEUE_TYPE_PREVIEW,
)
from anki.decks import DeckId
from anki.errors import NotFoundError
from anki.exporting import AnkiPackageExporter
from anki.importing.apkg import AnkiPackageImporter
from anki.lang import set_lang
from anki.notes import Note, NoteId
from overrides import override
from typeguard import typechecked

from src.backend.modules.srs.abstract_srs import AbstractCard, AbstractDeck, AbstractSRS, CardState, Flag
from src.backend.modules.srs.abstract_srs import CardID as LocalCardID
from src.backend.modules.srs.abstract_srs import DeckID as LocalDeckID

from src.backend.modules.search.llama_index import LlamaIndexExecutor

logger = logging.getLogger(__name__)

# General directory for storing Anki collections
# base_dir\user_name\collection.anki2
_base_dir = os.getenv("ANKI_COLLECTION_PATH", "data/anki_collection")


@typechecked
class AnkiDeck(AbstractDeck):
    pass


@typechecked
class AnkiCard(AbstractCard):
    note: Note
    deck: AnkiDeck
    raw_card: Card

    __type_map = {  # map is exhaustive
        CardState.NEW: CARD_TYPE_NEW,
        CardState.LEARNING: CARD_TYPE_LRN,
        CardState.REVIEW: CARD_TYPE_REV,
        CardState.BURIED: CARD_TYPE_NEW,  # should do it
        CardState.SUSPENDED: CARD_TYPE_NEW,  # should do it
    }

    __type_map_rev = {  # map is exhaustive; there are no other internal card types.
        CARD_TYPE_NEW: CardState.NEW,
        CARD_TYPE_LRN: CardState.LEARNING,
        CARD_TYPE_REV: CardState.REVIEW,
        CARD_TYPE_RELEARNING: CardState.REVIEW,
    }

    __queue_map = {
        CardState.BURIED: QUEUE_TYPE_MANUALLY_BURIED,
        CardState.SUSPENDED: QUEUE_TYPE_SUSPENDED,
        CardState.NEW: QUEUE_TYPE_NEW,
        CardState.LEARNING: QUEUE_TYPE_LRN,
        CardState.REVIEW: QUEUE_TYPE_REV,
    }

    __queue_map_rev = {  # map is exhaustive; there are no other internal queue types.
        QUEUE_TYPE_MANUALLY_BURIED: CardState.BURIED,
        QUEUE_TYPE_SIBLING_BURIED: CardState.BURIED,
        QUEUE_TYPE_SUSPENDED: CardState.SUSPENDED,
        QUEUE_TYPE_NEW: CardState.NEW,
        QUEUE_TYPE_LRN: CardState.LEARNING,
        QUEUE_TYPE_REV: CardState.REVIEW,
        QUEUE_TYPE_DAY_LEARN_RELEARN: CardState.REVIEW,
        QUEUE_TYPE_PREVIEW: CardState.NEW,  # should be fine
    }

    __flag_map = {  # exhaustive; these are all flags that exist
        Flag.NONE: 0,
        Flag.RED: 1,
        Flag.ORANGE: 2,
        Flag.GREEN: 3,
        Flag.BLUE: 4,
        Flag.PINK: 5,
        Flag.TURQUOISE: 6,
        Flag.PURPLE: 7,
    }

    __flag_map_rev = {__v: __k for (__k, __v) in __flag_map.items()}  # map is exhaustive

    @staticmethod  # needed for constructor
    def __get_flag_raw(raw_card: Card):
        return AnkiCard.__flag_map_rev[raw_card.flags]

    def get_flag(self) -> Flag:
        return self.__get_flag_raw(self.raw_card)

    def set_flag(self, new_flag: Flag) -> None:
        """Note that this is **not yet** persisted to the Anki collection."""
        self.raw_card.flags = self.__flag_map[new_flag]

    @staticmethod  # needed for constructor
    def __get_state_raw(raw_card: Card):
        queue_state = AnkiCard.__queue_map_rev[raw_card.queue]
        type_state = AnkiCard.__type_map_rev[raw_card.type]

        if queue_state == type_state:
            return queue_state

        # queue state is more specific
        if type_state not in {CardState.LEARNING, CardState.REVIEW}:
            return queue_state

        raise RuntimeError(f"Internal states are incompatible: Raw queue {queue_state} and raw type {type_state}.")

    def get_state(self) -> CardState:
        return self.__get_state_raw(self.raw_card)

    def set_state(self, new_state: CardState) -> None:
        """Note that this is **not yet** persisted to the Anki collection."""
        self.raw_card.queue = self.__queue_map[new_state]
        self.raw_card.type = self.__type_map[new_state]

    def __init__(self, note: Note, deck: AnkiDeck, raw_card: Card):
        # If raw_card.ord == 0, then the first field is the question, the second the answer.
        # If .ord == 1, other way around.
        super().__init__(
            LocalCardID(raw_card.id),
            question=note.fields[raw_card.ord],
            answer=note.fields[1 - raw_card.ord],
            state=AnkiCard.__get_state_raw(raw_card),
            flag=AnkiCard.__get_flag_raw(raw_card),
            deck=deck,
        )
        self.note = note
        self.raw_card = raw_card

    @override
    def to_hashable(self) -> Any:
        return self.question, self.answer, self.raw_card.type, self.raw_card.queue, self.raw_card.flags

    def __str__(self):
        return (
            f"AnkiCard(id={self.id}, deck={self.deck.name}, question={self.question}, answer={self.answer}, "
            f"state={self.state}, flag={self.flag}, raw_card={self.raw_card}), "
            f"note={self.note.id})"
        )


@dataclass
@typechecked
class NoteCreationResult:
    note: Note
    cards: list[Card]


@dataclass
@typechecked
class CardsDueToday:
    new: int
    learning: int
    review: int
    relearn: int
    total: int


class NoteType(Enum):
    BASIC = "Basic"
    BASIC_REVERSED = "Basic (and reversed card)"
    # BASIC_OPTIONAL_REVERSED = "Basic (optional reversed card)"
    # BASIC_TYPE_IN_ANSWER = "Basic (type in the answer)"
    # CLOZE = "Cloze"
    # IMAGE_OCCLUSION = "Image Occlusion"


class AnkiSRS(AbstractSRS[AnkiCard, AnkiDeck]):
    """
    Implements an AbstractSRS for Anki.
    The following additional methods are implemented:
    """

    dir: str
    col: Collection

    def __init__(self, anki_user: str):
        """
        Initialize an Anki instance with a user-specific data directory.

        If the directory for the given user does not exist, it will be created.
        A new or existing Anki collection (collection.anki2) will be loaded from this path.
        """
        if anki_user == "":
            raise ValueError("anki_user cannot be empty string.")

        anki_directory = os.path.join(_base_dir, anki_user)

        if os.path.exists(anki_directory):
            logger.debug(f"Anki directory {anki_directory} already exists.")
        else:
            logger.debug(f"Creating new anki directory {anki_directory}.")
            os.makedirs(anki_directory)

        self.dir = anki_directory
        collection_path = os.path.join(self.dir, "collection.anki2")
        logger.debug(f"Collection path: {os.path.abspath(collection_path)}")
        self.col = Collection(collection_path)
        self.llama_index_executor = LlamaIndexExecutor(anki_user)

    # Decks
    @override
    def add_deck(self, deck_name: str) -> AnkiDeck:
        logger.debug(f"Trying to add deck: '{deck_name}'")
        if deck_name == "":
            raise ValueError("deck_name cannot be empty string.")

        if self.get_deck_by_name_or_none(deck_name) is not None:
            raise ValueError(f"Deck '{deck_name}' already exists.")

        deck_id = self.col.decks.id(deck_name)
        logger.debug(f"Deck '{deck_name}' is added.")
        anki_deck = AnkiDeck(LocalDeckID(deck_id), deck_name)
        self.llama_index_executor.add_deck(anki_deck)
        return anki_deck

    @override
    def deck_exists(self, deck: AnkiDeck) -> bool:
        deck = self.col.decks.by_name(deck.name)
        return deck is not None

    def _verify_deck_exists(self, deck: AnkiDeck) -> None:
        """Raises ValueError if the deck does not exist."""
        if not self.deck_exists(deck):
            raise ValueError(f"Deck '{deck.name}' does not exist.")

    @override
    def get_deck_by_name_or_none(self, deck_name: str) -> AnkiDeck | None:
        logger.debug(f"Looking up deck by name: '{deck_name}'...")
        deck_dict = self.col.decks.by_name(deck_name)
        if deck_dict is None:
            logger.debug(f"Deck '{deck_name}' not found.")
            return None
        return AnkiDeck(LocalDeckID(deck_dict["id"]), deck_dict["name"])

    @override
    def get_deck_by_id_or_none(self, deck_id: LocalDeckID) -> AnkiDeck | None:
        logger.debug(f"Looking up deck by ID: {deck_id.numeric_id}...")
        deck_dict = self.col.decks.get(DeckId(deck_id.numeric_id))
        if deck_dict is None:
            logger.debug(f"Deck with ID {deck_id.numeric_id} not found.")
            return None
        return AnkiDeck(LocalDeckID(deck_dict["id"]), deck_dict["name"])

    @override
    def get_all_decks(self) -> list[AnkiDeck]:
        """Returns all deck names and corresponding IDs."""
        decks = self.col.decks.all_names_and_ids()
        return [AnkiDeck(LocalDeckID(deck.id), deck.name) for deck in decks]

    @override
    def rename_deck(self, deck: AnkiDeck, new_name: str) -> None:
        self._verify_deck_exists(deck)
        logger.debug(f"Rename deck '{deck.name}' (ID={deck.id.numeric_id}) to '{new_name}'")
        self.col.decks.rename(deck.id.numeric_id, new_name)
        deck.name = new_name
        self.llama_index_executor.modify_deck(deck)

    @override
    def delete_deck(self, deck: AnkiDeck) -> None:
        self._verify_deck_exists(deck)
        if deck.name == "Default":
            raise ValueError("The default deck cannot be deleted.")
        self.col.decks.remove([deck.id.numeric_id])
        self.llama_index_executor.remove_deck(deck.id)

    # Cards
    @override
    def add_card(self, deck: AnkiDeck, question: str, answer: str, flag: Flag, state: CardState) -> AnkiCard:
        logger.debug(f"Adding card to deck '{deck.name}'...")
        # noinspection PyTypeChecker
        new = self.add_note(deck, question, answer)
        assert len(new.cards) == 1  # Now only "Basic" model is supported

        card = AnkiCard(new.note, deck, new.cards[0])

        if flag != Flag.NONE:
            self.edit_card_flag(card, flag)

        if state != CardState.NEW:
            self.edit_card_state(card, state)

        self.llama_index_executor.add_card(card)

        return self.get_card(card.id)

    @override
    def card_exists(self, card: AnkiCard) -> bool:
        card2 = self.get_card_or_none(card.id)
        return card2 is not None

    def _verify_card_exists(self, card: AnkiCard) -> None:
        if not self.card_exists(card):
            raise ValueError(f"Card '{card.id}' does not exist.")

    @override
    def get_card_or_none(self, card_id: LocalCardID) -> AnkiCard | None:
        try:
            logger.debug(f"Looking up card ID {card_id.numeric_id}...")
            card = self.col.get_card(CardId(card_id.numeric_id))
            deck_dict = self.col.decks.get(card.did)

            deck = AnkiDeck(LocalDeckID(deck_dict["id"]), deck_dict["name"])
            note = self.col.get_note(card.nid)

            return AnkiCard(note, deck, card)

        except NotFoundError:
            logger.debug(f"Card with ID {card_id.numeric_id} not found.")
            return None

    @override
    def get_deck_of_card(self, card: AnkiCard) -> AnkiDeck:
        self._verify_card_exists(card)
        deck_dict = self.col.decks.get(card.raw_card.did)
        return AnkiDeck(LocalDeckID(deck_dict["id"]), deck_dict["name"])

    @override
    def change_deck_of_card(self, card: AnkiCard, new_deck: AnkiDeck) -> AnkiCard:
        self._verify_card_exists(card)
        self._verify_deck_exists(new_deck)

        new_did = new_deck.id.numeric_id
        raw_card = card.raw_card
        raw_card.did = new_did if new_did else 1
        self.col.update_card(raw_card)
        logger.debug(f"Change deck of card ID={card.id.numeric_id} from '{card.deck.name}' to '{new_deck.name}'.")

        modified_card = AnkiCard(card.note, new_deck, card.raw_card)
        self.llama_index_executor.modify_card(modified_card)
        return modified_card

    @override
    def get_cards_in_deck(self, deck: AnkiDeck) -> list[AnkiCard]:
        self._verify_deck_exists(deck)
        card_ids = self.col.find_cards(f'deck:"{deck.name}"')
        logger.debug(f"Retrieved {len(card_ids)} cards from deck '{deck.name}'.")

        return [self.get_card_or_none(LocalCardID(card_id)) for card_id in card_ids]

    @override
    def edit_card_question(self, card: AnkiCard, new_question: str) -> AnkiCard:
        self._verify_card_exists(card)
        self.edit_note(card.note.id, question=new_question)
        modified_card = self.get_card_or_none(card.id)
        self.llama_index_executor.modify_card(modified_card)
        return modified_card

    @override
    def edit_card_answer(self, card: AnkiCard, new_answer: str) -> AnkiCard:
        self._verify_card_exists(card)
        self.edit_note(card.note.id, answer=new_answer)
        modified_card = self.get_card_or_none(card.id)
        self.llama_index_executor.modify_card(modified_card)
        return modified_card

    @override
    def edit_card_flag(self, card: AnkiCard, new_flag: Flag) -> AnkiCard:
        self._verify_card_exists(card)
        card.set_flag(new_flag)
        self.col.update_card(card.raw_card)
        modified_card = self.get_card_or_none(card.id)
        self.llama_index_executor.modify_card(modified_card)
        return modified_card

    @override
    def edit_card_state(self, card: AnkiCard, new_state: CardState) -> AnkiCard:
        self._verify_card_exists(card)
        card.set_state(new_state)
        self.col.update_card(card.raw_card)
        return self.get_card_or_none(card.id)

    @override
    def copy_card_to(self, card: AnkiCard, deck: AnkiDeck) -> AnkiCard:
        self._verify_card_exists(card)
        self._verify_deck_exists(deck)
        logger.debug(f"Copying card ID={card.id.numeric_id} from deck '{card.deck.name}' to deck '{deck.name}'...")
        new_card = self.add_card(deck, card.question, card.answer, card.flag, card.state)
        self.llama_index_executor.add_card(new_card)
        return new_card

    @override
    def delete_card(self, card: AnkiCard) -> None:
        self._verify_card_exists(card)
        self.delete_cards_by_ids([card.id.numeric_id])  # returns the amount of cards deleted
        self.llama_index_executor.remove_card(card.id)

    def delete_cards_by_ids(self, card_ids: list[int]) -> int:
        """
        Delete the specified cards.
        If the card is the last card from a note, this note will also be automatically deleted.

        :param card_ids: List of card IDs to delete
        :return: Number of cards deleted
        """
        deleted_cards = []
        old_note_ids = set(self.list_all_notes())
        for card_id in card_ids:
            try:
                # noinspection PyProtectedMember
                result = self.col._backend.remove_cards([card_id])
                if result.count != 0:
                    deleted_cards.append(card_id)
            except Exception as e:
                logger.debug(f"Card ID {card_id} is invalid: {e}")

        deleted_notes = sorted(set(old_note_ids) - set(self.list_all_notes()))
        logger.debug(f"Delete Cards: {deleted_cards}. " + f"Automatically deleted notes: {deleted_notes}")
        return len(deleted_cards)

    # ###########################################################
    # ################# ANKI-Specific Functions #################
    # ###########################################################

    def export_deck_to_apkg(self, deck: AnkiDeck, path: str = None) -> None:
        """Export the specified deck to a .apkg file.
        Path should include the file name, for example "/tmp/mydeck.apkg".
        """
        self._verify_deck_exists(deck)

        if path is None:
            path = os.path.join(self.dir, f"{deck.name}.apkg")

        exp = AnkiPackageExporter(self.col)
        exp.did = deck.id.numeric_id
        exp.exportInto(path)
        logger.debug(f"Deck {deck.name} is exported to {path}.")

    def import_deck_from_apkg(self, path: str) -> None:
        """Import a deck from .apkg file.
        If you can't find the deck after importing, please check if it is an empty deck.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find file: {path}")

        importer = AnkiPackageImporter(self.col, path)
        set_lang("en_US")  # If not set, an error will be reported.
        importer.run()
        logger.debug(f"Deck is imported from {path}.")

    # Note
    @typechecked
    def add_note(
        self, deck: AnkiDeck, front: str, back: str, model_name: NoteType = NoteType.BASIC
    ) -> NoteCreationResult:
        """
        Create a Note with the specified NoteType (model).
        Front/back correspond to the field content respectively.

        Notably, creating a note will automatically create cards in the specified deck.

        return: the note_id of the new Note,
                and automatically added card_ids from this new note.
        """
        # Make sure Deck exists
        self._verify_deck_exists(deck)

        # Set NoteType
        model = self.col.models.by_name(model_name.value)
        if model is None:
            raise ValueError(f"Cannot find NoteType: {model_name.value}.")

        # Add note
        note: Note = self.col.new_note(model)
        note.fields[0] = front
        note.fields[1] = back

        # 1. When adding a note, Anki will automatically generate one or more cards
        # based on the NoteType template to which the note belongs.
        # 2. The note_id may equal the first card's card_id.
        self.col.add_note(note, DeckId(deck.id.numeric_id))
        logger.debug(f"Note {note.id} is added.")
        cards = note.cards()
        card_ids = [card.id for card in cards]
        logger.debug(f"Automatically added Cards: {card_ids}")

        return NoteCreationResult(note=note, cards=cards)

    def delete_notes_by_ids(self, note_ids: list[int]) -> None:
        """
        Delete the specified notes (and all their cards).

        :param note_ids: List of note IDs to delete
        """
        if note_ids:
            for note_id in note_ids:
                try:
                    note = self.col.get_note(NoteId(note_id))
                    card_ids = [card.id for card in note.cards()]
                    logger.debug(f"Note {note_id} is deleted. Also deleted Cards: {card_ids}.")
                except Exception as e:
                    # Note might already be deleted or invalid
                    logger.debug(f"Note ID {note_id} is invalid: {e}")

            self.col.remove_notes(note_ids)

    def list_all_notes(self) -> list[int]:
        """List all notes in anki.collection."""
        note_ids = self.col.find_notes("")  # The empty string matches all notes
        return [it.real for it in note_ids]

    def list_notes_for_cards_in_deck(self, deck_name: str) -> list[Note]:
        """List all Notes for cards in the specified Deck,
        returning a list of (note_id, front, back)."""
        deck = self.get_deck_by_name_or_none(deck_name)
        if deck is None:
            return []

        card_ids = self.col.find_cards(f'deck:"{deck_name}"')
        note_ids = {self.col.get_card(cid).nid for cid in card_ids}

        result = []
        for nid in note_ids:
            note = self.col.get_note(nid)
            if len(note.fields) >= 2:
                result.append(note)
        return result

    def get_note_id_by_card_id(self, card_id: int) -> int | None:
        """Given a card ID, return the Note ID it belongs to."""
        try:
            card = self.col.get_card(CardId(card_id))
            return card.nid
        except NotFoundError:
            logger.debug(f"Card ID {card_id} is invalid.")
            return None

    def edit_note(self, note_id: int, question: str = "", answer: str = "") -> None:
        """Edit the question and answer pair.
        When you change a note, all cards will change accordingly."""
        note = self.col.get_note(NoteId(note_id))
        logger.debug(f"Editing note ID={note_id}...'")

        if question.strip():
            note.fields[0] = question  # The first field → front (question)
            logger.debug(f"New note_question: {question}")
        if answer.strip():
            note.fields[1] = answer  # The second field → back (answer)
            logger.debug(f"New note_answer: {answer}")

        self.col.update_note(note)

    def list_card_ids_from_note(self, note_id: int) -> list[int]:
        """List all card IDs from the specified note."""
        note = self.col.get_note(NoteId(note_id))
        card_ids = [card.id for card in note.cards()]
        return card_ids

    def set_memory_grade(self, card_id: int, ease: str) -> None:
        """
        Simulate user memory feedback:
        - 'again': can't remember (try again)
        - 'hard': difficult
        - 'good': remember (normal)
        - 'easy': very easy
        """
        grade_map = {
            "again": 0,
            "hard": 1,
            "good": 2,
            "easy": 3,
        }
        if ease not in grade_map:
            raise ValueError("The memory level must be: again / hard / good / easy.")

        card = self.col.get_card(CardId(card_id))
        card.answer = grade_map[ease]
        logger.debug(f"Set CardID_{card.id} memory grade: {ease}")
        self.col.update_card(card)

    # noinspection SqlNoDataSourceInspection
    def activate_preview_cards(self, deck_name: str) -> None:
        """
        Activate all new cards in queue=0 (Preview) of the specified deck to queue=1 (New),
        so that it can enter the normal learning process.
        """
        self.col.db.execute(
            "UPDATE cards SET queue = 1 WHERE did = ? AND type = 0 AND queue = 0",
            self.get_deck_by_name(deck_name).id.numeric_id,
        )

    # noinspection SqlNoDataSourceInspection
    def count_cards_due_today(self, deck_name: str) -> CardsDueToday:
        """How many cards need to be learned today."""
        today = self.col.sched.today

        # Query all active cards
        cards = self.col.db.list(
            "SELECT id FROM cards WHERE did = ? AND queue IN (1, 2, 3)",
            self.get_deck_by_name(deck_name).id.numeric_id,
        )

        count = {key: 0 for key in ("new", "learning", "review", "relearn")}

        for card in cards:
            card = self.col.get_card(card)

            if card.queue == 1:
                count["new"] += 1
            elif card.due <= today:
                if card.type == 3:
                    count["relearn"] += 1
                elif card.queue == 2:
                    count["learning"] += 1
                elif card.queue == 3:
                    count["review"] += 1

        count["total"] = sum(count.values())

        return CardsDueToday(
            new=count["new"],
            learning=count["learning"],
            review=count["review"],
            relearn=count["relearn"],
            total=count["total"],
        )

    # # TODO fix this
    def learning_process(self):
        raise NotImplementedError("Working")

        self.activate_preview_cards(self.user_context.current_deck.name)

        cards = self.col.db.list(
            "SELECT id FROM cards WHERE did = ? AND queue IN (1, 2, 3)",
            self.get_deck_by_name(self.user_context.current_deck.name).id.numeric_id,
        )

        for card_id in cards:
            # card = self.col.get_card(card_id)

            # set self.user_context.current_card
            # get question and answer
            ...

            # 1.tts
            # 2.wait user response
            # 3.llm judge
            # 4.set memory grade
            # 5.ask user if he wants to set flag
            # 6.set flag if needed

    # ###########################################################
    # ######### Following functions are not used ################
    # ###########################################################
    # TODO @Zicheng please consider removing these functions. They are unused, and I can't imagine how we would use them
    def set_type(self, card_id: int, type_code: int) -> None:
        """
        0: "New", # New card
        1: "Learn", # Learning
        2: "Review", # Review
        3: "Relearn" # Relearn, once mastered but forgotten
        """
        assert type_code in [0, 1, 2, 3]
        card = self.col.get_card(CardId(card_id))
        card.type = CardType(type_code)
        self.col.update_card(card)

    def set_queue(self, card_id: int, queue_code: int) -> None:
        """
        We ignored "BURIED": -3 -2.

        -1: "Suspended",  # Manually suspended; excluded from scheduling
        0: "Preview",    # In preview mode (typically via filtered decks)
        1: "New",        # New card, not yet learned
        2: "Learning",   # In learning phase with steps
        3: "Review",     # Due for review based on interval
        4: "Filtered",   # In a filtered deck; temporary scheduling
        """
        assert queue_code in [-1, 0, 1, 2, 3, 4]
        card = self.col.get_card(CardId(card_id))
        card.queue = CardQueue(queue_code)
        self.col.update_card(card)

    def set_due(self, card_id: int, due: int) -> None:
        """Set the due date (in days) for a specific card."""
        card = self.col.get_card(CardId(card_id))
        card.due = due
        self.col.update_card(card)

    def set_interval(self, card_id: int, ivl: int) -> None:
        """Set the review interval (in days) for a specific card."""
        card = self.col.get_card(CardId(card_id))
        card.ivl = ivl
        self.col.update_card(card)

    def set_review_stats(self, card_id: int, reps: int = None, lapses: int = None, left: int = None) -> None:
        """
        :reps: total number of reviews
        :lapses: number of abandonments (forgetting)
        :left: number of remaining study times for the day
        """
        card = self.col.get_card(CardId(card_id))
        if reps is not None:
            card.reps = reps
        if lapses is not None:
            card.lapses = lapses
        if left is not None:
            card.left = left
        self.col.update_card(card)
