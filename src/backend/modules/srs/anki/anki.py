import logging
import os
from dataclasses import dataclass
from overrides import override

from anki.collection import Collection
from anki.errors import NotFoundError
from anki.exporting import AnkiPackageExporter
from anki.importing.apkg import AnkiPackageImporter
from anki.lang import set_lang
from anki.notes import Note
import anki.cards
from typeguard import typechecked

from src.backend.modules.srs.abstract_srs import DeckID, CardID, TDeck, AbstractDeck, AbstractCard, TCard
from src.backend.modules.srs.abstract_srs import AbstractSRS
from src.backend.modules.ai_assistant.llm_cmd_registration import llm_command

logger = logging.getLogger(__name__)

# General directory for storing Anki collections
# base_dir\user_name\collection.anki2
_base_dir = os.getenv("ANKI_COLLECTION_PATH", "data/anki_collection")


@typechecked
class AnkiDeck(AbstractDeck):
    def __init__(self, deck_id: DeckID, name: str):
        super().__init__(deck_id, name)


@typechecked
class AnkiCard(AbstractCard):
    note: Note
    deck: AnkiDeck
    raw_card: anki.cards.Card

    __type_map = {
        0: "New",  # New card
        1: "Learn",  # Learning
        2: "Review",  # Review
        3: "Relearn",  # Relearn, once mastered but forgotten
    }

    __queue_map = {
        -1: "Suspended",  # Not participating in review
        0: "Preview",  # Preview
        1: "New",  # New cards waiting for first learning
        2: "Learning",  # In the learning queue
        3: "Review",  # In the review queue
        4: "Filtered",
    }

    def __init__(self, note: Note, deck: AnkiDeck, raw_card: anki.cards.Card):
        super().__init__(
            CardID(raw_card.id),
            question=note.fields[raw_card.ord],
            answer=note.fields[1 - raw_card.ord]
        )  # If raw_card.ord == 0, then the first field is the question, the second the answer. If .ord == 1, other way around.
        self.note = note
        self.deck = deck
        self.raw_card = raw_card

    @property
    def type(self) -> str:
        return self.__type_map.get(self.raw_card.type, "Unknown")

    @property
    def queue(self) -> str:
        return self.__queue_map.get(self.raw_card.queue, "Unknown")

    def __str__(self) -> str:
        return f"AnkiCard(id={self.id}, question={self.question}, answer={self.answer}, deck={self.deck.name}, note={self.note}, raw_card={self.raw_card}, type={self.type}, queue={self.queue})"


@dataclass
@typechecked
class NoteCreationResult:
    note: Note
    cards: list[anki.cards.Card]


@dataclass
@typechecked
class CardsDueToday:
    new: int
    learning: int
    review: int
    relearn: int
    total: int


class Anki(AbstractSRS[AnkiCard, AnkiDeck]):
    """
    Implements an AbstractSRS for Anki.

    The following additional methods are implemented:
    TODO
    """

    def __init__(self, user_name: str):
        """
        Initializes a new Anki object for the specified user.

        Parameters:
        user_name : str
            The name of the user. It may not be an empty string.
            If the user does not exist, it will be created.

        Raises:
        ValueError
            If the user_name is an empty string.
        """
        if user_name == "":
            raise ValueError("user_name cannot be empty string.")

        user_dir = os.path.join(_base_dir, user_name)
        if os.path.exists(user_dir):
            logger.debug(f"User {user_name} already exists.")
        else:
            logger.debug(f"Creating new user {user_name}.")
            os.makedirs(user_dir)
        self.user_dir = user_dir
        collection_path = os.path.join(user_dir, "collection.anki2")
        logger.debug(f"Collection path: {os.path.abspath(collection_path)}")
        self.col = Collection(collection_path)

    # Decks
    @override
    def add_deck(self, deck_name: str) -> AnkiDeck:
        if deck_name == "":
            raise ValueError("deck_name cannot be empty string.")

        if self.get_deck(deck_name) is not None:
            raise ValueError(f"Deck '{deck_name}' already exists.")

        deck_id = self.col.decks.id(deck_name)
        logger.debug(f"Deck '{deck_name}' is added.")

        return AnkiDeck(DeckID(deck_id), deck_name)

    @override
    def deck_exists(self, deck: TDeck) -> bool:
        deck = self.col.decks.by_name(deck.name)
        return deck is not None

    def _verify_deck_exists(self, deck: AnkiDeck) -> None:
        if not self.deck_exists(deck):
            raise ValueError(f"Deck '{deck.name}' does not exist.")

    @override
    def get_deck(self, deck_name: str) -> AnkiDeck | None:
        deck_dict = self.col.decks.by_name(deck_name)
        if deck_dict is None: return None
        return AnkiDeck(DeckID(deck_dict["id"]), deck_dict["name"])

    @override
    def get_all_decks(self) -> list[AnkiDeck]:
        """Returns all deck names and corresponding IDs."""
        decks = self.col.decks.all_names_and_ids()
        # return [Deck(name=deck.name, id=deck.id) for deck in decks] # TODO
        return [AnkiDeck(deck.id, deck.name) for deck in decks]

    @override
    def rename_deck(self, deck: AnkiDeck, new_name: str) -> None:
        self._verify_deck_exists(deck)
        self.col.decks.rename(deck.id.numeric_id, new_name)

    @override
    def delete_deck(self, deck: AnkiDeck) -> None:
        self._verify_deck_exists(deck)
        self.col.decks.remove([deck.id.numeric_id])

    # new deck methods
    def export_deck_to_apkg(self, deck: AnkiDeck, path: str = None) -> None:
        """Export the specified deck to a .apkg file.
        Path should include the file name, for example "/tmp/mydeck.apkg".
        """
        self._verify_deck_exists(deck)

        if path is None:
            path = os.path.join(self.user_dir, f"{deck.name}.apkg")

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
    # TODO: Can you please use an enum for model_name?
    # Running
    # ", ".join([it[1]["name"] for it in self.col.models.models.items()])
    # gives
    # 'Basic, Basic (and reversed card), Basic (optional reversed card), Basic (type in the answer), Cloze, Image Occlusion'
    def add_note(
            self, deck: AnkiDeck, front: str, back: str, model_name: str = "Basic"
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
        model = self.col.models.by_name(model_name)
        if model is None:
            raise ValueError(f"Cannot find NoteType: {model_name}.")

        # Add note
        note: Note = self.col.new_note(model)
        note.fields[0] = front
        note.fields[1] = back

        # 1. When adding a note, Anki will automatically generate one or more cards
        # based on the NoteType template to which the note belongs.
        # 2. The note_id may equal the first card's card_id.
        self.col.add_note(note, deck.id.numeric_id)
        logger.debug(f"Note {note.id} is added.")
        logger.warning("FABIAN TODO: Please check that this actually works!!!!")  # TODO
        cards = note.cards()
        logger.debug(f"Automatically added Cards: {cards}")

        return NoteCreationResult(note=note, cards=cards)

    def delete_notes_by_ids(self, note_ids: list[int]) -> None:
        """
        Delete the specified notes (and all their cards).

        :param note_ids: List of note IDs to delete
        """
        if note_ids:
            for note_id in note_ids:
                try:
                    note = self.col.get_note(note_id)
                    card_ids = [card.id for card in note.cards()]
                    logger.debug(
                        f"Note {note_id} is deleted. Also deleted Cards: {card_ids}."
                    )
                except Exception as e:
                    # Note might already be deleted or invalid
                    logger.debug(f"Note ID {note_id} is invalid: {e}")

            self.col.remove_notes(note_ids)

    def list_all_notes(self) -> list[int]:
        """List all notes in collection."""
        note_ids = self.col.find_notes("")  # The empty string matches all notes
        return note_ids

    def list_notes_for_cards_in_deck(self, deck_name: str) -> list[Note]:
        """List all Notes for cards in the specified Deck,
        returning a list of (note_id, front, back)."""
        deck = self.get_deck(deck_name)
        if not deck:
            return []

        card_ids = self.col.find_cards(f"deck:{deck_name}")
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
            card = self.col.get_card(card_id)
            return card.nid
        except NotFoundError:
            logger.debug(f"Card ID {card_id} is invalid.")
            return None

    def edit_note(self, note_id: int, question: str = "", answer: str = "") -> None:
        """Edit the question & answer pair.
        When you change a note, all cards will change accordingly."""
        note = self.col.get_note(note_id)

        if question.strip():
            note.fields[0] = question  # The first field → front (question)
        if answer.strip():
            note.fields[1] = answer  # The second field → back (answer)

        self.col.update_note(note)

    # Card
    @override
    def add_card(self, deck: AnkiDeck, question: str, answer: str) -> AnkiCard:
        new = self.add_note(deck, question, answer, model_name="Basic")

        return AnkiCard(new.note, deck, new.cards[0])

    @override
    def get_card(self, card_id: CardID) -> AnkiCard | None:
        try:
            card = self.col.get_card(card_id.numeric_id)
            raw_deck = self.col.decks.get(card.did)
            deck = AnkiDeck(DeckID(raw_deck["id"]), raw_deck["name"])
            note = self.col.get_note(card.nid)

            return AnkiCard(note, deck, card)
        except NotFoundError:
            return None

    @override
    def card_exists(self, card: AnkiCard) -> bool:
        card2 = self.get_card(card.id.numeric_id)
        return card2 is not None

    def _verify_card_exists(self, card: AnkiCard) -> None:
        if not self.card_exists(card):
            raise ValueError(f"Card '{card.id}' does not exist.")

    @override
    def get_deck_of_card(self, card: AnkiCard) -> TDeck | None:
        self._verify_card_exists(card)
        try:
            return self.col.decks.get(card.raw_card.did)
        except NotFoundError:
            return None

    @override
    def get_cards_in_deck(self, deck: TDeck) -> list[TCard]:
        self._verify_deck_exists(deck)
        card_ids = self.col.find_cards(f"deck:{deck.name}")
        return [self.get_card(CardID(card_id)) for card_id in card_ids]

    @override
    def delete_card(self, card: TCard) -> bool:
        self._verify_card_exists(card)
        return self.delete_cards_by_ids([card.id.numeric_id]) >= 1

    def delete_cards_by_ids(self, card_ids: list[int]) -> int:
        """Delete the specified cards.
        If the card is the last card from a note,
        this note will also be automatically deleted.

        :param card_ids: List of card IDs to delete
        :return: Number of cards deleted
        """
        deleted_cards = []
        old_note_ids = set(self.list_all_notes())
        for card_id in card_ids:
            try:
                result = self.col._backend.remove_cards([card_id])
                if result.count != 0:
                    deleted_cards.append(card_id)
            except Exception as e:
                logger.debug(f"Card ID {card_id} is invalid: {e}")

        deleted_notes = sorted(set(old_note_ids) - set(self.list_all_notes()))
        logger.debug(
            f"Delete Cards: {deleted_cards}. "
            + f"Automatically deleted notes: {deleted_notes}"
        )
        return len(deleted_cards)

    # TODO: Do we need any of these functions?

    def list_card_ids_from_note(self, note_id: int) -> list[int]:
        """List all card IDs from the specified note."""
        note = self.col.get_note(note_id)
        card_ids = [card.id for card in note.cards()]
        return card_ids

    def set_type(self, card_id: int, type_code: int) -> None:
        """
        0: "New", # New card
        1: "Learn", # Learning
        2: "Review", # Review
        3: "Relearn" # Relearn, once mastered but forgotten
        """
        assert type_code in [0, 1, 2, 3]
        card = self.col.get_card(card_id)
        card.type = type_code
        self.col.update_card(card)

    def set_queue(self, card_id: int, queue_code: int) -> None:
        """
        -1: "Suspended", # Not participating in review
        0: "Preview", # Preview
        1: "New", # New cards waiting for first learning
        2: "Learning", # In the learning queue
        3: "Review", # In the review queue
        4: "Filtered"
        """
        assert queue_code in [-1, 0, 1, 2, 3, 4]
        card = self.col.get_card(card_id)
        card.queue = queue_code
        self.col.update_card(card)

    def set_deck(self, card_id: int, new_deck_name: str) -> None:
        card = self.col.get_card(card_id)
        new_id = self.get_deck_id(new_deck_name)
        card.did = new_id if new_id else 1
        self.col.update_card(card)

    def set_due(self, card_id: int, due: int) -> None:
        card = self.col.get_card(card_id)
        card.due = due
        self.col.update_card(card)

    def set_interval(self, card_id: int, ivl: int) -> None:
        card = self.col.get_card(card_id)
        card.ivl = ivl
        self.col.update_card(card)

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

        card = self.col.get_card(card_id)
        card.answer = grade_map[ease]
        self.col.update_card(card)

    def set_review_stats(
            self, card_id: int, reps: int = None, lapses: int = None, left: int = None
    ) -> None:
        """
        :reps: total number of reviews
        :lapses: number of abandonments (forgetting)
        :left: number of remaining study times for the day
        """
        card = self.col.get_card(card_id)
        if reps is not None:
            card.reps = reps
        if lapses is not None:
            card.lapses = lapses
        if left is not None:
            card.left = left
        self.col.update_card(card)

    def set_flag(self, card_id: int, flag: int) -> None:
        """Set flag:"""
        flag_map = {
            "none": 0,
            "red": 1,
            "orange": 2,
            "green": 3,
            "blue": 4,
            "pink": 5,
            "cyan": 6,
            "purple": 7,
        }
        if flag not in flag_map:
            raise ValueError(f"Invalid flag: {flag}")

        card = self.col.get_card(card_id)
        card.flags = flag_map[flag]
        self.col.update_card(card)

    def activate_preview_cards(self, deck_name: str) -> None:
        """
        Activate all new cards in queue=0 (Preview)
        of the specified deck to queue=1 (New),
        so that it can enter the normal learning process.
        """
        self.col.db.execute(
            "UPDATE cards SET queue = 1 WHERE did = ? AND type = 0 AND queue = 0",
            self.get_deck(deck_name).id.numeric_id,
        )

    def count_cards_due_today(self, deck_name: str) -> CardsDueToday:
        """How many cards need to be learned today."""
        today = self.col.sched.today

        # Query all active cards
        cards = self.col.db.list(
            "SELECT id FROM cards WHERE did = ? AND queue IN (1, 2, 3)",
            self.get_deck(deck_name).id.numeric_id,
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
