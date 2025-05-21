import logging
import os

from anki.collection import Collection
from anki.errors import NotFoundError
from anki.exporting import AnkiPackageExporter
from anki.importing.apkg import AnkiPackageImporter
from anki.lang import set_lang
from anki.notes import Note

from src.backend.domain.abstract_adapters import AbstractAnki
from src.backend.domain.srs import (
    CardInfo,
    CardsDueToday,
    DeckCardsInfo,
    DeckInfo,
    NoteCreationResult,
    NoteInfo,
)

logger = logging.getLogger(__name__)

# General directory for storing Anki collections
# base_dir\user_name\collection.anki2
base_dir = os.getenv("ANKI_COLLECTION_PATH", "data/anki_collection")


"""
The following methods are implemented:
- Deck: add, 
        delete, 
        list_all_decks, 
        get_deck_id,
        rename,
        export/import deck
        # sub deck is now not supported.

- Note: add, 
        delete, 
        list_all_notes,
        list_notes_for_cards_in_deck, 
        get_note_id_by_card_id

- Card: delete,
        list_card_ids_from_note,
        list_cards_in_deck,
        get_card_info,
        set_xxx,
        activate_preview_cards,
        count_cards_due_today
        # More will be added later if necessary.
"""


class Anki(AbstractAnki):
    def __init__(self, user_name: str):
        if user_name == "":
            raise ValueError("User_name cannot be empty string.")

        user_dir = os.path.join(base_dir, user_name)
        if os.path.exists(user_dir):
            logger.debug("User already exists.")
        else:
            logger.debug("User does not exist, will create...")
            os.makedirs(user_dir)
        self.user_dir = user_dir
        collection_path = os.path.join(user_dir, "collection.anki2")
        self.col = Collection(collection_path)

    # Deck
    def add_deck(self, deck_name: str) -> int:
        """
        If the deck corresponding to deck_name already exists, return its ID;
        Otherwise, create a new deck with the given name and return its ID.
        """
        if deck_name == "":
            raise ValueError("Deck_name cannot be empty string.")

        deck_id = self.get_deck_id(deck_name)
        if deck_id is None:
            deck_id = self.col.decks.id(deck_name)
            logger.debug(f"Deck '{deck_name}' is added.")
        else:
            logger.debug(f"Deck '{deck_name}' already exists.")

        return deck_id

    def delete_deck(self, deck_name: str) -> None:
        """Delete the specified Deck and all cards in it"""
        deck_id = self.get_deck_id(deck_name)
        if not deck_id:
            logger.debug(f"Deck '{deck_name}' does not exist.")
            return
        else:
            self.col.decks.remove((deck_id,))
            logger.debug(f"Deck '{deck_name}' is deleted.")

    def get_deck_id(self, deck_name: str) -> int | None:
        """
        Get the deck_id corresponding to the specified deck name.
        If the deck does not exist, return None.
        """
        deck = self.col.decks.by_name(deck_name)
        if deck is not None:
            return deck["id"]
        return None

    def list_all_decks(self) -> list[DeckInfo]:
        """Returns all deck names and corresponding IDs."""
        decks = self.col.decks.all_names_and_ids()
        return [DeckInfo(name=deck.name, id=deck.id) for deck in decks]

    def rename_deck(self, old_name: str, new_name: str) -> None:
        # Rename
        deck_id = self.get_deck_id(old_name)
        if deck_id is None:
            raise ValueError(f"Cannot find deck: {old_name}")
        self.col.decks.rename(deck_id, new_name)
        logger.debug(f"Deck '{old_name}' has been renamed to '{new_name}'.")

    def export_deck_to_apkg(self, deck_name: str, path: str = None) -> None:
        """Export the specified deck to a .apkg file.
        Path should include the file name, for example "/tmp/mydeck.apkg".
        """
        deck_id = self.get_deck_id(deck_name)
        if deck_id is None:
            raise ValueError(f"Cannot find deck: {deck_name}")

        if path is None:
            path = os.path.join(self.user_dir, f"{deck_name}.apkg")

        exp = AnkiPackageExporter(self.col)
        exp.did = deck_id
        exp.exportInto(path)
        logger.debug(f"Deck {deck_name} is exported to {path}.")

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
    def add_note(
        self, deck_name: str, front: str, back: str, model_name: str = "Basic"
    ) -> NoteCreationResult:
        """
        Create a Note with the specified NoteType (model).
        Front/back correspond to the field content respectively.

        Notably, creating a note will automatically create cards in the specified deck.

        return: the note_id of the new Note,
                and automatically added card_ids from this new note.
        """
        # Make sure Deck exists
        deck_id = self.get_deck_id(deck_name)
        if not deck_id:
            logger.debug(f"Deck '{deck_name}' does not exist.")
            raise ValueError(f"Cannot find Deck: {deck_name}.")

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
        # 2. The note_id may equals the first card's card_id.
        self.col.add_note(note, deck_id)
        logger.debug(f"Note {note.id} is added.")
        card_ids = [card.id for card in note.cards()]
        logger.debug(f"Automatically added Cards: {card_ids}")

        return NoteCreationResult(note_id=note.id, card_ids=card_ids)

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

    def list_notes_for_cards_in_deck(self, deck_name: str) -> list[NoteInfo]:
        """List all Notes for cards in the specified Deck,
        returning a list of (note_id, front, back)."""
        deck = self.get_deck_id(deck_name)
        if not deck:
            return []

        card_ids = self.col.find_cards(f"deck:{deck_name}")
        note_ids = {self.col.get_card(cid).nid for cid in card_ids}

        result = []
        for nid in note_ids:
            note = self.col.get_note(nid)
            if len(note.fields) >= 2:
                result.append(
                    NoteInfo(note_id=note.id, front=note.fields[0], back=note.fields[1])
                )
        return result

    def get_note_id_by_card_id(self, card_id: int) -> int | None:
        """Given a card ID, return the Note ID it belongs to."""
        try:
            card = self.col.get_card(card_id)
            return card.nid
        except NotFoundError:
            logger.debug(f"Card ID {card_id} is invalid.")
            return None

    # Card
    def delete_cards_by_ids(self, card_ids: list[int]) -> None:
        """Delete the specified cards.
        If the card is the last card from a note,
        this note will also be automatically deleted.

        :param card_ids: List of card IDs to delete
        """
        deleted_cards = []
        old_note_ids = self.list_all_notes()
        if card_ids:
            for card_id in card_ids:
                try:
                    result = self.col._backend.remove_cards([card_id])
                    if result.count != 0:
                        deleted_cards.append(card_id)
                except Exception as e:
                    logger.debug(f"Card ID {card_id} is invalid: {e}")

        new_note_ids = self.list_all_notes()
        deleted_notes = [x for x in old_note_ids if x not in new_note_ids]
        logger.debug(
            f"Delete Cards: {deleted_cards}. "
            + f"Automatically deleted notes: {deleted_notes}"
        )

    def list_card_ids_from_note(self, note_id: int) -> list[int]:
        """List all card IDs from the specified note."""
        if note_id:
            note = self.col.get_note(note_id)
            card_ids = [card.id for card in note.cards()]
            return card_ids

    def list_cards_in_deck(self, deck_name: str) -> DeckCardsInfo | None:
        """
        List all card IDs in the specified deck.

        :param deck_name: deck name
        :return: list of card IDs belonging to the deck
        """
        deck_id = self.get_deck_id(deck_name)
        if not deck_id:
            return None

        card_ids = self.col.find_cards(f"deck:{deck_name}")

        return DeckCardsInfo(total_cards=len(card_ids), card_ids=card_ids)

    def get_card_info(self, card_id: int) -> CardInfo:
        """
        Get detailed information of the card and return a dictionary containing:
        - card_id, note_id, deck_id, template_index
        - card type (new card/studying/reviewing/relearning)
        - queue type (queue number and name)
        - due (due), interval (ivl)
        - factor (ease), review/study times (reps, lapses, left)
        - flags (flags), tags (tags) and note field content (fields)
        """
        try:
            card = self.col.get_card(card_id)
            note = self.col.get_note(card.nid)

            type_map = {
                0: "New",  # New card
                1: "Learn",  # Learning
                2: "Review",  # Review
                3: "Relearn",  # Relearn, once mastered but forgotten
            }

            queue_map = {
                -1: "Suspended",  # Not participating in review
                0: "Preview",  # Preview
                1: "New",  # New cards waiting for first learning
                2: "Learning",  # In the learning queue
                3: "Review",  # In the review queue
                4: "Filtered",
            }

            return CardInfo(
                card_id=card.id,
                note_id=card.nid,
                deck_id=card.did,
                template_index=card.ord,
                type={"code": card.type, "name": type_map.get(card.type, "Unknown")},
                queue={
                    "code": card.queue,
                    "name": queue_map.get(card.queue, "Unknown"),
                },
                due=card.due,  # due number/days
                ivl=card.ivl,  # current interval (days)
                ease=card.factor,  # factor
                reps=card.reps,  # total number of reviews
                lapses=card.lapses,  # number of abandonments
                left=card.left,  # number of remaining study times for the day
                flags=card.flags,  # user tags
                tags=note.tags,  # list of tags for this note
                fields=note.fields,  # all fields of this note
            )
        except NotFoundError:
            return None

    def set_type(self, card_id: int, type_code: int) -> None:
        assert type_code in [0, 1, 2, 3]
        card = self.col.get_card(card_id)
        card.type = type_code
        self.col.update_card(card)

    def set_queue(self, card_id: int, queue_code: int) -> None:
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

    def set_ease_factor(self, card_id: int, factor: int) -> None:
        """Modify ease (factor, integer, e.g. 2500 means 2.5 times)"""
        card = self.col.get_card(card_id)
        card.factor = factor
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

    def activate_preview_cards(self, deck_name: str) -> None:
        """
        Activate all new cards in queue=0 (Preview)
        of the specified deck to queue=1 (New),
        so that it can enter the normal learning process.
        """
        self.col.db.execute(
            "UPDATE cards SET queue = 1 WHERE did = ? AND type = 0 AND queue = 0",
            self.get_deck_id(deck_name),
        )

    def count_cards_due_today(self, deck_name: str) -> CardsDueToday:
        """How many cards need to be learned today."""
        today = self.col.sched.today

        # Query all active cards
        cards = self.col.db.list(
            "SELECT id FROM cards WHERE did = ? AND queue IN (1, 2, 3)",
            self.get_deck_id(deck_name),
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
