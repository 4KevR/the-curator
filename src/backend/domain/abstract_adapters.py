from abc import ABC, abstractmethod
from typing import Optional

from src.backend.domain.srs import (
    CardInfo,
    CardsDueToday,
    DeckCardsInfo,
    DeckInfo,
    NoteCreationResult,
    NoteInfo,
)


class AbstractLLM(ABC):
    """Abstract class for LLM adapters."""

    @abstractmethod
    def generate(self, messages: list) -> str:
        """Generate text using the LLM."""
        raise NotImplementedError


class AbstractASR(ABC):
    """Abstract class for ASR adapters."""

    @abstractmethod
    def transcribe(self, audio_chunk: str, duration: int) -> str:
        """Transcribe audio to text."""
        raise NotImplementedError


class AbstractPDFReader(ABC):
    """Abstract class for PDF reader adapters."""

    @abstractmethod
    def read(
        self, file_path: str, page_range: Optional[tuple[int, int]] = None
    ) -> dict:
        """Read text from a PDF file."""
        raise NotImplementedError


class AbstractAnki(ABC):
    """Abstract class for Anki adapters."""

    @abstractmethod
    def add_deck(self, deck_name: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def delete_deck(self, deck_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_deck_id(self, deck_name: str) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def list_all_decks(self) -> list[DeckInfo]:
        raise NotImplementedError

    @abstractmethod
    def rename_deck(self, old_name: str, new_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_deck_to_apkg(self, deck_name: str, path: str = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def import_deck_from_apkg(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_note(
        self, deck_name: str, front: str, back: str, model_name: str = "Basic"
    ) -> NoteCreationResult:
        raise NotImplementedError

    @abstractmethod
    def delete_notes_by_ids(self, note_ids: list[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_all_notes(self) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def list_notes_for_cards_in_deck(self, deck_name: str) -> list[NoteInfo]:
        raise NotImplementedError

    @abstractmethod
    def get_note_id_by_card_id(self, card_id: int) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def delete_cards_by_ids(self, card_ids: list[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_card_ids_from_note(self, note_id: int) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def list_cards_in_deck(self, deck_name: str) -> DeckCardsInfo | None:
        raise NotImplementedError

    @abstractmethod
    def get_card_info(self, card_id: int) -> CardInfo:
        raise NotImplementedError

    @abstractmethod
    def set_type(self, card_id: int, type_code: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_queue(self, card_id: int, queue_code: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_deck(self, card_id: int, new_deck_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_due(self, card_id: int, due: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_interval(self, card_id: int, ivl: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_review_stats(
        self, card_id: int, reps: int = None, lapses: int = None, left: int = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def activate_preview_cards(self, deck_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def count_cards_due_today(self, deck_name: str) -> CardsDueToday:
        raise NotImplementedError

    @abstractmethod
    def edit_note(self, note_id: int, question: str = "", answer: str = "") -> None:
        raise NotImplementedError

    @abstractmethod
    def set_memory_grade(self, card_id: int, ease: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_flag(self, card_id: int, flag: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_cards_by_ids(self, card_ids: list[int]) -> list:
        raise NotImplementedError

    @abstractmethod
    def get_card_content(self, card_id: int) -> list:
        raise NotImplementedError
