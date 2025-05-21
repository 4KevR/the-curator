from dataclasses import dataclass


@dataclass
class DeckInfo:
    name: str
    id: int


@dataclass
class NoteInfo:
    note_id: int
    front: str
    back: str


@dataclass
class CardInfo:
    card_id: int
    note_id: int
    deck_id: int
    template_index: int
    type: dict
    queue: dict
    due: int
    ivl: int
    ease: int
    reps: int
    lapses: int
    left: int
    flags: int
    tags: list[str]
    fields: list[str]


@dataclass
class CardsDueToday:
    new: int
    learning: int
    review: int
    relearn: int
    total: int


@dataclass
class NoteCreationResult:
    note_id: int
    card_ids: list[int]


@dataclass
class DeckCardsInfo:
    total_cards: int
    card_ids: list[int]
