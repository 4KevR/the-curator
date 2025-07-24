import pathlib

from src.backend.modules.evaluation.load_test_data.import_data_classes import _Test_Data
from src.backend.modules.srs.abstract_srs import CardState, Flag
from src.backend.modules.srs.anki_module import AnkiSRS

USER_TO_SETUP: str = "PRESENTATION"
PATH = "./tests/data/tests.json"
SKIP_DECKS = [
    "Latin Literature",
    "Java Programming 10",
]

if __name__ == "__main__":
    json_path = pathlib.Path(PATH)
    data = _Test_Data.model_validate_json(json_path.read_text(encoding="utf-8"))
    anki_srs = AnkiSRS(USER_TO_SETUP)
    for deck_key, deck in data.test_decks.items():
        if deck.name in SKIP_DECKS:
            print(f"Skipping deck {deck.name}")
            continue
        print(f"Setting up deck {deck.name} with {len(deck.cards)} cards")
        srs_deck = anki_srs.add_deck(deck.name)
        for card in deck.cards:
            card = anki_srs.add_card(
                deck=srs_deck,
                question=card.question,
                answer=card.answer,
                state=CardState.from_str(card.cardState),
                flag=Flag.from_str(card.flag),
            )
    anki_srs.close()
