import json
import pathlib

from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import Document

from src.backend.modules.search.llama_index import LlamaIndexPostgresLoader, SetupType

# Steps to set up the environment:
# 1. Populate a .env.db file based on the .env.db.example file.
# 2. Start vector-db (docker-compose up -d vector-db)
# 3. Run the run-setup.py script in the root directory of the project

# Ideally, in the future, we will have an index that will be maintained
# when the user makes changes to the environment.

storage_context_decks = LlamaIndexPostgresLoader.get_storage_context_for_setup(SetupType.DECKS)
storage_context_cards = LlamaIndexPostgresLoader.get_storage_context_for_setup(SetupType.CARDS)


def __load_test_data(path: str) -> dict:
    json_path = pathlib.Path(path)
    return json.loads(json_path.read_text(encoding="utf-8"))


def __get_deck_summaries(test_data: dict) -> list[dict]:
    decks = test_data["test_decks"]
    summaries = []
    for _, deck in decks.items():
        card_count = len(deck["cards"])
        card_titles = [card["question"] for card in deck["cards"]]
        summary = {
            "deck_name": deck["name"],
            "card_count": card_count,
            "card_titles": card_titles,
            "summary": (
                f"Deck '{deck['name']}' contains {card_count} cards. Titles: "
                f"{', '.join(card_titles[:5]) + ('...' if card_count > 5 else '')}"
            ),
        }
        summaries.append(summary)
    return summaries


def __get_all_cards(test_data: dict) -> list[dict]:
    decks = test_data["test_decks"]
    cards = []
    for deck in decks.values():
        for card in deck["cards"]:
            cards.append(
                {
                    "deck_name": deck["name"],
                    "question": card["question"],
                    "answer": card["answer"],
                }
            )
    return cards


def setup_llama_index():
    index_exists = True
    try:
        deck_index = load_index_from_storage(storage_context=storage_context_decks)
        card_index = load_index_from_storage(storage_context=storage_context_cards)
    except ValueError:
        index_exists = False
    if index_exists:
        print("LlamaIndex already set up. Skipping setup.")
        return

    test_data_path = "tests/data/tests.json"
    test_data = __load_test_data(test_data_path)

    # Index 1: Deck summaries
    deck_summaries = __get_deck_summaries(test_data)
    deck_docs = [Document(text=deck["summary"], metadata={"deck_name": deck["deck_name"]}) for deck in deck_summaries]
    deck_index = VectorStoreIndex.from_documents(deck_docs, storage_context=storage_context_decks, show_progress=True)
    print(f"Created deck_index with id {deck_index.index_id}")

    # Index 2: All cards (question + answer)
    card_entries = __get_all_cards(test_data)
    card_docs = [
        Document(
            text=f"Q: {card['question']}\nA: {card['answer']}",
            metadata={"deck_name": card["deck_name"]},
        )
        for card in card_entries
    ]
    card_index = VectorStoreIndex.from_documents(card_docs, storage_context=storage_context_cards, show_progress=True)
    print(f"Created card_index with id {card_index.index_id}")
    print("LlamaIndex setup completed successfully.")
