import json
import os
import pathlib

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.schema import Document
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.storage.index_store.postgres import PostgresIndexStore
from llama_index.vector_stores.postgres import PGVectorStore

# Steps to set up the environment:
# 1.  Populate a .env.db file based on the .env.db.example file.
# 2.  Start vector-db (docker-compose up -d vector-db)
# 3.  Run this script with CREATE_INDEX set to True to create new indices
# 3.a Copy the created index IDs from the output. Add them to L159, L160
# 4.  In the future, set CREATE_INDEX to False to use existing indices

# Ideally, in the future, we will have an index that will be maintained
# when the user makes changes to the environment.

load_dotenv(".env")
load_dotenv(".env.db")

CREATE_INDEX = True  # Set to True to create a new index, False to use an existing one
RUN_TESTS = False  # Set to True to run the question answering tests after init

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.llm = HuggingFaceInferenceAPI(
    model=os.getenv("LLM_URL"),
    task="text-generation",
)

vector_store_cards = PGVectorStore.from_params(
    database=os.getenv("POSTGRES_DB"),
    host="localhost",
    password=os.getenv("POSTGRES_PASSWORD"),
    port=5432,
    user=os.getenv("POSTGRES_USER"),
    table_name="the_curator_cards",
    embed_dim=1024,  # BAAI/bge-large-en-v1.5 embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

vector_store_decks = PGVectorStore.from_params(
    database=os.getenv("POSTGRES_DB"),
    host="localhost",
    password=os.getenv("POSTGRES_PASSWORD"),
    port=5432,
    user=os.getenv("POSTGRES_USER"),
    table_name="the_curator_decks",
    embed_dim=1024,  # BAAI/bge-large-en-v1.5 embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

index_store_decks = PostgresIndexStore.from_params(
    database=os.getenv("POSTGRES_DB"),
    host="localhost",
    password=os.getenv("POSTGRES_PASSWORD"),
    port=5432,
    user=os.getenv("POSTGRES_USER"),
    table_name="the_curator_index_store_decks",
)

index_store_cards = PostgresIndexStore.from_params(
    database=os.getenv("POSTGRES_DB"),
    host="localhost",
    password=os.getenv("POSTGRES_PASSWORD"),
    port=5432,
    user=os.getenv("POSTGRES_USER"),
    table_name="the_curator_index_store_cards",
)

storage_context_decks = StorageContext.from_defaults(
    vector_store=vector_store_decks, index_store=index_store_decks
)
storage_context_cards = StorageContext.from_defaults(
    vector_store=vector_store_cards, index_store=index_store_cards
)


def load_test_data(path: str) -> dict:
    json_path = pathlib.Path(path)
    return json.loads(json_path.read_text(encoding="utf-8"))


def get_deck_summaries(test_data: dict) -> list[dict]:
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


def get_all_cards(test_data: dict) -> list[dict]:
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


if CREATE_INDEX:
    test_data_path = "./tests/data/tests.json"
    test_data = load_test_data(test_data_path)

    # Index 1: Deck summaries
    deck_summaries = get_deck_summaries(test_data)
    deck_docs = [
        Document(text=deck["summary"], metadata={"deck_name": deck["deck_name"]})
        for deck in deck_summaries
    ]
    deck_index = VectorStoreIndex.from_documents(
        deck_docs, storage_context=storage_context_decks, show_progress=True
    )
    print(f"Created deck_index with id {deck_index.index_id}")

    # Index 2: All cards (question + answer)
    card_entries = get_all_cards(test_data)
    card_docs = [
        Document(
            text=f"Q: {card['question']}\nA: {card['answer']}",
            metadata={"deck_name": card["deck_name"]},
        )
        for card in card_entries
    ]
    card_index = VectorStoreIndex.from_documents(
        card_docs, storage_context=storage_context_cards, show_progress=True
    )
    print(f"Created card_index with id {card_index.index_id}")
else:
    deck_index_id = "a92ac466-dd85-4e7a-9480-665fb342b24a"
    card_index_id = "25a89de1-b32d-4b38-b76e-48a64ef64d73"
    deck_index = load_index_from_storage(
        storage_context=storage_context_decks, index_id=deck_index_id
    )
    card_index = load_index_from_storage(
        storage_context=storage_context_cards, index_id=card_index_id
    )
    print("Loaded existing indices:")
    print(f"deck_index_id: {deck_index.index_id}")
    print(f"card_index_id: {card_index.index_id}")


deck_query_engine = deck_index.as_query_engine()
card_query_engine = card_index.as_query_engine()

deck_retriever = QueryEngineTool.from_defaults(
    query_engine=deck_index.as_query_engine(),
    description="Use this query engine to obtain general information about decks",
)
card_retriever = QueryEngineTool.from_defaults(
    query_engine=card_index.as_query_engine(),
    description="Use this query engine for specific questions about content",
)

query_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=[deck_retriever, card_retriever]
)

# The above router is not working consistently (due to to Llama 7B) - needs more
# investigation. For now, we will use the card query engine directly for testing
# (works good for question retrieval, although generating a lot of useless
# information after the first sentence).
query_engine = card_query_engine

# --- BEGIN: Automated Question Answering Test Runner ---


def run_question_answering_tests():
    print("\n=== Running Question Answering Tests ===\n")
    test_data_path = "./tests/data/tests.json"
    with open(test_data_path, encoding="utf-8") as f:
        test_json = json.load(f)
    qa_tests = test_json.get("question_answering", {})
    if not qa_tests:
        print("No question_answering tests found in test data.")
        return

    amount_of_environments = len(qa_tests.items())
    for index, (env, tests) in enumerate(qa_tests.items()):
        print(
            f"\n--- Environment: {env} "
            f"({index + 1} out of {amount_of_environments}) ---\n"
        )
        amount_of_tests = len(tests)
        for index_test, test in enumerate(tests):
            print(f"\n--- Test {index_test + 1} out of {amount_of_tests} ---")
            test_name = test.get("name", "Unnamed Test")
            queries = test.get("queries", [])
            expected = test.get("expected_answer", "<no expected answer>")
            print(f"Test: {test_name}")
            for query_group in queries:
                for query in query_group:
                    print(f"\nQuery: {query}")
                    print(f"Expected: {expected}")
                    try:
                        result = query_engine.query(query)
                        print(f"Actual:   {result}\n")
                    except Exception as e:
                        print(f"Actual:   [ERROR] {e}\n")
            print("-" * 40)
    print("\n=== Finished Question Answering Tests ===\n")


if RUN_TESTS:
    run_question_answering_tests()
