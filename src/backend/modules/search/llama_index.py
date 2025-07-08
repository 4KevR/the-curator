import logging
import os
import re
from enum import Enum

import nltk
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import Document
from llama_index.storage.index_store.postgres import PostgresIndexStore
from llama_index.vector_stores.postgres import PGVectorStore

from src.backend.modules.search.abstract_card_searcher import AbstractCardSearcher, C
from src.backend.modules.srs.abstract_srs import AbstractCard, AbstractDeck, CardID, DeckID
from src.backend.modules.srs.testsrs.testsrs import TestFlashcardManager

logger = logging.getLogger(__name__)


def _get_host() -> str:
    if int(os.getenv("IN_DOCKER", "0")):
        return "vector-db"
    return "localhost"


def abstract_deck_to_document(deck: AbstractDeck) -> Document:
    return Document(
        doc_id=deck.id.hex_id(),
        text=deck.name,
        metadata={},
    )


# TODO missing deck name for card in LlamaIndex Document representation
def abstract_card_to_document(card: AbstractCard) -> Document:
    return Document(
        doc_id=card.id.hex_id(),
        text=f"Q: {card.question}\nA: {card.answer}",
        metadata={"flag": card.flag.value, "deck_id": card.deck.id.hex_id()},
    )


class LlamaIndexTestManager:
    def __init__(self, environments: dict[str, TestFlashcardManager]):
        self.environments = environments
        self.llama_index_executors = {
            test_environment_name: LlamaIndexExecutor(store_name=f"TEST_{test_environment_name}")
            for test_environment_name in environments.keys()
        }
        for environment, executor in self.llama_index_executors.items():
            if executor.was_already_set_up:
                logger.info(f"Environment '{environment}' already set up with existing indexes.")
                continue
            logger.info(f"Setting up environment '{environment}' with new indexes.")
            all_decks = self.environments[environment].get_all_decks()
            for deck in all_decks:
                executor.add_deck(deck)
                cards_in_deck = self.environments[environment].get_cards_in_deck(deck)
                for card in cards_in_deck:
                    executor.add_card(card)


class LlamaIndexExecutor:
    """
    Class for interacting with a LlamaIndex store in a PostgreSQL database.
    It allows adding, removing, and modifying cards and decks, as well as querying them.
    It will store the indexes based on the user name, if provided.
    If no store name is provided, it will use the default table names.
    """

    def __init__(self, store_name: str | None = None):
        def sanitize_name(name: str) -> str:
            return re.sub(r"[^a-zA-Z0-9]", "_", name)

        prefixed_user_name = f"_{sanitize_name(store_name)}" if store_name else ""
        deck_table_name = f"decks{prefixed_user_name}"
        self.vector_store_decks = LlamaIndexPostgresLoader.get_vector_store_from_table_name(deck_table_name)
        card_table_name = f"cards{prefixed_user_name}"
        self.vector_store_cards = LlamaIndexPostgresLoader.get_vector_store_from_table_name(card_table_name)
        deck_index_store_name = f"idx_store_decks{prefixed_user_name}"
        self.index_store_decks = LlamaIndexPostgresLoader.get_index_store_from_table_name(deck_index_store_name)
        card_index_store_name = f"idx_store_cards{prefixed_user_name}"
        self.index_store_cards = LlamaIndexPostgresLoader.get_index_store_from_table_name(card_index_store_name)
        self.storage_context_decks = StorageContext.from_defaults(
            vector_store=self.vector_store_decks, index_store=self.index_store_decks
        )
        self.storage_context_cards = StorageContext.from_defaults(
            vector_store=self.vector_store_cards, index_store=self.index_store_cards
        )
        try:
            self.deck_index = load_index_from_storage(
                storage_context=self.storage_context_decks,
            )
            self.card_index = load_index_from_storage(
                storage_context=self.storage_context_cards,
            )
            self.was_already_set_up = True
        except ValueError:
            logger.info("No indexes found in database, creating new ones for decks and cards.")
            self.deck_index = VectorStoreIndex.from_documents(
                documents=[],
                storage_context=self.storage_context_decks,
            )
            self.card_index = VectorStoreIndex.from_documents(
                documents=[],
                storage_context=self.storage_context_cards,
            )
            self.was_already_set_up = False
        self.deck_query_engine = self.deck_index.as_query_engine(
            response_mode="compact",
        )
        self.card_query_engine = self.card_index.as_query_engine(response_mode="compact")

    def add_card(self, card: AbstractCard):
        if not isinstance(card, AbstractCard):
            raise TypeError("Card must be an instance of AbstractCard")
        if not card.question or not card.answer:
            raise ValueError("Card must have a question and an answer.")
        card_document = abstract_card_to_document(card)
        self.card_index.insert(card_document)

    def remove_card(self, card_id: CardID):
        self.card_index.delete_ref_doc(card_id.hex_id(), delete_from_docstore=True)

    def modify_card(self, card: AbstractCard):
        card_document = abstract_card_to_document(card)
        self.card_index.update_ref_doc(
            card_document,
        )

    def add_deck(self, deck: AbstractDeck):
        if not isinstance(deck, AbstractDeck):
            raise TypeError("Deck must be an instance of AbstractDeck")
        if not deck.name:
            raise ValueError("Deck must have a name.")
        deck_document = abstract_deck_to_document(deck)
        self.deck_index.insert(deck_document)

    def remove_deck(self, deck_id: DeckID):
        self.deck_index.delete_ref_doc(deck_id.hex_id(), delete_from_docstore=True)

    def modify_deck(self, deck: AbstractDeck):
        deck_document = abstract_deck_to_document(deck)
        self.deck_index.update_ref_doc(
            deck_document,
        )

    def load_index(self, index_id: str, storage_context: StorageContext):
        if not index_id:
            raise RuntimeError("No index_id found for the given table.")
        return load_index_from_storage(storage_context=storage_context, index_id=index_id)

    def query_decks(self, query: str):
        return self.deck_query_engine.query(query)

    def query_cards(self, query: str):
        query_response = self.card_query_engine.query(query)
        stripped_response = nltk.sent_tokenize(query_response.response)[:2]
        return " ".join(stripped_response).strip()

    def search_cards(self, query: str) -> list[tuple[str, float]]:
        """
        Returns cards in format

        Q: <question>
        A: <answer>

        with scores.
        """
        query_response = self.card_query_engine.query(query)
        fitting_cards: list[tuple[str, float]] = [(node.text, node.score) for node in query_response.source_nodes]
        return fitting_cards


class LlamaIndexSearcher(AbstractCardSearcher):
    def __init__(self, executor: LlamaIndexExecutor, prompt: str):
        self._llama_result = executor.search_cards(prompt)

    def _search(self, card: C) -> bool:
        for index_card, _ in self._llama_result:
            if card.question in index_card and card.answer in index_card:
                return True
        return False


class SetupType(Enum):
    DECKS = "decks"
    CARDS = "cards"


class LlamaIndexPostgresLoader:
    @staticmethod
    def get_storage_context_for_setup(setup: SetupType) -> StorageContext:
        if setup == SetupType.DECKS:
            vector_store = LlamaIndexPostgresLoader.get_vector_store_from_table_name("decks")
            index_store = LlamaIndexPostgresLoader.get_index_store_from_table_name("idx_store_decks")
        elif setup == SetupType.CARDS:
            vector_store = LlamaIndexPostgresLoader.get_vector_store_from_table_name("cards")
            index_store = LlamaIndexPostgresLoader.get_index_store_from_table_name("idx_store_cards")
        else:
            raise ValueError(f"Unknown setup type: {setup}")
        return StorageContext.from_defaults(vector_store=vector_store, index_store=index_store)

    @staticmethod
    def get_vector_store_from_table_name(table_name: str) -> PGVectorStore:
        return PGVectorStore.from_params(
            database=os.getenv("POSTGRES_DB"),
            host=_get_host(),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=5432,
            user=os.getenv("POSTGRES_USER"),
            table_name=table_name,
            embed_dim=1024,  # BAAI/bge-large-en-v1.5 embedding dimension
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

    @staticmethod
    def get_index_store_from_table_name(table_name: str) -> PostgresIndexStore:
        return PostgresIndexStore.from_params(
            database=os.getenv("POSTGRES_DB"),
            host=_get_host(),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=5432,
            user=os.getenv("POSTGRES_USER"),
            table_name=table_name,
        )
