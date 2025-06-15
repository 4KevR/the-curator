import os

import nltk
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.storage.index_store.postgres import PostgresIndexStore
from llama_index.vector_stores.postgres import PGVectorStore

from src.backend.modules.search.abstract_card_searcher import AbstractCardSearcher, C

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

Settings.llm = HuggingFaceInferenceAPI(
    model=os.getenv("LLM_URL"),
    task="text-generation",
    num_output=64,
    temperature=0.1,
)
# Settings.tokenizer = AutoTokenizer.from_pretrained(
#     "meta-llama/Llama-3.1-8B-Instruct", token=""
# )


class LlamaIndexExecutor:
    def __init__(self):
        self.vector_store_decks = self.__load_vector_store_from_table("the_curator_decks")
        self.vector_store_cards = self.__load_vector_store_from_table("the_curator_cards")
        self.index_store_decks = self.__load_index_store_from_table("the_curator_index_store_decks")
        self.index_store_cards = self.__load_index_store_from_table("the_curator_index_store_cards")
        self.storage_context_decks = StorageContext.from_defaults(
            vector_store=self.vector_store_decks, index_store=self.index_store_decks
        )
        self.storage_context_cards = StorageContext.from_defaults(
            vector_store=self.vector_store_cards, index_store=self.index_store_cards
        )
        self.deck_index = load_index_from_storage(
            storage_context=self.storage_context_decks,
        )
        self.card_index = load_index_from_storage(
            storage_context=self.storage_context_cards,
        )
        self.deck_query_engine = self.deck_index.as_query_engine(
            response_mode="compact",
        )
        self.card_query_engine = self.card_index.as_query_engine(response_mode="compact")

    def __load_vector_store_from_table(self, table_name: str) -> PGVectorStore:
        return PGVectorStore.from_params(
            database=os.getenv("POSTGRES_DB"),
            host="localhost",
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

    def __load_index_store_from_table(self, table_name: str) -> PostgresIndexStore:
        return PostgresIndexStore.from_params(
            database=os.getenv("POSTGRES_DB"),
            host="localhost",
            password=os.getenv("POSTGRES_PASSWORD"),
            port=5432,
            user=os.getenv("POSTGRES_USER"),
            table_name=table_name,
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
    def __init__(self, prompt: str):
        self._llama_result = LlamaIndexExecutor().search_cards(prompt)

    def _search(self, card: C) -> bool:
        for index_card, _ in self._llama_result:
            if card.question in index_card and card.answer in index_card:
                return True
        return False
