# TODO: This class is now unused. Delete!

from typing import Optional

from typeguard import typechecked

from src.backend.modules.ai_assistant.chunked_card_stream import ChunkedCardStream
from src.backend.modules.search.llama_index import LlamaIndexExecutor
from src.backend.modules.ai_assistant.llm_interactor.llm_command_list import (
    LLMCommandList,
    llm_command,
)
from src.backend.modules.ai_assistant.llm_interactor.inherit_docstrings import (
    InheritDocstrings,
)
from src.backend.modules.ai_assistant.llm_interactor.llm_interactor import LLMInteractor
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.search.abstract_card_searcher import AbstractCardSearcher
from src.backend.modules.srs.abstract_srs import CardID, DeckID
from src.backend.modules.srs.testsrs.testsrs import (
    CardState,
    Flag,
    TestCard,
    TestDeck,
    TestFlashcardManager,
    TestTemporaryCollection,
)

_commands = LLMCommandList(
    {
        "TestTemporaryCollection": "TemporaryCollection",
        "TestDeck": "Deck",
        "TestFlashcardManager": "FlashcardManager",
    },
    card_type=TestCard,
    deck_type=TestDeck,
    temp_collection_type=TestTemporaryCollection,
)


# noinspection PyTypeChecker
@typechecked
class LLMInteractorTest(LLMInteractor, metaclass=InheritDocstrings):
    """
    A LLMInteractor for TestSRS.
    """

    flashcard_manager: TestFlashcardManager
    command_list = _commands

    def __init__(
        self,
        flashcard_manager: TestFlashcardManager,
        content_search_llm: AbstractLLM,
        llama_index_executor: LlamaIndexExecutor,
    ):
        super().__init__(flashcard_manager, content_search_llm, llama_index_executor)

    @llm_command(_commands)
    def create_deck(self, name: str) -> TestDeck:
        return super().create_deck(name)

    @llm_command(_commands)
    def list_decks(self) -> list[TestDeck]:
        return super().list_decks()

    @llm_command(_commands)
    def rename_deck(self, deck_id_str: str, new_name: str) -> None:
        super().rename_deck(deck_id_str, new_name)

    @llm_command(_commands)
    def delete_deck(self, deck_id_str: str) -> None:
        super().delete_deck(deck_id_str)

    @llm_command(_commands)
    def add_card(self, deck_id_str: str, question: str, answer: str) -> None:
        super().add_card(deck_id_str, question, answer)

    @llm_command(_commands)
    def add_card_with_metadata(self, deck_id_str: str, question: str, answer: str, state: str, flag: str) -> None:
        """
        Create a new card in a deck. The deck id must be a string in the format 'deck_xxxx_xxxx'.
        The question, answer, state, and flag must all be non-empty strings.
        Valid flags are:
        ['none', 'red', 'orange', 'green', 'blue', 'pink', 'turquoise', 'purple']
        Valid card states are:
        ['new', 'learning', 'review', 'suspended', 'buried']

        """
        if not all(isinstance(x, str) and x.strip() for x in [question, answer, state, flag]):
            raise ValueError("Question, answer, state, and flag must all be non-empty strings.")
        deck_id = DeckID.from_hex_string(deck_id_str)
        deck = self.flashcard_manager.get_deck(deck_id)
        flag = Flag.from_str(flag)
        state = CardState.from_str(state)
        self.flashcard_manager.add_full_card(deck, question, answer, flag, state)

    @llm_command(_commands)
    def edit_card_question(self, card_id_str: str, new_question: str) -> None:
        super().edit_card_question(card_id_str, new_question)

    @llm_command(_commands)
    def edit_card_answer(self, card_id_str: str, new_answer: str) -> None:
        super().edit_card_answer(card_id_str, new_answer)

    @llm_command(_commands)
    def edit_card_flag(self, card_id_str: str, new_flag: str) -> None:
        """
        Edit the flag of a card. The card_id_str must be a string in the format 'card_xxxx_xxxx'.
        Valid flags are:
        ['none', 'red', 'orange', 'green', 'blue', 'pink', 'turquoise', 'purple']
        """
        if not isinstance(new_flag, str) or not new_flag.strip():
            raise ValueError("New flag must be a non-empty string.")
        card_id = CardID.from_hex_string(card_id_str)
        card = self.flashcard_manager.get_card(card_id)
        new_flag = Flag.from_str(new_flag)
        self.flashcard_manager.edit_card_flag(card, new_flag)

    @llm_command(_commands)
    def edit_card_state(self, card_id_str: str, new_state: str) -> None:
        """
        Edit the state of a card. The card_id_str must be a string in the format 'card_xxxx_xxxx'.

        Valid card states are:
        ['new', 'learning', 'review', 'suspended', 'buried']
        """
        if not isinstance(new_state, str) or not new_state.strip():
            raise ValueError("New state must be a non-empty string.")
        card_id = CardID.from_hex_string(card_id_str)
        card = self.flashcard_manager.get_card(card_id)
        new_state = CardState.from_str(new_state)
        self.flashcard_manager.edit_card_state(card, new_state)

    @llm_command(_commands)
    def delete_card(self, card_id_str: str) -> None:
        super().delete_card(card_id_str)

    @llm_command(_commands)
    def create_new_temporary_collection(self, temporary_collection_description: str) -> TestTemporaryCollection:
        return super().create_new_temporary_collection(temporary_collection_description)

    @llm_command(_commands)
    def get_temporary_collections(self) -> list[TestTemporaryCollection]:
        return super().get_temporary_collections()

    @llm_command(_commands)
    def delete_temporary_collection(self, temporary_collection_id_str: str) -> None:
        super().delete_temporary_collection(temporary_collection_id_str)

    @llm_command(_commands)
    def temporary_collection_add_card(self, temporary_collection_str: str, card_id_str: str) -> None:
        super().temporary_collection_add_card(temporary_collection_str, card_id_str)

    @llm_command(_commands)
    def temporary_collection_remove_card(self, temporary_collection_str: str, card_id_str: str) -> None:
        super().temporary_collection_remove_card(temporary_collection_str, card_id_str)

    @llm_command(_commands)
    def list_cards_temporary_collection(self, temporary_collection_id_str: str) -> ChunkedCardStream:
        return super().list_cards_temporary_collection(temporary_collection_id_str)

    @llm_command(_commands)
    def add_all_cards_from_temporary_collection_to_deck(
        self, temporary_collection_hex_string: str, deck_hex_string: str
    ) -> None:
        super().add_all_cards_from_temporary_collection_to_deck(temporary_collection_hex_string, deck_hex_string)

    @llm_command(_commands)
    def search_for_substring(
        self,
        deck_id_str: str,
        search_substring: str,
        search_in_question: bool = True,
        search_in_answer: bool = True,
        case_sensitive: bool = False,
        fuzzy: Optional[float] = None,
    ) -> TestTemporaryCollection:
        return super().search_for_substring(
            deck_id_str,
            search_substring,
            search_in_question,
            search_in_answer,
            case_sensitive,
            fuzzy,
        )

    @llm_command(_commands)
    def search_for_content(
        self,
        deck_id_str: str,
        search_prompt: str,
        search_in_question: bool = True,
        search_in_answer: bool = True,
    ) -> TestTemporaryCollection:
        return super().search_for_content(deck_id_str, search_prompt, search_in_question, search_in_answer)

    def _search(self, deck_id_str: str, searcher: AbstractCardSearcher, description: str) -> TestTemporaryCollection:
        return super()._search(deck_id_str, searcher, description)

    @llm_command(_commands)
    def respond_to_question_answering_query(self, search_prompt: str) -> str:
        return super().respond_to_question_answering_query(search_prompt)
