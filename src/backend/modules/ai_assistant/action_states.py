import json
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Any, Optional

import pandas as pd
from rapidfuzz.distance import Levenshtein

from src.backend.modules.helpers.string_util import find_substring_in_llm_response_or_null, remove_block
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.llm.llm_communicator import LLMCommunicator
from src.backend.modules.search.abstract_card_searcher import AbstractCardSearcher
from src.backend.modules.search.llama_index import LlamaIndexExecutor, LlamaIndexSearcher
from src.backend.modules.search.search_by_substring import SearchBySubstring
from src.backend.modules.search.search_by_substring_fuzzy import SearchBySubstringFuzzy
from src.backend.modules.srs.abstract_srs import (
    AbstractCard,
    AbstractDeck,
    AbstractSRS,
    CardState,
    Flag,
    MissingDeckException,
)


class AbstractActionState(ABC):

    @abstractmethod
    def act(self) -> Optional["AbstractActionState"]:
        """
        Returns
        * A (different) ActionState if the llm is now in another state.
        * A string to send to the llm and to send the response back to this state.
        * None if the task is completed.
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class ExceedingMaxAttemptsError(Exception):
    """
    Custom exception for exceeding the maximum number of attempts.
    """

    def __init__(self, state_name):
        super().__init__(f"Exceeding maximum number of attempts in state {state_name}.")


class StateAction(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. You assist a user in interacting with the system (creating/modifying/deleting cards/decks etc.) and in answering questions about the content of the flashcards.

The user gave the following prompt:

{user_input}

If you think the user wants you to **interact** with the flashcard system (e.g. creating, modifying, or deleting cards or decks), please answer "task".
If the user wants you to answer a question about the content of the flashcards, please answer "question".
Do not answer anything else.
"""
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS, llama_index_executor: LlamaIndexExecutor):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs
        self.llama_index_executor = llama_index_executor

    def act(self) -> AbstractActionState | None:
        for attempt in range(self.MAX_ATTEMPTS):
            if attempt == 0:
                message = self._prompt_template.format(user_input=self.user_prompt)
            else:
                message = "Your answer must be either 'question' or 'task'."

            response = self.llm_communicator.send_message(message)

            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            resp = find_substring_in_llm_response_or_null(response, "question", "task", True)

            if resp is True:
                return StateQuestion(self.user_prompt, self.llm, self.llama_index_executor)
            elif resp is False:
                return StateTask(self.user_prompt, self.llm, self.srs)

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateQuestion(AbstractActionState):
    _prompt_template = (
        "You are an assistant of a flashcard management system. You assist the user in answering questions about the"
        " content of the flashcards in the system. The user asked the following question:\n"
        "{user_input}\n"
        "The following cards in the system fit the question:\n"
        "{cards}\n"
        "**Using only the information in the cards above, answer the question.**\n"
        "If the question cannot be answered using the cards above, respond accordingly."
        "Answer with a single, short sentence, without any additional information."
    )

    def __init__(self, user_prompt: str, llm: AbstractLLM, llama_index_executor: LlamaIndexExecutor):
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.llama_index_executor = llama_index_executor

    def act(self) -> AbstractActionState | None:
        fitting_nodes = self.llama_index_executor.search_cards(self.user_prompt)
        fitting_nodes = sorted(fitting_nodes, key=lambda x: x[1], reverse=True)[:5]
        fitting_nodes = "\n".join(fn[0] for fn in fitting_nodes)

        message = self._prompt_template.format(user_input=self.user_prompt, cards=fitting_nodes)
        response = self.llm_communicator.send_message(message)
        return StateAnswer(response)


class StateAnswer(AbstractActionState):
    def __init__(self, answer: str):
        self.answer = answer

    def act(self) -> None:
        return None  # final state


class StateTask(AbstractActionState):
    _prompt_template = """You are an assistant of a flashcard management system. You assist a user in executing tasks.
The flashcard management system consists of decks consisting of cards.

The user gave the following input:

{user_input}

There are two categories of tasks:

Category 'local' contains the following tasks:

* Editing specific cards.
* Deleting specific cards.


Category 'global' contains the following tasks:

* Adding new, empty decks.
* Renaming decks.
* Deleting decks.
* Creating new cards and adding them to a deck.

Is the user prompt a 'local' or 'global' task? Please **only** answer with 'local' or 'global' and **nothing else**.
""".strip()  # TODO: I could add the option here to find impossible tasks.

    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs

    def preprocess_user_task(self, raw_task) -> str:
        """
        Preprocess the user task. Currently quite minimal, but could be expanded.
        """
        # TODO: Maybe even own state? Update: Definitely add own state that only fixes the task.
        # currently only quotation marks. They cant be transcribed anyways.

        return raw_task  # .replace('"', "").replace("'", "")

    def act(self) -> AbstractActionState | None:
        for attempt in range(self.MAX_ATTEMPTS):
            cleaned_user_prompt = self.preprocess_user_task(self.user_prompt)

            if attempt == 0:
                message = self._prompt_template.format(user_input=cleaned_user_prompt)
            else:
                message = "Your answer must be either 'local' or 'global'."

            response = self.llm_communicator.send_message(message)

            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            resp = find_substring_in_llm_response_or_null(response, "local", "global", True)

            if resp is True:
                return StateTaskSearchDecks(cleaned_user_prompt, self.llm, self.srs)
            elif resp is False:
                return StateTaskNoSearch(cleaned_user_prompt, self.llm, self.srs)

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateTaskSearchDecks(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system.
You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:

{user_input}

You already decided that you have to search for cards. Now you have to decide in which decks you want to search.
The following decks are available:

{decks}

If you want to search in all decks, answer "all". If you want to search in a specific deck, answer the name of the deck.
If you want to search in multiple, specific decks, answer a comma-separated list of deck names.
If you are unsure, rather include than exclude a deck. Make sure to exactly match the deck names.
**Do not answer anything else**!
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.srs = srs
        self.user_prompt = user_prompt
        self.possible_decks = self.srs.get_all_decks()
        self.possible_deck_names = {deck.name for deck in self.possible_decks}

    def act(self) -> AbstractActionState | None:

        message = self._prompt_template.format(
            user_input=self.user_prompt, decks=[str(it) for it in self.srs.get_all_decks()]
        )

        for attempt in range(self.MAX_ATTEMPTS):
            response = self.llm_communicator.send_message(message)
            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            response = response.strip()

            if response == "all":
                return StateTaskSearch(self.user_prompt, self.llm, self.possible_decks, self.srs)
            else:
                deck_strings = {s.strip() for s in response.split(",")}
                unknown_deck_strings = deck_strings - self.possible_deck_names

                if len(unknown_deck_strings) == 0:
                    decks = [it for it in self.possible_decks if it.name in deck_strings]
                    return StateTaskSearch(self.user_prompt, self.llm, decks, self.srs)

                message = (
                    f"The following deck names are unknown: {', '.join(unknown_deck_strings)}.\n"
                    'If you want to search in all decks, answer "all" (and nothing else!)."'
                    "If you want to search in a specific deck, answer the name of the deck.\n"
                    "If you want to search in multiple, specific decks, answer a comma-separated list of deck names.\n"
                    "Please make sure to exactly match the deck names."
                )

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateTaskSearch(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system.
You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:

{user_input}

You already decided that you have to search for cards.
Please decide now how you want to search for cards. Your options are:

* Search for keyword (exact): You give me one or multiple keywords and I will search for all cards that contain at least one of these keywords. You will be able to specify whether you want to search in the question or the answer or both. You can also decide whether the search should be case sensitive or not.

* Search for keyword (fuzzy search): You give me one or multiple keywords and I will search for all cards that contain at least one of these keywords, or contain a 'similar' substring. You will be able to specify whether you want to search in the question or the answer or both. You can also decide whether the search should be case sensitive or not.

* Search cards with fitting content: You give me a search prompt and I will search for cards that fit the search prompt. The search is *not* limited to exact wording, but searches for cards with fitting content.


If you have exact keywords to look for, you should use exact search. If you have one or more words/phrases to search for, but you cannot be sure that all fitting cards contain the keywords/phrases exactly (e.g. plural form, quotation marks, etc.), use fuzzy search. If you want to search for a topic, use content-based search.

Please answer "exact", "fuzzy" or "content", and **nothing else**. All other details will be determined later.
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, decks_to_search_in: list[AbstractDeck], srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs

    def act(self) -> AbstractActionState | None:
        for attempt in range(self.MAX_ATTEMPTS):
            if attempt == 0:
                message = self._prompt_template.format(user_input=self.user_prompt)
            else:
                message = 'Your answer must be either "exact", "fuzzy" or "content", **and nothing else**.'

            response = self.llm_communicator.send_message(message)

            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            response = response.lower().strip()

            if response == "exact":
                return StateKeywordSearch(self.user_prompt, self.llm, self.decks_to_search_in, self.srs)
            if response == "fuzzy":
                return StateFuzzySearch(self.user_prompt, self.llm, self.decks_to_search_in, self.srs)
            if response == "content":
                return StateContentSearch(self.user_prompt, self.llm, self.decks_to_search_in, self.srs)

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateKeywordSearch(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system.
You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:

{user_input}

You already decided that you have to search for cards, and that you want to use keyword search. You may search for one or more keywords.
Please fill in the following template. Make sure to produce valid json.
{{
    "search_substring": "<search_substring_here>",
    "search_in_question": <bool here>,
    "search_in_answer": <bool here>,
    "case_sensitive": <bool here>
}}

If you are unsure, use these defaults:
  search_in_question: true
  search_in_answer: true
  case_sensitive: false

Please answer only with an json array [...] of filled-in, valid json object as described above.
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, decks_to_search_in: list[AbstractDeck], srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs

    def act(self) -> AbstractActionState | None:
        message = self._prompt_template.format(user_input=self.user_prompt)
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self.llm_communicator.send_message(message)
                parsed_list = json.loads(response.strip())
                if not isinstance(parsed_list, list):
                    raise ValueError("Response must be a list.")

                searchers = []
                for parsed in parsed_list:
                    if not isinstance(parsed, dict):
                        raise ValueError("Response must be a dictionary")
                    if set(parsed.keys()) != {
                        "search_substring",
                        "search_in_question",
                        "search_in_answer",
                        "case_sensitive",
                    }:
                        raise ValueError("Response must contain exactly the required keys")
                    if not isinstance(parsed["search_substring"], str):
                        raise ValueError("search_substring must be a string")
                    if not isinstance(parsed["search_in_question"], bool):
                        raise ValueError("search_in_question must be a boolean")
                    if not isinstance(parsed["search_in_answer"], bool):
                        raise ValueError("search_in_answer must be a boolean")
                    if not isinstance(parsed["case_sensitive"], bool):
                        raise ValueError("case_sensitive must be a boolean")

                    searcher = SearchBySubstring(
                        search_substring=parsed["search_substring"],
                        search_in_question=parsed["search_in_question"],
                        search_in_answer=parsed["search_in_answer"],
                        case_sensitive=parsed["case_sensitive"],
                    )
                    searchers.append(searcher)

                return StateVerifySearch(self.user_prompt, self.llm, self.decks_to_search_in, self.srs, searchers)

            except JSONDecodeError as jde:
                message = f"Your answer must be a valid json string. Exception: {jde}. Please try again."
            except Exception as e:
                message = f"An exception occurred: {e}. Please try again."

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateFuzzySearch(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system.
You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:

{user_input}

You already decided that you have to search for cards, and that you want to use fuzzy keyword search. You may search for one or more keywords.
Please fill in the following template. Make sure to produce valid json.
{{
    "search_substring": "<search_substring_here>",
    "search_in_question": <bool here>,
    "search_in_answer": <bool here>,
    "case_sensitive": <bool here>,
    "fuzzy": <float here>
}}

If you are unsure, use these defaults:
  search_in_question: true
  search_in_answer: true
  case_sensitive: false
  fuzzy: 0.8

Please answer only with an json array [...] of filled-in, valid json object as described above.
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, decks_to_search_in: list[AbstractDeck], srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs

    def act(self) -> AbstractActionState | None:
        message = self._prompt_template.format(user_input=self.user_prompt)
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self.llm_communicator.send_message(message)
                parsed_list = json.loads(response.strip())
                if not isinstance(parsed_list, list):
                    raise ValueError("Response must be a list.")

                searchers = []
                for parsed in parsed_list:
                    if not isinstance(parsed, dict):
                        raise ValueError("Response must be a dictionary")
                    if set(parsed.keys()) != {
                        "search_substring",
                        "search_in_question",
                        "search_in_answer",
                        "case_sensitive",
                        "fuzzy",
                    }:
                        raise ValueError("Response must contain exactly the required keys")
                    if not isinstance(parsed["search_substring"], str):
                        raise ValueError("search_substring must be a string")
                    if not isinstance(parsed["search_in_question"], bool):
                        raise ValueError("search_in_question must be a boolean")
                    if not isinstance(parsed["search_in_answer"], bool):
                        raise ValueError("search_in_answer must be a boolean")
                    if not isinstance(parsed["case_sensitive"], bool):
                        raise ValueError("case_sensitive must be a boolean")
                    if not isinstance(parsed["fuzzy"], float):
                        raise ValueError("fuzzy must be a float")

                    searcher = SearchBySubstringFuzzy(
                        search_substring=parsed["search_substring"],
                        search_in_question=parsed["search_in_question"],
                        search_in_answer=parsed["search_in_answer"],
                        case_sensitive=parsed["case_sensitive"],
                        fuzzy=parsed["fuzzy"],
                    )
                    searchers.append(searcher)

                return StateVerifySearch(self.user_prompt, self.llm, self.decks_to_search_in, self.srs, searchers)

            except JSONDecodeError as jde:
                message = f"Your answer must be a valid json string. Exception: {jde}. Please try again."
            except Exception as e:
                message = f"An exception occurred: {e}. Please try again."

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateContentSearch(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system.
You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:

{user_input}

You already decided that you have to search for cards, and that you want to use content based search.
Please fill in the following template. Make sure to produce valid json.
{{
    "search_prompt": "<string here>"
}}

Please answer only with the filled-in, valid json.
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, decks_to_search_in: list[AbstractDeck], srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs

    def act(self) -> AbstractActionState | None:
        message = self._prompt_template.format(user_input=self.user_prompt)
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self.llm_communicator.send_message(message)
                parsed = json.loads(response.strip())
                if not isinstance(parsed, dict):
                    raise ValueError("Response must be a dictionary")
                if set(parsed.keys()) != {"search_prompt"}:
                    raise ValueError("Response must contain exactly the required keys")
                if not isinstance(parsed["search_prompt"], str):
                    raise ValueError("search_prompt must be a string")

                searcher = LlamaIndexSearcher(prompt=parsed["search_prompt"])
                return StateVerifySearch(self.user_prompt, self.llm, self.decks_to_search_in, self.srs, [searcher])

            except JSONDecodeError as jde:
                message = f"Your answer must be a valid json string. Exception: {jde}. Please try again."
            except Exception as e:
                message = f"An exception occurred: {e}. Please try again."

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateVerifySearch(AbstractActionState):
    _prompt_template = (
        "You are an assistant of a flashcard management system. You assist a user in executing tasks "
        "(creating/modifying/deleting cards/decks etc.).\n\n"
        "The user gave the following input:\n\n"
        "{user_input}\n\n"
        "You decided to search for cards. Your search returned {amount_cards} cards."
        " Here is a sample of the cards you found:\n\n"
        "{cards_sample}\n\n"
        "You now have to decide if the search went okay.\n"
        " * If the search went fine, please answer 'yes'.\n"
        " * If the search results seem to be at least okay, please answer 'yes'.\n"
        " * Only if something went really wrong, you should answer 'no'.\n"
        "Please only answer 'yes' or 'no', and **nothing else**."
    )
    MAX_ATTEMPTS = 3
    SAMPLE_SIZE = 5

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        decks_to_search_in: list[AbstractDeck],
        srs: AbstractSRS,
        searchers: list[AbstractCardSearcher],
    ):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs
        self.searchers = searchers
        self.all_cards: list[AbstractCard] = [
            card for deck in self.decks_to_search_in for card in self.srs.get_cards_in_deck(deck)
        ]
        self.found_cards = AbstractCardSearcher.union_search_all(searchers, self.all_cards)

    def act(self) -> AbstractActionState | None:
        for attempt in range(self.MAX_ATTEMPTS):
            if attempt == 0:
                if len(self.found_cards) <= self.SAMPLE_SIZE:
                    sample = self.found_cards
                else:
                    sample = pd.Series(self.found_cards).sample(self.SAMPLE_SIZE).to_list()

                message = self._prompt_template.format(
                    user_input=self.user_prompt,
                    amount_cards=len(self.found_cards),
                    cards_sample="\n\n".join(str(it) for it in sample),
                )
            else:
                message = "Your answer must be either 'yes' or 'no'."

            response = self.llm_communicator.send_message(message)

            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            resp = find_substring_in_llm_response_or_null(response, "yes", "no", True)

            if resp is True:
                return StateTaskWorkOnFoundCards(
                    self.user_prompt, self.llm, self.decks_to_search_in, self.srs, self.found_cards
                )
            elif resp is False:
                raise NotImplementedError()  # TODO! Add a *one-time-only* loopback here.

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateTaskWorkOnFoundCards(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:

{user_input}

You decided to search for cards, and found {amount_cards} fitting cards. Nothing has been done with these cards yet. Now you have to decide what to do with the cards you found. You have the following options:

* You can delete all cards you found from the system. ('delete_all')
* You can copy all cards you found to a new deck, but keep them unchanged. You will be asked to specify the name of the new deck. ('copy_to_deck')
* You can be presented every single card and you can decide individually what to do with each card (options are editing the card, deleting the card, or doing nothing with the card). ('stream_cards')

Please answer only with the operation you want to perform, and nothing else. Again, the operations are:
delete_all, copy_to_deck, stream_cards
""".strip()
    MAX_ATTEMPTS = 3
    SAMPLE_SIZE = 3

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        decks_to_search_in: list[AbstractDeck],
        srs: AbstractSRS,
        found_cards: list[AbstractCard],
    ):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs
        self.found_cards = found_cards

    def act(self) -> AbstractActionState | None:
        for attempt in range(self.MAX_ATTEMPTS):
            if attempt == 0:
                message = self._prompt_template.format(
                    user_input=self.user_prompt,
                    amount_cards=len(self.found_cards),
                )
            else:
                message = (
                    'Your answer must be either "delete_all", "copy_to_deck" or "stream_cards", **and nothing else**.'
                )

            response = self.llm_communicator.send_message(message)

            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            response = response.lower().strip()

            if response == "delete_all":
                for card in self.found_cards:
                    self.srs.delete_card(card)

                return StateFinishedTask(f"{len(self.found_cards)} cards deleted.")

            if response == "copy_to_deck":
                # TODO: This got too big. Make an own state.
                deck_list = "\n".join([f" * {it.name}" for it in self.srs.get_all_decks()])
                deck_name = self.llm_communicator.send_message(
                    "The cards can either be copied to an existing or new deck, depending on the user's request. "
                    "If you are unsure, please create a new deck. If the user says to add the cards to 'the deck' and "
                    "only one deck exists, please use that one.\n"
                    "\n"
                    "If you have to create a new deck, and the user provided a name, use that name. "
                    "Else create a fitting name. Only use letters, numbers, spaces and underscores for the name."
                    "\n\n"
                    f"Remember, the user prompt was:\n {self.user_prompt}\n"
                    "Currently, the following decks exist:\n"
                    f"{deck_list}"
                    "\n\n"
                    "Please answer only with the name of the deck, and nothing else. "
                )  # noqa: E999

                deck_created = False
                deck = self.srs.get_deck_by_name_or_none(deck_name)
                if deck is None:
                    deck = self.srs.add_deck(deck_name)
                    deck_created = True

                for card in self.found_cards:
                    self.srs.copy_card_to(card, deck)

                if deck_created:
                    return StateFinishedTask(f"{len(self.found_cards)} cards copied to newly created deck {deck_name}.")
                else:
                    return StateFinishedTask(f"{len(self.found_cards)} cards copied to existing deck {deck_name}.")

            if response == "stream_cards":
                return StateStreamFoundCards(
                    self.user_prompt, self.llm, self.decks_to_search_in, self.srs, self.found_cards
                )

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateStreamFoundCards(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. It is your job to execute the task given by the user on the given card.

## Task
Your task is:

{user_task}

## Card
The given card is:

*Question*: {question}
*Answer*: {answer}
*State*: {state}
*Flag*: {flag}

## Action
You can choose one of the following actions:

* Do nothing with this card: Respond "do_nothing".
* Delete this card: Respond "delete_card".
* Edit this card. Respond with "edit_card" and the following, filled-out template:
  {{
    "question": "<new question here>",
    "answer": "<new answer here>",
    "flag": "<new flag here>",
    "state": "<new card state here>"
  }}
  Do not forget to include the quotation marks around the strings to create valid json!
  These flag options exist: ["none", "red", "orange", "green", "blue", "pink", "turquoise", "purple"]
  These card state options exist: ["new", "learning", "review", "suspended", "buried"]

Please answer only with the operation you want to perform in the given format, and answer nothing else!
""".strip()
    # Lesson learned: You cannot tell llama-8b to just respond a json object to edit the card; it always says
    # "edit_card" before, even if not instructed to do so.

    MAX_ATTEMPTS_PER_CARD = 3

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        decks_to_search_in: list[AbstractDeck],
        srs: AbstractSRS,
        found_cards: list[AbstractCard],
    ):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs
        self.found_cards = found_cards

    def _execute_command(self, response: str, card: AbstractCard):
        response = response.strip()

        if response == "do_nothing":
            return
        if response == "delete_card":
            self.srs.delete_card(card)
            return

        # only editing or wrong input left.
        # Yes, this removes any combination of these letters. Doesnt matter.
        response = response.lstrip(" \nedit_card")
        parsed = json.loads(response.strip())  # may throw error

        # verify format
        if not isinstance(parsed, dict):
            raise ValueError("Response must be a dict in the given format!")

        if not all(isinstance(it, str) for it in list(parsed.keys()) + list(parsed.values())):
            raise ValueError("Response must be a dict[str, str].")

        valid_keys = {"question", "answer", "flag", "state"}
        if not len(valid_keys - parsed.keys()) == 0:
            additional_keys = ", ".join(sorted(set(parsed.keys()) - valid_keys))
            raise ValueError(
                f"Response may only contain the following keys: {', '.join(sorted(valid_keys))}."
                f" Got unexpected keys: {additional_keys}."
            )

        # edit card
        if "question" in parsed and parsed["question"] != card.question:
            self.srs.edit_card_question(card, parsed["question"])
        if "answer" in parsed and parsed["answer"] != card.answer:
            self.srs.edit_card_answer(card, parsed["answer"])
        if "flag" in parsed and parsed["flag"] != card.flag:
            flag = Flag.from_str(parsed["flag"])
            self.srs.edit_card_flag(card, flag)
        if "state" in parsed and parsed["state"] != card.state:
            state = CardState.from_str(parsed["state"])
            self.srs.edit_card_state(card, state)

        return

    def act(self) -> AbstractActionState | None:
        for card in self.found_cards:
            message = self._prompt_template.format(
                user_task=self.user_prompt,
                question=card.question,
                answer=card.answer,
                flag=card.flag,
                state=card.state,
            )
            response = self.llm_communicator.send_message(message)
            self._execute_command(response, card)
            self.llm_communicator.start_visibility_block()

            for attempt in range(self.MAX_ATTEMPTS_PER_CARD):
                response = self.llm_communicator.send_message(message)
                try:
                    self._execute_command(response, card)
                    break  # if the command executed successfully
                except JSONDecodeError as jde:
                    message = f"Your answer must be a valid json string. Exception: {jde}. Please try again."
                except Exception as e:
                    message = f"An exception occurred: {e}. Please try again."
            else:  # only run if no break!
                raise ExceedingMaxAttemptsError(self.__class__.__name__)

        return StateFinishedTask(f"{len(self.found_cards)} cards handled in a stream.")  # TODO (command counts?)


class StateTaskNoSearch(AbstractActionState):

    _prompt_template = """
You are an assistant of a flashcard management system. You execute a task for a user.

The user gave the following task:

{user_input}

The following decks currently exist:

{current_decks}

You now have to call zero, one or more of the following functions:

* create_deck: {{"task": "create_deck", "name": "<deck name here>"}}
Calling this function will create a new deck with the given name.
If the deck already exists, you will receive an error and can try again.

* rename_deck: {{"task": "rename_deck", "old_name": "<old deck name here>", "new_name": "<new deck name here>"}}
Calling this function will rename the deck to the given name.
If no deck exists with the old name, you will receive an error and can try again.

* delete_deck: {{"task": "delete_deck", "name": "<deck name here>"}}
Calling this function will delete the deck with the given name.
If no deck exists with the given name, you will receive an error and can try again.

* add_card: {{"task": "add_card", "deck_name": "<deck name here>", "question": "<question here>",
"answer": "<answer here>", "state": "<card state here>", "flag": "<flag here>"}}
Calling this function will add a new card to the deck with the given name.
If no deck exists with the given name, you will receive an error and can try again.
The user input has speech-to-text errors, so please fix capitalization in question and answer!
Valid flags are: ['none', 'red', 'orange', 'green', 'blue', 'pink', 'turquoise', 'purple']
Valid card states are: ['new', 'learning', 'review', 'suspended', 'buried']

If you want to execute no function, return an empty list [].
If you want to execute one or more functions, return them inside a json array.

Please answer only with the filled-in, valid json.
""".strip()
    MAX_ATTEMPTS = 10

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.srs = srs
        self.user_prompt = user_prompt

    @staticmethod
    def _parse_commands(response: str) -> list[dict[str, str]]:
        """
        Parses the commands and does all the "ex ante" checking.
        List of dict of strings, right keys, right values.
        Does not test anything that has to do with the srs.
        """
        parsed = json.loads(response.strip())

        if not isinstance(parsed, list):
            raise ValueError("Response must be a list in JSON format.")

        for cmd_dict in parsed:
            if not isinstance(cmd_dict, dict):
                raise ValueError(f"Command {cmd_dict} must be a dictionary")

            # check that all keys values are strings
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in cmd_dict.items()):
                raise ValueError(f"Command {cmd_dict}: All keys and values must be strings")

            # check that task is present and one of the expected tasks
            if "task" not in cmd_dict:
                raise ValueError(f"Command {cmd_dict}: Response must contain a task key")

            valid_tasks = ["create_deck", "rename_deck", "delete_deck", "add_card"]
            if cmd_dict["task"] not in valid_tasks:
                raise ValueError(f"Command {cmd_dict}: Response must contain a valid task")

            # check that the task is one of the expected tasks
            task_str = cmd_dict.get("task", None)
            if task_str == "create_deck":
                deck_name = cmd_dict.get("name", None)
                if not isinstance(deck_name, str):
                    raise ValueError(f"Command {cmd_dict}: Deck name must be a string")

            if task_str == "rename_deck":
                old_name = cmd_dict.get("old_name", None)
                new_name = cmd_dict.get("new_name", None)
                if not isinstance(old_name, str) or not isinstance(new_name, str):
                    raise ValueError(f"Command {cmd_dict}: Names must be strings")

            if task_str == "delete_deck":
                name = cmd_dict.get("name", None)
                if not isinstance(name, str):
                    raise ValueError(f"Command {cmd_dict}: Name must be a string")

            if task_str == "add_card":
                deck_name = cmd_dict.get("deck_name", None)
                question = cmd_dict.get("question", None)
                answer = cmd_dict.get("answer", None)
                state = cmd_dict.get("state", None)
                flag = cmd_dict.get("flag", None)
                if not isinstance(deck_name, str):
                    raise ValueError(f"Command {cmd_dict}: Name must be a string")
                if not isinstance(question, str):
                    raise ValueError(f"Command {cmd_dict}: Question must be a string")
                if not isinstance(answer, str):
                    raise ValueError(f"Command {cmd_dict}: Answer must be a string")
                if not isinstance(state, str):
                    raise ValueError(f"Command {cmd_dict}: State must be a string")
                if not isinstance(flag, str):
                    raise ValueError(f"Command {cmd_dict}: Flag must be a string")
                CardState.from_str(state)
                Flag.from_str(flag)  # run to test if it throws an error

        return parsed

    def _execute_command(self, cmd_dict: dict[Any, Any]) -> None:
        # execute tasks
        if cmd_dict["task"] == "create_deck":
            deck_name = cmd_dict["name"]
            deck = self.srs.get_deck_by_name_or_none(deck_name)
            if deck is not None:
                raise ValueError("Deck already exists")
            self.srs.add_deck(deck_name)
            return

        if cmd_dict["task"] == "rename_deck":
            old_name = cmd_dict["old_name"]
            new_name = cmd_dict["new_name"]
            deck = self.srs.get_deck_by_name_or_none(old_name)
            if deck is None:
                raise MissingDeckException(old_name)
            if self.srs.get_deck_by_name_or_none(new_name) is not None:
                raise ValueError(f"New name {new_name} already exists")
            self.srs.rename_deck(deck, new_name)
            return

        if cmd_dict["task"] == "delete_deck":
            name = cmd_dict["name"]
            deck = self.srs.get_deck_by_name_or_none(name)
            if deck is None:
                raise MissingDeckException(name)
            self.srs.delete_deck(deck)
            return

        if cmd_dict["task"] == "add_card":
            deck_name = cmd_dict["deck_name"]
            question = cmd_dict["question"]
            answer = cmd_dict["answer"]
            state = cmd_dict["state"]
            flag = cmd_dict["flag"]
            state = CardState.from_str(state)
            flag = Flag.from_str(flag)
            deck = self.srs.get_deck_by_name_or_none(deck_name)
            if deck is None:
                raise MissingDeckException(deck_name)

            self.srs.add_card(deck, question, answer, flag, state)
            return

        raise AssertionError("Unreachable.")

    def act(self) -> AbstractActionState | None:
        deck_info = [
            f'name: "{it.name}", cards: {len(self.srs.get_cards_in_deck(it))}' for it in self.srs.get_all_decks()
        ]

        message = self._prompt_template.format(user_input=self.user_prompt, current_decks="\n".join(deck_info))
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self.llm_communicator.send_message(message)
                parsed = self._parse_commands(response)

                for command in parsed:
                    self._execute_command(command)

                return StateFinishedTask("No more tasks to execute.")  # TODO command count.
                # TODO Now, there is only one iterations - all commands must be sent the first time.
                #        Llama was absolutely unable to use [] to finish command execution.
                # message = (
                #     "The commands you sent were all executed successfully! "
                #     "If that was all, respond with []. If you have other commands to execute, send them."
                # )
            except JSONDecodeError as jde:
                message = f"Your answer must be a valid json string. Exception: {jde}. Please try again."
            except MissingDeckException as mde:
                if mde.deck_name is None:
                    message = "You must provide a deck name."
                else:
                    deck_names = [deck.name for deck in self.srs.get_all_decks()]
                    similar_deck_names = "\n".join(
                        f"* {deck_name}"
                        for deck_name in sorted(deck_names, key=lambda x: Levenshtein.distance(x, mde.deck_name))[:2]
                    )
                    message = (
                        f"The deck {mde.deck_name} does not exist. The following existing decks have similar names:\n\n"
                        f"{similar_deck_names}"
                        "\n\nIf one of this names roughly matches the name the user gave you, please just assume there "
                        "was an audio-to-text error and just use this deck name! Please try again."
                    )
            except Exception as e:  # TODO: We need a rollback-function here.
                message = f"An exception occurred during command execution: {e}. Please try again."

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateFinishedTask(AbstractActionState):

    def __init__(self, message: str):
        self.message = message

    def act(self) -> AbstractActionState | None:
        return None
