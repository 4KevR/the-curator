import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Optional

import pandas as pd
from rapidfuzz.distance import Levenshtein

from src.backend.modules.ai_assistant.history_manager import HistoryManager, SrsAction
from src.backend.modules.ai_assistant.progress_callback import ProgressCallback
from src.backend.modules.ai_assistant.states import AbstractActionState, ExceedingMaxAttemptsError
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


@dataclass(frozen=True)
class TaskInfo:
    original_prompt: str
    llm: AbstractLLM
    srs: AbstractSRS
    llama_index: LlamaIndexExecutor
    progress_callback: ProgressCallback
    history_manager: HistoryManager


class StateRewriteTask(AbstractActionState):
    _prompt_template = """
You are an AI assistant for a flashcard management system. The flashcard manager contains decks of flashcards (cards).
The user wants you to execute a task (adding, modifying or deleting cards or decks).
Please rewrite the following user input so that a LLM will understand it better.
You are given the history of user queries and the latest srs actions that were executed by the system.
It is now your task to update the user input to contain all necessary information for the LLM to execute the new single task.

Please make sure that you satisfy the following requirements:
* Do not change the content!
* The output should be approximately the same length as the input.
* Do not refer to the original task; include all necessary information in your output.
* The input is transcribed from voice, so please try to correct speech recognition errors like double words or miss-spelling.

This is the history of user queries:
{history}

This is the history of executed srs actions:
{actions}

The raw input is:
{user_input}

Only answer with the new task description!
""".strip()

    MIN_LENGTH_REWRITE = 250

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        srs: AbstractSRS,
        llama_index: LlamaIndexExecutor,
        progress_callback: ProgressCallback,
        history_manager: HistoryManager,
    ):
        self.info = TaskInfo(user_prompt, llm, srs, llama_index, progress_callback, history_manager)
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.history_manager = history_manager

    def act(self) -> AbstractActionState | None:
        message = self._prompt_template.format(
            history=str(self.history_manager.latest_queries),
            actions=self.history_manager.get_string_history(),
            user_input=self.user_prompt,
        )
        response = self.llm_communicator.send_message(message)
        return StateTask(self.info, response)


class StateTask(AbstractActionState):
    _prompt_template = """You are an assistant of a flashcard management system. You assist a user in executing tasks.
The flashcard management system consists of decks consisting of cards.

Given the user input, please select the best fitting task type.

1: Create a new, empty deck.
2: Create new flashcards from the user-provided information.
3: Renaming one or multiple decks.
4: Delete one or more existing decks.
5: Searching for cards and add found cards to a new or existing deck.
6: Searching for cards and edit found cards.
7: Searching for cards and delete found cards.
8: Create a new deck out of cards **that already are in the system**!.
9: Use previously created cards which need to be modified or deleted (this is often the case when users want to change existing cards they created in the same session - referring to questions and answers).

The user gave the following input:
{user_input}

Which task type fits the best? Only output the number!
""".strip()

    MAX_ATTEMPTS = 3

    def __init__(self, info: TaskInfo, user_prompt: str):
        self.info = info
        self.llm_communicator = LLMCommunicator(info.llm)
        self.user_prompt = user_prompt

    def act(self) -> AbstractActionState | None:
        for attempt in range(self.MAX_ATTEMPTS):
            if attempt == 0:
                message = self._prompt_template.format(
                    history=str(self.info.history_manager.latest_queries), user_input=self.user_prompt
                )
            else:
                message = "Please just respond with a the number of the best fitting task type."

            # TODO: Set max_tokens here. Everything but 1 token is wrong anyways -> can cap like 10 tokens.
            response = self.llm_communicator.send_message(message)

            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            response = response.strip()

            try:
                response_int = int(response)

                # 1,2,3,4 -> no search. 5,6,7,8 -> search
                if 1 <= response_int <= 4:
                    return StateTaskNoSearch(self.info, self.user_prompt)
                elif 5 <= response_int <= 8:
                    return StateTaskSearchDecks(self.info, self.user_prompt)
                elif response_int == 9:
                    return StateTaskReferencePreviousCards(self.info, self.user_prompt)
            except ValueError:
                pass

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
If you are unsure, rather include than exclude a deck.
If you have no information about which decks to search in, answer "all".

Make sure to exactly match the deck names.
**Do not answer anything else**!
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, info: TaskInfo, user_prompt: str):
        self.info = info
        self.llm_communicator = LLMCommunicator(info.llm)
        self.user_prompt = user_prompt

    def act(self) -> AbstractActionState | None:
        possible_decks = self.info.srs.get_all_decks()
        possible_deck_names = {deck.name for deck in possible_decks}

        message = self._prompt_template.format(
            user_input=self.user_prompt, decks=[str(it) for it in self.info.srs.get_all_decks()]
        )

        for attempt in range(self.MAX_ATTEMPTS):
            response = self.llm_communicator.send_message(message)
            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            response = response.strip()

            if response == "all":
                return StateTaskSearch(self.info, self.user_prompt, possible_decks)
            else:
                deck_strings = {s.strip() for s in response.split(",")}
                unknown_deck_strings = deck_strings - possible_deck_names

                if len(unknown_deck_strings) == 0:
                    decks = [it for it in possible_decks if it.name in deck_strings]
                    return StateTaskSearch(self.info, self.user_prompt, decks)

                message = (
                    f"The following deck names are unknown: {', '.join(unknown_deck_strings)}.\n"
                    'If you want to search in all decks, answer "all" (and nothing else!)."'
                    "If you want to search in a specific deck, answer the name of the deck.\n"
                    "If you want to search in multiple, specific decks, answer a comma-separated list of deck names.\n"
                    "Please make sure to exactly match the deck names."
                )

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateTaskReferencePreviousCards(AbstractActionState):
    def __init__(self, info: TaskInfo, user_prompt: str):
        self.info = info
        self.user_prompt = user_prompt

    def act(self) -> AbstractActionState | None:
        previous_cards: list[AbstractCard] = []
        seen_card_ids = set()
        for action in self.info.history_manager.srs_action_history:
            if isinstance(action.result_object, AbstractCard):
                card = action.result_object
                if card.id not in seen_card_ids:
                    try:
                        retrieved_card = self.info.srs.get_card(card.id)
                    except ValueError:
                        continue
                    previous_cards.append(retrieved_card)
                    seen_card_ids.add(card.id)

        if not previous_cards:
            raise ExceedingMaxAttemptsError(
                "No previous cards found in SRS history to reference for modification or deletion."
            )

        return StateStreamFoundCards(self.info, self.user_prompt, previous_cards)


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


If you have exact keywords to look for, you should use exact search. If you have one or more words/phrases to search for, but you cannot be sure that all fitting cards contain the keywords/phrases exactly (e.g. plural form, quotation marks, etc.), use fuzzy search. If you search for a topic, a category or a concept, or have any other search term that does not contain concrete search terms, use content search.

Please answer "exact", "fuzzy" or "content", and **nothing else**. All other details will be determined later.
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, info: TaskInfo, user_prompt: str, decks_to_search_in: list[AbstractDeck]):
        self.info = info
        self.llm_communicator = LLMCommunicator(info.llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in

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
                return StateKeywordSearch(self.info, self.user_prompt, self.decks_to_search_in)
            if response == "fuzzy":
                return StateFuzzySearch(self.info, self.user_prompt, self.decks_to_search_in)
            if response == "content":
                return StateContentSearch(self.info, self.user_prompt, self.decks_to_search_in)

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateKeywordSearch(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system.
You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:

{user_input}

You already decided that you have to search for cards, and that you want to use keyword search. You may search for one or more keywords.
Please fill in the following template. Make sure to produce valid json.
[
{{
    "search_substring": "<search_substring_here>",
    "search_in_question": <bool here>,
    "search_in_answer": <bool here>,
    "case_sensitive": <bool here>
}}
]

If you are unsure, use these defaults:
  search_in_question: true
  search_in_answer: true
  case_sensitive: false

Please answer only with the json list of filled-in, valid json object as described above.
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, info: TaskInfo, user_prompt: str, decks_to_search_in: list[AbstractDeck]):
        self.info = info
        self.llm_communicator = LLMCommunicator(info.llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in

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

                return StateVerifySearch(self.info, self.user_prompt, self.decks_to_search_in, searchers)

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
[
{{
    "search_substring": "<search_substring_here>",
    "search_in_question": <bool here>,
    "search_in_answer": <bool here>,
    "case_sensitive": <bool here>,
    "fuzzy": <float here>
}}
]

If you are unsure, use these defaults:
  search_in_question: true
  search_in_answer: true
  case_sensitive: false
  fuzzy: 0.8

If multiple keywords are specified, each card that matches at least one of the keywords will be returned. Only use multiple keywords if necessary; do not use substrings of other keywords.

Please answer only with the json list of filled-in, valid json object as described above.
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, info: TaskInfo, user_prompt: str, decks_to_search_in: list[AbstractDeck]):
        self.info = info
        self.llm_communicator = LLMCommunicator(info.llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in

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

                return StateVerifySearch(self.info, self.user_prompt, self.decks_to_search_in, searchers)

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

    def __init__(self, info: TaskInfo, user_prompt: str, decks_to_search_in: list[AbstractDeck]):
        self.info = info
        self.llm_communicator = LLMCommunicator(info.llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in

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

                searcher = LlamaIndexSearcher(executor=self.info.llama_index, prompt=parsed["search_prompt"])
                return StateVerifySearch(self.info, self.user_prompt, self.decks_to_search_in, [searcher])

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
        info: TaskInfo,
        user_prompt: str,
        decks_to_search_in: list[AbstractDeck],
        searchers: list[AbstractCardSearcher],
    ):
        self.info = info
        self.llm_communicator = LLMCommunicator(info.llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.searchers = searchers

    def act(self) -> AbstractActionState | None:
        all_cards: list[AbstractCard] = [
            card for deck in self.decks_to_search_in for card in self.info.srs.get_cards_in_deck(deck)
        ]
        found_cards = AbstractCardSearcher.union_search_all(self.searchers, all_cards)

        for attempt in range(self.MAX_ATTEMPTS):
            if attempt == 0:
                if len(found_cards) <= self.SAMPLE_SIZE:
                    sample = found_cards
                else:
                    sample = pd.Series(found_cards).sample(self.SAMPLE_SIZE).to_list()

                message = self._prompt_template.format(
                    user_input=self.user_prompt,
                    amount_cards=len(found_cards),
                    cards_sample="\n\n".join(str(it) for it in sample),
                )
            else:
                message = "Your answer must be either 'yes' or 'no'."

            response = self.llm_communicator.send_message(message)

            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            resp = find_substring_in_llm_response_or_null(response, "yes", "no", True)

            # no you can not change this to resp == true.
            # noinspection PySimplifyBooleanCheck
            if resp is True:
                return StateTaskWorkOnFoundCards(self.info, self.user_prompt, self.decks_to_search_in, found_cards)
            elif resp is False:
                raise NotImplementedError()  # TODO! Add a *one-time-only* loopback here.

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateTaskWorkOnFoundCards(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:
{user_input}

After searching, what should the system do?

1: Add all found cards to an existing or newly created deck.
2: Delete all found cards.
3: Edit all or some of the found cards.
4: Delete some of the found cards.

Please **only** respond with the number of the operation that fits the user's query the best.
""".strip()
    MAX_ATTEMPTS = 3
    SAMPLE_SIZE = 3

    def __init__(
        self,
        info: TaskInfo,
        user_prompt: str,
        decks_to_search_in: list[AbstractDeck],
        found_cards: list[AbstractCard],
    ):
        self.info = info
        self.llm_communicator = LLMCommunicator(info.llm)
        self.user_prompt = user_prompt
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
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

            if response == "2":
                for card in self.found_cards:
                    action = SrsAction.delete_card(self.info.srs, card)
                    self.info.history_manager.add_action(action)
                    self.info.progress_callback.handle(action.description, True)

                return StateFinishedTask(f"{len(self.found_cards)} cards deleted.")

            if response == "1":
                return StateSearchCopyToDeck(self.info, self.user_prompt, self.found_cards)

            if response in "34":
                return StateStreamFoundCards(self.info, self.user_prompt, self.found_cards)

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateSearchCopyToDeck(AbstractActionState):
    _prompt_template = """You are an ai assistant of a flashcard management system. You assist a user and execute tasks for them.

You already searched for cards and decided to add them to a (new or existing) deck. Now you have to decide to which (new or existing) deck to add the cards to.

The user prompt is:
{user_input}

Currently, the following decks exist:
{deck_list}

Now please decide which deck to add the cards to.

* If the user wants to create a new deck, please answer with the name the user told you to. **Use the exact name the user told you to!**
* If the user wants to add the cards to an existing deck, please answer with the name of the deck.
* If the user does not specify whether to use an existing deck or to create a new deck, and a deck with a very similar name already exist, please answer the name of the existing deck, else the name of the new deck.
* If the user tells you to add the cards to 'the deck' and only one deck exists, please use that one.

Now please answer the name of the deck that the search result should be saved to. Please answer only with the name of the deck, and nothing else.
""".strip()

    def __init__(
        self,
        info: TaskInfo,
        user_prompt: str,
        found_cards: list[AbstractCard],
    ):
        self.info = info
        self.llm_communicator = LLMCommunicator(info.llm)
        self.user_prompt = user_prompt
        self.found_cards = found_cards

    def act(self) -> Optional["AbstractActionState"]:
        deck_list = "\n".join([f" * {it.name}" for it in self.info.srs.get_all_decks()])
        prompt = self._prompt_template.format(deck_list=deck_list, user_input=self.user_prompt)
        deck_name = self.llm_communicator.send_message(prompt)

        deck_created = False
        deck = self.info.srs.get_deck_by_name_or_none(deck_name)
        if deck is None:
            deck = self.info.srs.add_deck(deck_name)
            action = SrsAction.add_deck(self.info.srs, deck)
            self.info.progress_callback.handle(action.description, True)
            self.info.history_manager.add_action(action)
            deck_created = True

        for card in self.found_cards:
            action = SrsAction.copy_card_to(self.info.srs, card, deck)
            self.info.progress_callback.handle(action.description, True)
            self.info.history_manager.add_action(action)

        if deck_created:
            return StateFinishedTask(f"{len(self.found_cards)} cards copied to newly created deck {deck_name}.")
        else:
            return StateFinishedTask(f"{len(self.found_cards)} cards copied to existing deck {deck_name}.")


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
        info: TaskInfo,
        user_prompt: str,
        found_cards: list[AbstractCard],
    ):
        self.info = info
        self.llm_communicator = LLMCommunicator(info.llm)
        self.user_prompt = user_prompt
        self.found_cards = found_cards

    def _execute_command(self, response: str, card: AbstractCard):
        response = response.strip()

        if response == "do_nothing":
            return
        if response == "delete_card":
            actor = SrsAction.delete_card(self.info.srs, card)
            self.info.progress_callback.handle(actor.description, True)
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
            action = SrsAction.edit_card_question(self.info.srs, card, parsed["question"])
            self.info.history_manager.add_action(action)
            self.info.progress_callback.handle(action.description, True)
        if "answer" in parsed and parsed["answer"] != card.answer:
            action = SrsAction.edit_card_answer(self.info.srs, card, parsed["answer"])
            self.info.history_manager.add_action(action)
            self.info.progress_callback.handle(action.description, True)
        if "flag" in parsed:
            flag = Flag.from_str(parsed["flag"])
            if flag != card.flag:
                action = SrsAction.edit_card_flag(self.info.srs, card, flag)
                self.info.history_manager.add_action(action)
                self.info.progress_callback.handle(action.description, True)
        if "state" in parsed:
            state = CardState.from_str(parsed["state"])
            if state != card.state:
                action = SrsAction.edit_card_state(self.info.srs, card, state)
                self.info.history_manager.add_action(action)
                self.info.progress_callback.handle(action.description, True)

        return

    def act(self) -> AbstractActionState | None:
        for card in self.found_cards:
            message = self._prompt_template.format(
                user_task=self.user_prompt,
                question=card.question,
                answer=card.answer,
                flag=card.flag.value,
                state=card.state.value,
            )
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

These were the user's last srs actions, you can use information from them to execute the task if nessary.
Often, users have the intention to work on previoulsly created decks or cards.

{last_actions}

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

If you want to execute one or more functions, return them inside a json array.

Please answer only with the filled-in, valid json but to not a markdown prefix for the json.

Rather use the missing_information task than to guess the user's intention for fill-in fields.
Do not generate any text for the fields that are not present in the user input. Leave the respective fields empty.
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, info: TaskInfo, user_prompt: str):
        self.info = info
        self.llm_communicator = LLMCommunicator(info.llm)
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

        return parsed

    def _execute_command(self, cmd_dict: dict[Any, Any]) -> Optional[AbstractActionState]:
        # execute tasks
        if cmd_dict["task"] == "create_deck":
            deck_name = cmd_dict["name"]
            if not deck_name:
                return StateFinishedDueToMissingInformation("You must provide a deck name.")
            deck = self.info.srs.get_deck_by_name_or_none(deck_name)
            if deck is not None:
                raise ValueError("Deck already exists")
            action = SrsAction.add_deck(self.info.srs, deck_name)
        elif cmd_dict["task"] == "rename_deck":
            old_name = cmd_dict["old_name"]
            new_name = cmd_dict["new_name"]
            if not old_name or not new_name:
                return StateFinishedDueToMissingInformation("You must provide both old and new deck names.")
            deck = self.info.srs.get_deck_by_name_or_none(old_name)
            if deck is None:
                raise MissingDeckException(old_name)
            if self.info.srs.get_deck_by_name_or_none(new_name) is not None:
                raise ValueError(f"New name {new_name} already exists")
            action = SrsAction.rename_deck(self.info.srs, deck, new_name)
        elif cmd_dict["task"] == "delete_deck":
            name = cmd_dict["name"]
            if not name:
                return StateFinishedDueToMissingInformation("You must provide a deck name to delete.")
            deck = self.info.srs.get_deck_by_name_or_none(name)
            if deck is None:
                raise MissingDeckException(name)
            action = SrsAction.delete_deck(self.info.srs, deck)
        elif cmd_dict["task"] == "add_card":
            deck_name = cmd_dict["deck_name"]
            if not deck_name:
                return StateFinishedDueToMissingInformation("You must provide a deck name to add the card to.")
            question = cmd_dict["question"]
            if not question:
                return StateFinishedDueToMissingInformation("You must provide a question for the card.")
            answer = cmd_dict["answer"]
            if not answer:
                return StateFinishedDueToMissingInformation("You must provide an answer for the card.")
            state = cmd_dict["state"]
            if not state:
                state = CardState.NEW
            else:
                state = CardState.from_str(state)
            flag = cmd_dict["flag"]
            if not flag:
                flag = Flag.NONE
            else:
                flag = Flag.from_str(flag)
            deck = self.info.srs.get_deck_by_name_or_none(deck_name)
            if deck is None:
                raise MissingDeckException(deck_name)
            action = SrsAction.add_card(self.info.srs, deck, question, answer, flag, state)
        else:
            raise AssertionError("Unreachable.")
        self.info.history_manager.add_action(action)
        self.info.progress_callback.handle(action.description, True)
        return None

    def act(self) -> AbstractActionState | None:
        deck_info = [
            f'name: "{it.name}", cards: {len(self.info.srs.get_cards_in_deck(it))}'
            for it in self.info.srs.get_all_decks()
        ]

        message = self._prompt_template.format(
            user_input=self.user_prompt,
            current_decks="\n".join(deck_info),
            last_actions=self.info.history_manager.get_string_history(),
        )
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self.llm_communicator.send_message(message)
                parsed = self._parse_commands(response)

                for command in parsed:
                    result_state = self._execute_command(command)
                    if result_state:  # If a state is returned (e.g., missing_information)
                        return result_state

                return StateFinishedTask(f"Executed {len(parsed)} commands.")
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
                    deck_names = [deck.name for deck in self.info.srs.get_all_decks()]
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


class StateFinishedDueToMissingInformation(AbstractActionState):
    def __init__(self, message: str):
        self.message = message

    def act(self) -> AbstractActionState | None:
        return None
