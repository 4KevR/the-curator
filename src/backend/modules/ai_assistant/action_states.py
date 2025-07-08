import json
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Any, Callable, Optional

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
    def act(
        self, progress_callback: Callable[[str, Optional[bool]], None] | None = None
    ) -> Optional["AbstractActionState"]:
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
You are an assistant of a flashcard management system. You assist a user in interacting in three ways:
1. Interacting with the flashcard system (creating/modifying/deleting cards/decks etc.),
2. Answering questions about the content of the flashcards,
3. Entering and managing study sessions (e.g., starting to learn or review a deck).

The user gave the following prompt:

{user_input}

If you think the user wants you to **interact** with the flashcard system (e.g. creating, modifying, or deleting cards or decks), please answer "task".
If the user wants you to answer a question about the content of the flashcards, please answer "question".
If the user wants to **enter study mode**, such as learning or reviewing a specific deck, please answer "study".
Do not answer anything else.
"""
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS, llama_index_executor: LlamaIndexExecutor):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs
        self.llama_index_executor = llama_index_executor

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
        if self.srs.study_mode is True:
            return StateClassify(self.user_prompt, self.llm, self.srs)
        else:
            for attempt in range(self.MAX_ATTEMPTS):
                if attempt == 0:
                    message = self._prompt_template.format(user_input=self.user_prompt)
                else:
                    message = "Your answer must be either 'question', 'task' or 'study'."

                response = self.llm_communicator.send_message(message)

                response = remove_block(response, "think")
                response = response.replace('"', "").replace("'", "")
                resp = find_substring_in_llm_response_or_null(response, "question", "task", True)

                if resp is True:
                    return StateQuestion(self.user_prompt, self.llm, self.llama_index_executor)
                elif resp is False:
                    return StateTask(self.user_prompt, self.llm, self.srs, self.llama_index_executor)
                elif resp is None:
                    if "study" in response.lower():
                        return StateStartLearn(self.user_prompt, self.llm, self.srs)

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

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
        fitting_nodes = self.llama_index_executor.search_cards(self.user_prompt)
        fitting_nodes = sorted(fitting_nodes, key=lambda x: x[1], reverse=True)[:5]
        fitting_nodes = "\n".join(fn[0] for fn in fitting_nodes)

        message = self._prompt_template.format(user_input=self.user_prompt, cards=fitting_nodes)
        response = self.llm_communicator.send_message(message)
        return StateAnswer(response)


class StateAnswer(AbstractActionState):
    def __init__(self, answer: str):
        self.answer = answer

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> None:
        return None  # final state


class StateRewriteTask(AbstractActionState):
    _prompt_template = """
You are an AI assistant for a flashcard management system. The flashcard manager contains decks of flashcards (cards).
The user wants you to execute a task (adding, modifying or deleting cards or decks).
Please rewrite the following user input so that a LLM will understand it better.

Please make sure that you satisfy the following requirements:
* Do not change the content!
* The output should be approximately the same length as the input.
* Do not refer to the original task; include all necessary information in your output.
* The input is transcribed from voice, so please try to correct speech recognition errors like double words or miss-spelling.

The raw input is:
{user_input}

Only answer with the new task description!
""".strip()

    MIN_LENGTH_REWRITE = 250

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:

        if len(self.user_prompt) >= self.MIN_LENGTH_REWRITE:
            message = self._prompt_template.format(user_input=self.user_prompt)
            response = self.llm_communicator.send_message(message)
            return StateTask(response, self.llm, self.srs)
        else:
            return StateTask(self.user_prompt, self.llm, self.srs)


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

The user gave the following input:
{user_input}

Which task type fits the best? Only output the number!
""".strip()

    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS, llama_index_executor: LlamaIndexExecutor):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs
        self.llama_index_executor = llama_index_executor

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
        for attempt in range(self.MAX_ATTEMPTS):
            if attempt == 0:
                message = self._prompt_template.format(user_input=self.user_prompt)
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
                    return StateTaskNoSearch(self.user_prompt, self.llm, self.srs)
                elif 5 <= response_int <= 8:
                    return StateTaskSearchDecks(self.user_prompt, self.llm, self.srs, self.llama_index_executor)
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

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS, llama_index_executor: LlamaIndexExecutor):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.srs = srs
        self.user_prompt = user_prompt
        self.possible_decks = self.srs.get_all_decks()
        self.possible_deck_names = {deck.name for deck in self.possible_decks}
        self.llama_index_executor = llama_index_executor

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
        message = self._prompt_template.format(
            user_input=self.user_prompt, decks=[str(it) for it in self.srs.get_all_decks()]
        )

        for attempt in range(self.MAX_ATTEMPTS):
            response = self.llm_communicator.send_message(message)
            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            response = response.strip()

            if response == "all":
                return StateTaskSearch(
                    self.user_prompt, self.llm, self.possible_decks, self.srs, self.llama_index_executor
                )
            else:
                deck_strings = {s.strip() for s in response.split(",")}
                unknown_deck_strings = deck_strings - self.possible_deck_names

                if len(unknown_deck_strings) == 0:
                    decks = [it for it in self.possible_decks if it.name in deck_strings]
                    return StateTaskSearch(self.user_prompt, self.llm, decks, self.srs, self.llama_index_executor)

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


If you have exact keywords to look for, you should use exact search. If you have one or more words/phrases to search for, but you cannot be sure that all fitting cards contain the keywords/phrases exactly (e.g. plural form, quotation marks, etc.), use fuzzy search. If you search for a topic, a category or a concept, or have any other search term that does not contain concrete search terms, use content search.

Please answer "exact", "fuzzy" or "content", and **nothing else**. All other details will be determined later.
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        decks_to_search_in: list[AbstractDeck],
        srs: AbstractSRS,
        llama_index_executor: LlamaIndexExecutor,
    ):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs
        self.llama_index_executor = llama_index_executor

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
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
                return StateContentSearch(
                    self.user_prompt, self.llm, self.decks_to_search_in, self.srs, self.llama_index_executor
                )

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

    def __init__(self, user_prompt: str, llm: AbstractLLM, decks_to_search_in: list[AbstractDeck], srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
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

    def __init__(self, user_prompt: str, llm: AbstractLLM, decks_to_search_in: list[AbstractDeck], srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
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

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        decks_to_search_in: list[AbstractDeck],
        srs: AbstractSRS,
        llama_index_executor: LlamaIndexExecutor,
    ):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs
        self.llama_index_executor = llama_index_executor

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
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

                searcher = LlamaIndexSearcher(executor=self.llama_index_executor, prompt=parsed["search_prompt"])
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

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
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

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
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
                    self.srs.delete_card(card)

                return StateFinishedTask(f"{len(self.found_cards)} cards deleted.")

            if response == "1":
                return StateSearchCopyToDeck(self.user_prompt, self.llm, self.srs, self.found_cards)

            if response in "34":
                return StateStreamFoundCards(
                    self.user_prompt, self.llm, self.decks_to_search_in, self.srs, self.found_cards
                )

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
        user_prompt: str,
        llm: AbstractLLM,
        srs: AbstractSRS,
        found_cards: list[AbstractCard],
    ):
        self.user_prompt = user_prompt
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.srs = srs
        self.found_cards = found_cards

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> Optional["AbstractActionState"]:
        deck_list = "\n".join([f" * {it.name}" for it in self.srs.get_all_decks()])
        prompt = self._prompt_template.format(deck_list=deck_list, user_input=self.user_prompt)
        deck_name = self.llm_communicator.send_message(prompt)

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

    def _execute_command(
        self, response: str, card: AbstractCard, progress_callback: Callable[[str, Optional[bool]], None] | None = None
    ):
        response = response.strip()

        if response == "do_nothing":
            return
        if response == "delete_card":
            self.srs.delete_card(card)
            if progress_callback:
                progress_callback(f"Deleted card: {card.question} - {card.answer}", True)
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
            if progress_callback:
                progress_callback(
                    f"Edited question of card: {card.question} - {card.answer} to {parsed['question']}", True
                )
        if "answer" in parsed and parsed["answer"] != card.answer:
            self.srs.edit_card_answer(card, parsed["answer"])
            if progress_callback:
                progress_callback(f"Edited answer of card: {card.question} - {card.answer} to {parsed['answer']}", True)
        if "flag" in parsed and parsed["flag"] != card.flag:
            flag = Flag.from_str(parsed["flag"])
            self.srs.edit_card_flag(card, flag)
            if progress_callback:
                progress_callback(f"Set flag of card: {card.question} - {card.answer} to {parsed['flag']}", True)
        if "state" in parsed and parsed["state"] != card.state:
            state = CardState.from_str(parsed["state"])
            self.srs.edit_card_state(card, state)
            if progress_callback:
                progress_callback(f"Set state of card: {card.question} - {card.answer} to {parsed['state']}", True)

        return

    def act(self, progress_callback: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
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
                    self._execute_command(response, card, progress_callback)
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
    MAX_ATTEMPTS = 3

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

    def _execute_command(
        self, cmd_dict: dict[Any, Any], progress_callback: Callable[[str, Optional[bool]], None] | None = None
    ) -> None:
        # execute tasks
        if cmd_dict["task"] == "create_deck":
            deck_name = cmd_dict["name"]
            deck = self.srs.get_deck_by_name_or_none(deck_name)
            if deck is not None:
                raise ValueError("Deck already exists")
            self.srs.add_deck(deck_name)
            if progress_callback:
                progress_callback(f"Created deck: {cmd_dict['name']}", True)
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
            if progress_callback:
                progress_callback(f"Renamed deck: {old_name} to {new_name}", True)
            return

        if cmd_dict["task"] == "delete_deck":
            name = cmd_dict["name"]
            deck = self.srs.get_deck_by_name_or_none(name)
            if deck is None:
                raise MissingDeckException(name)
            self.srs.delete_deck(deck)
            if progress_callback:
                progress_callback(f"Deleted deck: {name}", True)
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
            if progress_callback:
                progress_callback(f"Added card to deck {deck_name}: {question} - {answer} (flag: {flag})", True)
            return

        raise AssertionError("Unreachable.")

    def act(self, progress_callback: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
        deck_info = [
            f'name: "{it.name}", cards: {len(self.srs.get_cards_in_deck(it))}' for it in self.srs.get_all_decks()
        ]

        message = self._prompt_template.format(user_input=self.user_prompt, current_decks="\n".join(deck_info))
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self.llm_communicator.send_message(message)
                parsed = self._parse_commands(response)

                for command in parsed:
                    self._execute_command(command, progress_callback)

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

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
        return None


class StateStartLearn(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system.
You help the user better remember the content of flashcards.

The user is about to start studying a deck and gave the following input:

{user_input}

The following decks are available:

{decks}

Your task is to determine which deck the user wants to study.
Return the exact name of the matching deck.

If there's no exact match, you may fix only minor errors (e.g., one wrong/missing character or common ASR mistakes).
Matching is case-sensitive.
If no reasonable match is found, respond with "None".

**Return only the exact name of the selected deck, or "None". Do not respond with anything else.**
""".strip()
    MAX_ATTEMPTS = 5

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.srs = srs
        self.user_prompt = user_prompt

    def act(self, progress_callback: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
        deck_info = [
            f'name: "{it.name}", cards: {len(self.srs.get_cards_in_deck(it))}' for it in self.srs.get_all_decks()
        ]

        message = self._prompt_template.format(user_input=self.user_prompt, decks="\n".join(deck_info))

        for attempt in range(self.MAX_ATTEMPTS):
            response = self.llm_communicator.send_message(message)
            response = remove_block(response, "think").replace('"', "").strip()  # No replacement for '

            deck_name = response

            if deck_name == "None":
                return StateFinishedLearn(
                    "The deck you want to learn is not found. Please check the name and try again."
                )
            else:
                deck = self.srs.get_deck_by_name_or_none(deck_name)
                if deck is not None:
                    cards = self.srs.get_cards_in_deck(deck)
                    if len(cards) == 0:
                        return StateFinishedLearn("The deck you want to learn is empty.")
                    else:
                        self.srs.study_mode = True
                        self.srs.cards_to_be_learned = cards
                        self.srs.card_index_currently_being_learned = 0
                        first_card_question = self.srs.cards_to_be_learned[
                            self.srs.card_index_currently_being_learned
                        ].question

                        if progress_callback:
                            progress_callback(
                                f"Learning session for deck '{deck.name}' initialized successfully.", True
                            )
                        msg_to_user = f"Enjoy your learning!\n Question: {first_card_question}\n"
                        return StateFinishedLearn(msg_to_user)
                else:
                    message = f"""No matching deck was found based on your previous response: '{deck_name}'.
                        **Return only the exact name of the selected deck, or "None". Do not respond with anything else.**"""

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateClassify(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard study system. Your job is to classify the user's input based on its content.

Given the following user input:

{user_input}

The current flashcard question is:

{card_question}

Analyze the user's input based on the following rules:
- If the input **only** contains an answer to the current flashcard, return "answer".
- If the input **only** contains a command to end the study session (e.g., "stop", "quit", "end"), return "end".
  End commands will only appear at the end of the input.
- If the input contains **both** an answer and a command to end the study session, return "both".

Examples:
- "NLP stands for natural language processing." → answer
- "quit" → end
- "The answer is Paris. Stop." → both
- "The answer is Paris. End study." → both

**Do not explain. Do not add any other text. Respond with only **one word**: 'answer', 'end', or 'both'.**
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs

    def act(self, progress_callback: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
        card_question = self.srs.cards_to_be_learned[self.srs.card_index_currently_being_learned].question
        message = self._prompt_template.format(user_input=self.user_prompt, card_question=card_question)

        for attempt in range(self.MAX_ATTEMPTS):
            response = self.llm_communicator.send_message(message)
            response = remove_block(response, "think").replace('"', "").replace("'", "").strip()
            resp = find_substring_in_llm_response_or_null(response, "answer", "end", True)

            if resp is True:
                return StateJudgeAnswer(self.user_prompt, self.llm, self.srs, False)
            elif resp is False:
                self.srs.study_mode = False
                if progress_callback:
                    progress_callback("Exit study mode.", True)
                return StateFinishedLearn("Exit study mode.")
            elif resp is None:
                if "both" in response.lower():
                    return StateExtractAnswer(self.user_prompt, self.llm, self.srs)

            message = "Return only one of the following: 'answer'', 'end', or 'both'. Do not return anything else."

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateExtractAnswer(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard study system.

Given the following user input:

{user_input}

The user's input contains two parts:
1. an answer to the current flashcard, and
2. a command to end the study session.

Your task is to extract only the answer part of the input.

Do not modify, correct, or interpret the content. Just return the answer exactly as it appears.

**Return only the answer part. Do not include the end command, any explanation, or anything else.**
""".strip()

    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
        message = self._prompt_template.format(user_input=self.user_prompt)

        for attempt in range(self.MAX_ATTEMPTS):
            response = self.llm_communicator.send_message(message)
            response = remove_block(response, "think").strip()
            if response != "":  # not always stable
                self.user_prompt = response
                return StateJudgeAnswer(self.user_prompt, self.llm, self.srs, True)

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateJudgeAnswer(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard study system. Please evaluate the user's answer.

The current flashcard is as follows:
- Question: {card_question}
- Correct Answer: {card_answer}

The user gave the following answers to the questions on the card:

{user_answer}


Your task is to evaluate the user's answer based primarily on the correct answer above, considering the following principles:

1. The correct answer is the most important criterion. Semantic correctness and alignment with the intended meaning are crucial.
2. Minor spelling or grammatical mistakes can be ignored as long as the meaning is clearly conveyed.
3. Answers must demonstrate real understanding. Vague or generic responses like "The answer is the answer" are not acceptable.

Based on this, return only one of the following evaluations:

- 'again': the user clearly did not remember the answer and should try again.
- 'hard': the user struggled or was mostly incorrect, but showed partial understanding.
- 'good': the user remembered the answer reasonably well with minor issues.
- 'easy': the user recalled the answer very easily and accurately.

**Return only one word of: 'again', 'hard', 'good', or 'easy'. Do not return anything else.**
""".strip()

    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS, end: bool):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs
        self.end = end

    def act(self, progress_callback: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
        card = self.srs.cards_to_be_learned[self.srs.card_index_currently_being_learned]
        message = self._prompt_template.format(
            user_answer=self.user_prompt, card_question=card.question, card_answer=card.answer
        )

        for attempt in range(self.MAX_ATTEMPTS):
            response = self.llm_communicator.send_message(message)
            response = remove_block(response, "think").replace('"', "").replace("'", "").replace(".", "").strip()
            try:
                self.srs.set_memory_grade(card.id, response)
                msg_to_user = f"Your answer to the previous card was rated as: {response}.\n"

                if self.srs.card_index_currently_being_learned == len(self.srs.cards_to_be_learned) - 1:
                    msg_to_user += "Congratulations on learning all the cards!"
                    self.srs.study_mode = False
                    if progress_callback:
                        progress_callback("Exit study mode.", True)
                    return StateFinishedLearn(msg_to_user)
                elif self.end:
                    msg_to_user += "Exit study mode."
                    self.srs.study_mode = False
                    if progress_callback:
                        progress_callback("Exit study mode.", True)
                    return StateFinishedLearn(msg_to_user)
                else:
                    self.srs.card_index_currently_being_learned += 1
                    next_card = self.srs.cards_to_be_learned[self.srs.card_index_currently_being_learned]
                    next_card_question = f"Question: {next_card.question}"
                    msg_to_user += next_card_question
                    return StateFinishedLearn(msg_to_user)
            except:  # noqa: E722
                pass

            message = "Return only one of: 'again', 'hard', 'good', or 'easy'. Do not return anything else."

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateFinishedLearn(AbstractActionState):

    def __init__(self, message: str):
        self.message = message

    def act(self, _: Callable[[str, Optional[bool]], None] | None = None) -> AbstractActionState | None:
        return None
