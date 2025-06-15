import json
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from src.backend.modules.helpers.string_util import remove_block, find_substring_in_llm_response_or_null
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.llm.llm_communicator import LLMCommunicator
from src.backend.modules.search.abstract_card_searcher import AbstractCardSearcher
from src.backend.modules.search.llama_index import LlamaIndexExecutor, LlamaIndexSearcher
from src.backend.modules.search.search_by_substring import SearchBySubstring
from src.backend.modules.search.search_by_substring_fuzzy import SearchBySubstringFuzzy
from src.backend.modules.srs.abstract_srs import AbstractSRS, AbstractDeck, AbstractCard
from src.backend.modules.srs.testsrs.testsrs import TestFlashcardManager, Flag, CardState


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


class StateAction(AbstractActionState):
    _prompt_template = (
        "You are an assistant of a flashcard management system. You assist a user in either executing a task"
        " (creating/modifying/deleting cards/decks etc.) or in answering questions about the content of the flashcards."
        " If the user asks you to find cards without stating a question that he wants answered, respond task.\n\n"
        "The user gave the following input:\n\n{user_input}\n\n"
        'If the user wants to execute a task, please answer "task". If the user wants to answer a question, answer'
        ' "question". Do not answer anything else.'
    )
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs

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
                return StateQuestion(self.user_prompt, self.llm)
            elif resp is False:
                return StateTask(self.user_prompt, self.llm, self.srs)

        raise RuntimeError("Exceeding maximum number of attempts.")


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

    def __init__(self, user_prompt: str, llm: AbstractLLM):
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt

    def act(self) -> AbstractActionState | None:
        fitting_nodes = LlamaIndexExecutor().search_cards(self.user_prompt)
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
    _prompt_template = (
        "You are an assistant of a flashcard management system. You assist a user in executing tasks "
        "(creating/modifying/deleting cards/decks etc.).\n\n"
        "The user gave the following input:\n\n"
        "{user_input}\n\n"
        'To execute this prompt, do you have to **search for cards**? **Only** answer "yes" or "no", nothing else!'
    )
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs

    def act(self) -> AbstractActionState | None:
        for attempt in range(self.MAX_ATTEMPTS):
            if attempt == 0:
                message = self._prompt_template.format(user_input=self.user_prompt)
            else:
                message = "Your answer must be either 'yes' or 'no'."

            response = self.llm_communicator.send_message(message)

            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            resp = find_substring_in_llm_response_or_null(response, "yes", "no", True)

            if resp is True:
                return StateTaskSearchDecks(self.user_prompt, self.llm, self.srs)
            elif resp is False:
                return StateTaskNoSearch(self.user_prompt, self.llm, self.srs)

        raise RuntimeError("Exceeding maximum number of attempts.")


class StateTaskSearchDecks(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:

{user_input}

You already decided that you have to search for cards. Now you have to decide in which decks you want to search.
The following decks are available:

{decks}

If you want to search in all decks, answer "all". If you want to search in a specific deck, answer the name of the deck.
If you want to search in multiple, specific decks, answer a comma-separated list of deck names. Please make sure to 
exactly match the deck names. **Do not answer anything else**!
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.srs = srs
        self.user_prompt = user_prompt
        self.possible_decks = srs.get_all_decks()
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
                    'If you want to search in all decks, answer "all" (and nothing else!). If you want to search in a specific deck, answer the name of the deck.\n'
                    "If you want to search in multiple, specific decks, answer a comma-separated list of deck names.\n"
                    "Please make sure to exactly match the deck names."
                )

        raise RuntimeError("Exceeding maximum number of attempts.")


class StateTaskSearch(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:

{user_input}

You already decided that you have to search for cards. Please decide now how you want to search for cards. Your options are:

* Search for keyword (exact): You give me a keyword and I will search for all cards that contain this keyword.
You will be able to specify whether you want to search in the question or the answer or both. You can also decide
whether the search should be case sensitive or not.

* Search for keyword (fuzzy search): You give me a keyword and I will search for all cards that contain this keyword, or
contain a 'similar' substring. You will be able to specify whether you want to search in the question or the answer or both.
You can also decide whether the search should be case sensitive or not.

* Search cards with fitting content: You give me a search prompt and I will search for cards that fit the search prompt.
The search is *not* limited to exact wording, but searches for cards with fitting content.


If you have an exact keyword to look for, you should use exact search. If you have a word to search for, but it can be slightly different
(e.g. plural form, etc.) use fuzzy search. In all remaining cases, use content-based search.

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

        raise RuntimeError("Exceeding maximum number of attempts.")


class StateKeywordSearch(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:

{user_input}

You already decided that you have to search for cards, and that you want to use keyword search.
Please fill in the following template. Make sure to produce valid json.
{{
    "search_substring": "<search_substring_here>",
    "search_in_question": <bool here>,
    "search_in_answer": <bool here>,
    "case_sensitive": <bool here>
}}

Please answer only with the filled-in, valid json. You may only send a single command at a time!
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
                return StateVerifySearch(self.user_prompt, self.llm, self.decks_to_search_in, self.srs, searcher)

            except Exception as e:
                message = f"Your answer must be a valid json string. Exception: {e}. Please try again."

        raise RuntimeError("Exceeding maximum number of attempts.")


class StateFuzzySearch(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

The user gave the following input:

{user_input}

You already decided that you have to search for cards, and that you want to use fuzzy keyword search.
Please fill in the following template. Make sure to produce valid json.
{{
    "search_substring": "<search_substring_here>",
    "search_in_question": <bool here>,
    "search_in_answer": <bool here>,
    "case_sensitive": <bool here>,
    "fuzzy": <float here>
}}

Usually, 0.8 is a good threshold for fuzzy search. Depending on your use case, you might want to adjust this value.
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
                return StateVerifySearch(self.user_prompt, self.llm, self.decks_to_search_in, self.srs, searcher)

            except Exception as e:
                message = f"Your answer must be a valid json string. Exception: {e}. Please try again."

        raise RuntimeError("Exceeding maximum number of attempts.")


class StateContentSearch(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. You assist a user in executing tasks (creating/modifying/deleting cards/decks etc.).

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
                return StateVerifySearch(self.user_prompt, self.llm, self.decks_to_search_in, self.srs, searcher)

            except Exception as e:
                message = f"Your answer must be a valid json string. Exception: {e}. Please try again."

        raise RuntimeError("Exceeding maximum number of attempts.")


class StateVerifySearch(AbstractActionState):
    _prompt_template = (
        "You are an assistant of a flashcard management system. You assist a user in executing tasks "
        "(creating/modifying/deleting cards/decks etc.).\n\n"
        "The user gave the following input:\n\n"
        "{user_input}\n\n"
        "You decided to search for cards. Your search returned {amount_cards} cards."
        "Here is a sample of the cards you found:\n\n"
        "{cards_sample}\n\n"
        "If something went wrong, e.g. you didn't find any cards, but you expected to find some cards, you can go back"
        " and use a different search method."
        " If you want to continue with the cards you found, answer 'yes'. If you want to go back and use a different "
        "search method, answer 'no'."
    )
    MAX_ATTEMPTS = 3
    SAMPLE_SIZE = 5

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        decks_to_search_in: list[AbstractDeck],
        srs: AbstractSRS,
        searcher: AbstractCardSearcher,
    ):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.decks_to_search_in = decks_to_search_in
        self.srs = srs
        self.searcher = searcher
        self.found_cards: list[AbstractCard] = [
            card for deck in self.decks_to_search_in for card in srs.get_cards_in_deck(deck)
        ]

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
                    cards_sample="\n".join(str(it) for it in sample),
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

        raise RuntimeError("Exceeding maximum number of attempts.")


class StateTaskWorkOnFoundCards(AbstractActionState):
    _prompt_template = (
        "You are an assistant of a flashcard management system. You assist a user in executing tasks "
        "(creating/modifying/deleting cards/decks etc.).\n\n"
        "The user gave the following input:\n\n"
        "{user_input}\n\n"
        "You decided to search for cards. Your search returned {amount_cards} cards."
        "Here is a sample of the cards you found:\n\n"
        "{cards_sample}\n\n"
        "Now you have to decide what to do with the cards you found. You have the following options:\n\n"
        "## Bulk operations\n"
        "* You can delete all cards you found from the system. ('delete_all')\n"
        "* Copy_to_deck: You can copy all cards you found to a new deck. You will be asked to specify the name of "
        "the new deck. ('copy_to_deck')\n"
        "\n"
        "## Per-Card operations\n"
        "The following operations will present you every single card and allow you to decide what to do with it.\n"
        "They are more costly than bulk operations, only use them if necessary.\n\n"
        "* You can edit card details (question, answer, other information the card has).\n"
        "* You can delete individual cards.\n"
        "* You can do nothing with some cards.\n"
        "\n"
        "If you want to be presented every single card, answer 'stream_cards'."
        "\n\n\n"
        "Please answer only with the operation you want to perform, and nothing else. Again, the operations are:\n"
        "delete_all, copy_to_deck, stream_cards"
    )
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
                if len(self.found_cards) <= self.SAMPLE_SIZE:
                    sample = self.found_cards
                else:
                    sample = pd.Series(self.found_cards).sample(self.SAMPLE_SIZE).to_list()

                message = self._prompt_template.format(
                    user_input=self.user_prompt,
                    amount_cards=len(self.found_cards),
                    cards_sample="\n".join(str(it) for it in sample),
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
                deck_name = self.llm_communicator.send_message(
                    "What should the name of the new deck be?"
                    "Please answer only with the name of the deck, and nothing else."
                    "Only use letters, numbers, spaces and underscores for the name."
                )

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
                    return StateFinishedTask(f"{len(self.found_cards)} cards copied to deck {deck_name}.")
            if response == "stream_cards":
                return StateStreamFoundCards(
                    self.user_prompt, self.llm, self.decks_to_search_in, self.srs, self.found_cards
                )

        raise RuntimeError("Exceeding maximum number of attempts.")


# TODO: This state assumes that we have a TestSRS!!!!!!!!!!!!!! Fix!
class StateStreamFoundCards(AbstractActionState):
    _prompt_template = (
        "You are an assistant of a flashcard management system. You assist a user in executing tasks "
        "(creating/modifying/deleting cards/decks etc.).\n\n"
        "The user gave the following input:\n\n"
        "{user_input}\n\n"
        "You decided to search for cards. You wanted me to present you every single card you found. "
        "I will now show you the cards one-by-one. You will only ever be able to see a single card. "
        "Here is the current card.:"
        "\n\n{card}\n\n"
        "You have the following options:\n\n"
        '* Edit the question: ["edit_question", "<new question here>"]\n'
        '* Edit the answer: ["edit_answer", "<new answer here">]\n'
        '* Edit the flag: ["edit_flag", "<new flag here">]\n'
        '  These flag options exist: ["none", "red", "orange", "green", "blue", "pink", "turquoise", "purple"]'
        '* Edit the card state: ["edit_card_state", "<new card state here>"]\n'
        '  These card state options exist: ["new", "learning", "review", "suspended", "buried"]'
        '* Delete the card: ["delete_card"]\n'
        '* Do nothing with the card: ["do_nothing"]\n\n'
        "Please answer only with the operation you want to perform in the given list format, and answer nothing else."
    )
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

    def act(self) -> AbstractActionState | None:

        for card in self.found_cards:
            message = self._prompt_template.format(
                user_input=self.user_prompt,
                card=card,
            )
            self.llm_communicator.start_visibility_block()

            for attempt in range(self.MAX_ATTEMPTS_PER_CARD):
                response = self.llm_communicator.send_message(message)

                try:
                    parsed = json.loads(response.strip())
                    if not isinstance(parsed, list):
                        raise ValueError("Response must be a list")
                    if len(parsed) not in [1, 2]:
                        raise ValueError("Response must contain one or two elements.")
                    if not isinstance(parsed[0], str):
                        raise ValueError("First element must be a string.")

                    command = parsed[0]
                    valid_commands = [
                        "edit_question",
                        "edit_answer",
                        "edit_flag",
                        "edit_card_state",
                        "delete_card",
                        "do_nothing",
                    ]
                    if command not in valid_commands:
                        raise ValueError("Response must contain exactly one of the required commands.")

                    if "edit_" in command:
                        if len(parsed) != 2 or not isinstance(parsed[1], str):
                            raise ValueError("Editing requires a second element; it must be a string.")
                        if command == "edit_question":
                            self.srs.edit_card_question(card, parsed[1])
                        elif command == "edit_answer":
                            self.srs.edit_card_answer(card, parsed[1])
                        elif command == "edit_flag":
                            # noinspection PyTypeChecker
                            srs_cast: TestFlashcardManager = self.srs  # TODO unsafe! Wrong! Change!
                            # noinspection PyTypeChecker
                            srs_cast.edit_card_flag(card, Flag.from_str(parsed[1]))
                        elif command == "edit_card_state":
                            # noinspection PyTypeChecker
                            srs_cast: TestFlashcardManager = self.srs  # TODO unsafe! Wrong! Change!
                            # noinspection PyTypeChecker
                            srs_cast.edit_card_state(card, CardState.from_str(parsed[1]))
                        else:
                            raise ValueError(f"Command {command} unknown.")  # unreachable
                    if len(parsed) != 1:
                        raise ValueError("Unexpected second element in response!")
                    if command == "delete_card":
                        self.srs.delete_card(card)
                    if command == "do_nothing":
                        pass
                    break  # one of the valid commands was used, break out of the loop

                except Exception as e:
                    message = f"Your answer must be a valid json string. Exception: {e}. Please try again."
            else:  # only run if no break!
                raise RuntimeError("Exceeding maximum number of attempts.")

        return StateFinishedTask(f"{len(self.found_cards)} cards handled in a stream.")  # TODO (command counts?)


class StateTaskNoSearch(AbstractActionState):

    _prompt_template = """
You are an assistant of a flashcard management system. You execute a task for a user.

The user gave the following task:

{user_input}

The following decks currently exist:

{current_decks}

You now have to call exactly one of the following functions:

* create_deck: {{"task": "create_deck", "name": "<deck name here>"}}
Calling this function will create a new deck with the given name. If the deck already exists, you will receive 
an error and can try again.

* rename_deck: {{"task": "rename_deck", "old_name": "<old deck name here>", "new_name": "<new deck name here>"}}
Calling this function will rename the deck to the given name. If no deck exists with the old name, you will receive an 
error and can try again.

* delete_deck: {{"task": "delete_deck", "name": "<deck name here>"}}
Calling this function will delete the deck with the given name. If no deck exists with the given name, you will receive an 
error and can try again.

* add_card: {{"task": "add_card", "deck_name": "<deck name here>", "question": "<question here>", "answer": "<answer here>", "state": "<card state here>", "flag": "<flag here>"}}
Calling this function will add a new card to the deck with the given name. If no deck exists with the given name, you will receive an error and can try again.
Valid flags are: ['none', 'red', 'orange', 'green', 'blue', 'pink', 'turquoise', 'purple']
Valid card states are: ['new', 'learning', 'review', 'suspended', 'buried']

* exit: {{"task": "exit"}}
Call this function after you finished your task.

* abort: {{"task": "abort", "reason": "<reason here>"}}
Call this function when you have to abort your task.

Please answer only with the filled-in, valid json. You may only send a single command at a time!
""".strip()
    MAX_ATTEMPTS = 10

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.srs = srs
        self.user_prompt = user_prompt

    def act(self) -> AbstractActionState | None:
        # TODO: On second thought: There are only 5 functions left from the legacy code that fit here.
        #     Way, way cleaner to re-implement them here.
        #     On third thought there are only 4 functions left, since I will always call the list_decks() function.

        # # can use legacy classes here, even if adapted.
        # llm_interactor = LLMInteractor(
        #     self.srs,
        #     self.llm, # content search is not used.
        #     LlamaIndexExecutor()
        # )
        # task_executor = TaskExecutor(
        #     llm_interactor=llm_interactor,
        #     llm_communicator=self.llm_communicator,
        # )
        # task_executor.execute_prompts([self.user_prompt])
        #
        # return StateFinishedTask("\n".join(str(it) for it in task_executor.log))
        deck_info = [f"{it.name}: {len(self.srs.get_cards_in_deck(it))}" for it in self.srs.get_all_decks()]
        message = self._prompt_template.format(user_input=self.user_prompt, current_decks="\n".join(deck_info))
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self.llm_communicator.send_message(message)
                parsed = json.loads(response.strip())
                if not isinstance(parsed, dict):
                    raise ValueError("Response must be a dictionary")
                # check that all keys values are strings
                if not all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()):
                    raise ValueError("All keys and values must be strings")
                # check that task is present and one of the expected tasks
                if "task" not in parsed:
                    raise ValueError("Response must contain a task key")

                valid_tasks = ["create_deck", "rename_deck", "delete_deck", "add_card", "exit", "abort"]
                if parsed["task"] not in valid_tasks:
                    raise ValueError("Response must contain a valid task")

                # check that the task is one of the expected tasks
                if parsed["task"] == "create_deck":
                    deck_name = parsed["name"]
                    if not isinstance(deck_name, str):
                        raise ValueError("Deck name must be a string")
                    deck = self.srs.get_deck_by_name_or_none(deck_name)
                    if deck is not None:
                        raise ValueError("Deck already exists")
                    new_deck = self.srs.add_deck(deck_name)
                    message = f"Deck '{deck_name}' created successfully."
                elif parsed["task"] == "rename_deck":
                    old_name = parsed["old_name"]
                    new_name = parsed["new_name"]
                    if not isinstance(old_name, str) or not isinstance(new_name, str):
                        raise ValueError("Names must be strings")
                    deck = self.srs.get_deck_by_name_or_none(old_name)
                    if deck is None:
                        raise ValueError("Deck does not exist")
                    if self.srs.get_deck_by_name_or_none(new_name) is not None:
                        raise ValueError("New name already exists")
                    self.srs.rename_deck(deck, new_name)
                    message = f"Deck {old_name} renamed to {new_name}."
                elif parsed["task"] == "delete_deck":
                    name = parsed["name"]
                    if not isinstance(name, str):
                        raise ValueError("Name must be a string")
                    deck = self.srs.get_deck_by_name_or_none(name)
                    if deck is None:
                        raise ValueError("Deck does not exist")
                    self.srs.delete_deck(deck)
                    message = f"Deck {name} deleted."
                elif parsed["task"] == "add_card":
                    deck_name = parsed["deck_name"]
                    question = parsed["question"]
                    answer = parsed["answer"]
                    state = parsed["state"]
                    flag = parsed["flag"]
                    if not isinstance(deck_name, str):
                        raise ValueError("Name must be a string")
                    if not isinstance(question, str):
                        raise ValueError("Question must be a string")
                    if not isinstance(answer, str):
                        raise ValueError("Answer must be a string")
                    if not isinstance(state, str):
                        raise ValueError("State must be a string")
                    if not isinstance(flag, str):
                        raise ValueError("Flag must be a string")
                    state = CardState.from_str(state)
                    flag = Flag.from_str(flag)
                    deck = self.srs.get_deck_by_name_or_none(deck_name)
                    if deck is None:
                        raise ValueError("Deck does not exist")

                    # TODO evil
                    # noinspection PyTypeChecker
                    test_srs: TestFlashcardManager = self.srs
                    # noinspection PyTypeChecker
                    test_srs.add_full_card(deck, question, answer, flag, state)
                    message = f"Card added to deck {deck_name}."
                elif parsed["task"] == "exit":
                    return StateFinishedTask(f"Task finished.")  # TODO: Command count?
                elif parsed["task"] == "abort":
                    reason = parsed["reason"]
                    if not isinstance(reason, str):
                        raise ValueError("Reason must be a string")
                    return StateFinishedTask(f"Task aborted.")
            except Exception as e:
                message = f"Your answer must be a valid json string. Exception: {e}. Please try again."

        raise RuntimeError("Exceeding maximum number of attempts.")


class StateFinishedTask(AbstractActionState):

    def __init__(self, message: str):
        self.message = message

    def act(self) -> AbstractActionState | None:
        return None
