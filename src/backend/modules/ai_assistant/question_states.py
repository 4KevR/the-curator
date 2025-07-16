import re

from src.backend.modules.ai_assistant.states import AbstractActionState, ExceedingMaxAttemptsError
from src.backend.modules.helpers.string_util import find_substring_in_llm_response_or_null, remove_block
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.llm.llm_communicator import LLMCommunicator
from src.backend.modules.search.llama_index import LlamaIndexExecutor
from src.backend.modules.srs.abstract_srs import AbstractSRS


class StateClassifyQuestion(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. You assist the user in answering questions about the content of the flashcards or about the system itself.

The user asked the following question: "{user_input}"

Please classify the user's input:
- If the question is about the content of the flashcards (e.g., definitions, concepts, facts), respond with "question".
- If the question is about the system itself (e.g. how many decks are available, or number of cards in a specific deck), respond with "system".

Examples:
- How are abstract classes defined in Python? → question
- What decks are available? → system
- How many cards should I study today? → system

**Do not explain. Do not add any other text. Respond with only **one word**: 'question' or 'system'.**
""".strip()
    MAX_ATTEMPTS = 3

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS, llama_index_executor: LlamaIndexExecutor):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.srs = srs
        self.user_prompt = user_prompt
        self.llama_index_executor = llama_index_executor

    def act(self) -> AbstractActionState | None:
        for attempt in range(self.MAX_ATTEMPTS):
            if attempt == 0:
                message = self._prompt_template.format(user_input=self.user_prompt)
            else:
                message = "Your answer must be either 'question' or 'system'."

            response = self.llm_communicator.send_message(message)

            response = remove_block(response, "think").replace('"', "").replace("'", "")
            resp = find_substring_in_llm_response_or_null(response, "question", "system", True)

            if resp is True:
                return StateQuestion(self.user_prompt, self.llm, self.llama_index_executor)
            elif resp is False:
                return StateQuestionAboutSystem(self.user_prompt, self.llm, self.srs)

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
    _MAX_CARDS_FOR_LLM = 10

    def __init__(self, user_prompt: str, llm: AbstractLLM, llama_index_executor: LlamaIndexExecutor):
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.llama_index_executor = llama_index_executor

    def act(self) -> AbstractActionState | None:
        fitting_nodes = self.llama_index_executor.search_cards(self.user_prompt)
        fitting_nodes = sorted(fitting_nodes, key=lambda x: x[1], reverse=True)
        fitting_nodes = fitting_nodes[: self._MAX_CARDS_FOR_LLM]
        fitting_nodes = "\n".join(
            re.sub(r"\nA: ", " - ", re.sub(r"^Q:", "", fn[0])).replace("\n", " ") for fn in fitting_nodes
        )

        message = self._prompt_template.format(user_input=self.user_prompt, cards=fitting_nodes)
        response = self.llm_communicator.send_message(message)
        return StateAnswer(response)


class StateQuestionAboutSystem(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. You assist the user in answering questions about the system itself.

We support four types of system-related queries:
1. Query1: Ask about general deck information (e.g., "How many decks are there?", "What decks exist?")
2. Query2: Ask about the number of cards in a specific deck (e.g., "How many cards are in deck X?")
3. Query3: Ask how many cards to study today overall (e.g., "How many cards should I study today?"). This returns the total number of cards to study across all decks.
4. Query4: Ask how many cards to study today from a specific deck (e.g., "How many cards to study today in deck X?")


Now, the user asked the question: "{user_input}"

The following decks are available in our system:

{decks}

Please classify the user's question:
- If it matches Query1 or Query3, respond with "Query1" or "Query3".
- If it matches Query2 or Query4, respond in the format: "Query2:{{deck_name}}" or "Query4:{{deck_name}}" — extracting the exact name of the matching deck.
- If the input does not match any of the four types, respond with "Unknown".

**Respond with only one of the following formats, do not explain your answer or add any other text:**
- "Query1"
- "Query2:{{deck_name}}"
- "Query3"
- "Query4:{{deck_name}}"
- "Unknown"
""".strip()
    MAX_ATTEMPTS = 5

    def __init__(self, user_prompt: str, llm: AbstractLLM, srs: AbstractSRS):
        self.user_prompt = user_prompt
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.srs = srs

    def act(self) -> AbstractActionState | None:
        decks = self.srs.get_all_decks()
        deck_info = [f'name: "{it.name}", cards: {len(self.srs.get_cards_in_deck(it))}' for it in decks]

        message = self._prompt_template.format(user_input=self.user_prompt, decks="\n".join(deck_info))

        for attempt in range(self.MAX_ATTEMPTS):
            response = self.llm_communicator.send_message(message)
            response = remove_block(response, "think").replace('"', "").strip()  # No replacement for '

            if response == "Query1":
                answer = f"There are {len(decks)} decks in total:\n" + "\n".join(deck_info)
                return StateAnswer(answer)
            elif response.startswith("Query2:"):
                deck_name = response.split(":", 1)[1]
                deck = self.srs.get_deck_by_name_or_none(deck_name)
                if deck:
                    num_of_cards = len(self.srs.get_cards_in_deck(deck))
                    answer = f"There are {num_of_cards} cards in deck '{deck_name}'."
                    return StateAnswer(answer)
                else:
                    message = "No matching deck was found based on your previous response."
            elif response == "Query3":
                count = len(self.srs.cards_to_be_learned_today())
                answer = f"There are {count} cards to study across all decks."
                return StateAnswer(answer)
            elif response.startswith("Query4:"):
                deck_name = response.split(":", 1)[1]
                deck = self.srs.get_deck_by_name_or_none(deck_name)
                if deck:
                    count = len(self.srs.cards_to_be_learned_today(deck))
                    answer = f"There are {count} cards to study in deck '{deck_name}'."
                    return StateAnswer(answer)
                else:
                    message = "No matching deck was found based on your previous response."
            elif response == "Unknown":
                answer = f"Sorry, the query '{self.user_prompt}' is not currently supported."
                return StateAnswer(answer)
            else:
                message = (
                    "The operation failed.\n"
                    'Please answer again in one the following formats: "Query1", "Query2:{{deck_name}}", "Query3", "Query4:{{deck_name}}" or "Unknown".'
                )

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateAnswer(AbstractActionState):
    def __init__(self, answer: str):
        self.answer = answer

    def act(self) -> None:
        return None  # final state
