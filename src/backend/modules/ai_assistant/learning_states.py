from typing import Callable, Optional

from src.backend.modules.ai_assistant.states import AbstractActionState, ExceedingMaxAttemptsError
from src.backend.modules.helpers.string_util import find_substring_in_llm_response_or_null, remove_block
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.llm.llm_communicator import LLMCommunicator
from src.backend.modules.srs.abstract_srs import AbstractSRS


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
