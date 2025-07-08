from typing import Callable, Optional

from src.backend.modules.ai_assistant.states import AbstractActionState, ExceedingMaxAttemptsError
from src.backend.modules.helpers.string_util import find_substring_in_llm_response_or_null, remove_block
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.llm.llm_communicator import LLMCommunicator
from src.backend.modules.srs.abstract_srs import AbstractSRS, MemoryGrade


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

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        srs: AbstractSRS,
        progress_callback: Callable[[str, Optional[bool]], None] | None = None,
    ):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.srs = srs
        self.user_prompt = user_prompt
        self.progress_callback = progress_callback

    def act(self) -> AbstractActionState | None:
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
                        self.srs.init_learning_state(deck, cards)
                        first_card_question = self.srs.get_current_learning_card().question

                        if self.progress_callback:
                            self.progress_callback(
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

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        srs: AbstractSRS,
        progress_callback: Callable[[str, Optional[bool]], None] | None = None,
    ):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs
        self.progress_callback = progress_callback

    def act(self) -> AbstractActionState | None:
        card_question = self.srs.get_current_learning_card().question
        message = self._prompt_template.format(user_input=self.user_prompt, card_question=card_question)

        for attempt in range(self.MAX_ATTEMPTS):
            response = self.llm_communicator.send_message(message)
            response = remove_block(response, "think").replace('"', "").replace("'", "").strip()
            resp = find_substring_in_llm_response_or_null(response, "answer", "end", True)

            if resp is True:
                return StateJudgeAnswer(self.user_prompt, self.llm, self.srs, False, self.progress_callback)
            elif resp is False:
                self.srs.study_mode = False
                if self.progress_callback:
                    self.progress_callback("Exit study mode.", True)
                return StateFinishedLearn("Exit study mode.")
            elif resp is None:
                if "both" in response.lower():
                    return StateExtractAnswer(self.user_prompt, self.llm, self.srs, self.progress_callback)

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

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        srs: AbstractSRS,
        progress_callback: Callable[[str, Optional[bool]], None] | None = None,
    ):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs
        self.progress_callback = progress_callback

    def act(self) -> AbstractActionState | None:
        message = self._prompt_template.format(user_input=self.user_prompt)

        for attempt in range(self.MAX_ATTEMPTS):
            response = self.llm_communicator.send_message(message)
            response = remove_block(response, "think").strip()
            if response != "":  # not always stable
                self.user_prompt = response
                return StateJudgeAnswer(self.user_prompt, self.llm, self.srs, True, self.progress_callback)

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateJudgeAnswer(AbstractActionState):
    _prompt_template = """
You are the answer evaluator for a flashcard system. You help users objectively and impartially evaluate the accuracy of their answers to flashcard questions.

Your task is to evaluate the user's answer based primarily on the correct answer above, considering the following principles:
0. **Capitalization is not a criterion. Minor spelling or grammatical mistakes can be ignored as long as the meaning is clearly conveyed.**
1. **Trivial-repetition check**: If the user's answer exactly matches the pattern “<term> is <term>” (case-insensitive, with optional period) without any explanatory content, immediately return 'again'.
2. **No relation**: If the user's answer has no semantic relation to the correct answer (completely off-topic or nonsense), you must return 'again'.
3. **Incompleteness**: Missing part of the answer, missing key points of the answer, or the answer is vague, return 'hard'.
4. **Uncertainty**: If the user actively expresses uncertainty about the answer, return 'hard'.
5. **Partial understanding**: If the user's answer is partial semantically consistent with the correct answer, return 'good'.
6. **Mostly correct**: If the user's answer conveys essentially the full meaning of the correct answer with only minor wording differences, return 'easy'.

Please return only one of the following evaluations:
- 'again': the user clearly did not remember the answer and should try again.
- 'hard': the user struggled or was mostly incorrect, but showed partial understanding.
- 'good': the user remembered the answer reasonably well with minor issues.
- 'easy': the user recalled the answer very easily and accurately.

Here are some examples:
Example 1:
- Question: What is UFO?
- Correct Answer: Unidentified Flying Object
- User Answer: UFO is UFO
→ again (Trivial-repetition check failed)

Example 2:
- Question: What is UFO?
- Correct Answer: Unidentified Flying Object
- User Answer: ufo is unidentified flying object
→ easy (Capitalization is not a criterion, the meaning is clearly conveyed)

Example 3:
- Question: What is UFO?
- Correct Answer: Unidentified Flying Object
- User Answer: ufo is Unidentified Flying
→ hard (Missing part of the answer "Object")

Example 4:
- Question: What is encapsulation?
- Correct Answer: Encapsulation is the bundling of data and methods with controlled access, enforced via access modifiers like private and public.
- User Answer: encapsulation is the bundling of data and methods with controlled access, enforced via access modifiers like private and public.
→ easy (answer is semantically and lexically identical to the correct answer (apart from minor punctuation or case))

Example 5:
- Question: What is encapsulation?
- Correct Answer: Encapsulation is the bundling of data and methods with controlled access, enforced via access modifiers like private and public.
- User Answer: It's when you put stuff inside a capsule.
→ again (No relation)

Example 6:
- Question: What is a hash table?
- Correct Answer: A hash table is a data structure that uses a hash function to map keys to indices in an array of buckets, providing average-case O(1) lookup, insertion, and deletion.
- User Answer: A hash table stores key-value pairs in an array and applies a hash function to compute an index, achieving average constant-time operations.
→ good (Partial understanding)

Example 7:
- Question: Define polymorphism.
- Correct Answer: Polymorphism allows objects of different classes to be treated through a common interface, with each class providing its own implementation of the shared methods.
- User Answer: It means many forms, but I'm not sure how it actually works in code.
→ hard (The user clearly expressed uncertainty)


Please rate the user's answer to the current flashcard:

The current flashcard is as follows:
- Question: {card_question}
- Correct Answer: {card_answer}

The user gave the following answers to the questions on the card:
{user_answer}

**Return only one word of: 'again', 'hard', 'good', or 'easy'. Do not return anything else.**
""".strip()

    MAX_ATTEMPTS = 5

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        srs: AbstractSRS,
        end: bool,
        progress_callback: Callable[[str, Optional[bool]], None] | None = None,
    ):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs
        self.end = end
        self.progress_callback = progress_callback

    def act(self) -> AbstractActionState | None:
        card = self.srs.get_current_learning_card()
        message = self._prompt_template.format(
            user_answer=self.user_prompt, card_question=card.question, card_answer=card.answer
        )

        for attempt in range(self.MAX_ATTEMPTS):
            response = self.llm_communicator.send_message(message)
            response = remove_block(response, "think").replace('"', "").replace("'", "").replace(".", "").strip()
            try:
                self.srs.set_memory_grade(card, MemoryGrade.from_str(response))

                msg_to_user = (
                    f"Your answer to the previous card was rated as: {response}.\n" f"Correct Answer: {card.answer}\n"
                )

                if response == "again" or response == "hard":
                    self.srs.repeat_learning_card()
                elif response == "good":
                    self.srs.repeat_learning_card(once=True)

                next_card = self.srs.get_next_learning_card()
                if next_card is None:
                    msg_to_user += "Congratulations! You have finished this deck for now."
                    self.srs.study_mode = False
                    if self.progress_callback:
                        self.progress_callback("Exit study mode.", True)
                    return StateFinishedLearn(msg_to_user)
                elif self.end:
                    msg_to_user += "Exit study mode."
                    self.srs.study_mode = False
                    if self.progress_callback:
                        self.progress_callback("Exit study mode.", True)
                    return StateFinishedLearn(msg_to_user)
                else:
                    msg_to_user += f"Question: {next_card.question}"
                    return StateFinishedLearn(msg_to_user)
            except:  # noqa: E722
                pass

            message = "Return only one of: 'again', 'hard', 'good', or 'easy'. Do not return anything else."

        raise ExceedingMaxAttemptsError(self.__class__.__name__)


class StateFinishedLearn(AbstractActionState):

    def __init__(self, message: str):
        self.message = message

    def act(self) -> AbstractActionState | None:
        return None
