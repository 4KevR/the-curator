# %%
import json
import os
import re
from dataclasses import asdict, dataclass

from src.backend.modules.ai_assistant.llm_interactor.llm_interactor import LLMInteractor
from src.backend.modules.ai_assistant.task_executor import TaskExecutor
from src.backend.modules.evaluation.load_test_data.load_test_data import InteractionTest
from src.backend.modules.helpers.matching import match_by_key, match_by_equals
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.llm.llm_communicator import LLMCommunicator
from src.backend.modules.srs.testsrs.testsrs import TestFlashcardManager, TestCard


# TODO: Currently only compares TestSRS with TestSRS. Implement comparing/converting AnkiSRS to TestSRS too!


@dataclass(frozen=True)
class TestInfo:
    passed: bool
    crashed: bool
    name: str
    queries: list[str]
    error_messages: list[str]
    log_messages: list[str]

    def pretty_print(self, skip_thinking=False):
        header = f"Test {self.name} " + ("PASSED" if self.passed else ("CRASHED" if self.crashed else "FAILED")) + "."
        queries = "\n".join(" * " + it for it in self.queries)

        log = []
        for role, message in self.log_messages:
            if skip_thinking:
                message = re.sub(r"<think>.*?</think>", "", message, flags=re.DOTALL)
                message = re.sub("\n\n+", "\n", message)
                message = message.strip()

            log.append(
                f"===================================== {role + ' ':>18}=====================================\n{message}\n\n"
            )

        log_msgs = "\n".join(log)
        if len(self.error_messages) == 0:
            errors = "No errors!"
        else:
            errors = "\n".join("\t" + it for it in self.error_messages)

        s = (
            f"##############################################################################################\n"
            f"{header}\n\n"
            f"####################################### Queries ##############################################\n"
            f"{queries}\n\n"
            f"####################################### Logs #################################################\n"
            f"{log_msgs}\n\n"
            f"####################################### Errors ###############################################\n"
            f"{errors}\n\n"
            f"##############################################################################################\n"
        )
        print(s)


class InteractionTestEvaluator:
    def __init__(self, llm_for_fuzzy_matching: AbstractLLM):
        self.llm_for_fuzzy_matching = llm_for_fuzzy_matching

    def _fuzzy_match_test_cards(self, expected_card: TestCard, actual_card: TestCard) -> bool:
        """
        Match two cards by their content using a llm as a judge.
        """
        required = [
            expected_card.cardState == actual_card.cardState,
            expected_card.flag == actual_card.flag,
            expected_card.fuzzymatch_question or expected_card.question == actual_card.question,
            expected_card.fuzzymatch_answer or expected_card.answer == actual_card.answer,
        ]
        if not all(required):
            return False

        prompt = f"""Please evaluate the following two flashcards, and tell me if they have the same content. It is fine if the spelling, the grammar, the length and the wording differs, as long as the cards contain roughly the same information. Punctuation or enclosing quotation marks are irrelevant. If these cards are quite similar, please end your response with "true", else with "false" (without quotation marks). Only the last word of your response will be evaluated.

Card 1:
Question: {expected_card.question}
Answer: {expected_card.answer}

Card 2:
Question: {actual_card.question}
Answer: {actual_card.answer}

Remember to only respond with 'true' or 'false'.
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_for_fuzzy_matching.generate(messages)

        false_index = response.rfind("false")
        true_index = response.rfind("true")

        if false_index != -1 and true_index != -1:
            raise ValueError(f"Unexpected llm response: {response!r}")

        return true_index > false_index

    def _compare_decks(self, expected: list[TestCard], actual: list[TestCard]) -> list[str]:
        """
        Compares two decks. The two decks must be of the same type.
        """
        exp_strict = [x for x in expected if not x.fuzzymatch_question and not x.fuzzymatch_answer]
        exp_fuzzy = [x for x in expected if x.fuzzymatch_question or x.fuzzymatch_answer]

        # match exact
        (_, unm_exp, tmp_unm_act) = match_by_key(
            exp_strict,
            actual,
            equals=lambda x, y: x.to_hashable() == y.to_hashable(),
            left_key=lambda x: x.to_hashable(),
            right_key=lambda x: x.to_hashable(),
        )

        (_, unm_exp_fuzzy, final_unm_act) = match_by_equals(exp_fuzzy, tmp_unm_act, equals=self._fuzzy_match_test_cards)

        # Now create the error messages
        errors = []
        for additional_fuzzy in unm_exp_fuzzy:
            errors += [f"The following expected, fuzzy-matching card has not found a partner:\n{additional_fuzzy}"]

        for additional_expected in unm_exp:
            errors += [f"The following expected card has not found a partner:\n{additional_expected}"]

        for additional_actual in final_unm_act:
            errors += [f"The following provided card was not expected:\n{additional_actual}"]
        return errors

    def _compare_srs(self, expected: TestFlashcardManager, actual: TestFlashcardManager) -> list[str]:
        (matched, unmatched_expected, unmatched_actual) = match_by_key(
            expected.get_all_decks(),
            actual.get_all_decks(),
            equals=(lambda x, y: x.name == y.name),  # checking by name is sufficient, as names must be unique
            left_key=lambda l: l.name,
            right_key=lambda r: r.name,
        )

        errors: list[str] = []
        for unmatched_expected_deck in unmatched_expected:
            errors += [f"The deck {unmatched_expected_deck.name} was expected, but was not in the actual result."]

        for unmatched_actual_deck in unmatched_actual:
            errors += [f"The deck {unmatched_actual_deck.name} was in the actual result, but was unexpected."]

        for e, a in matched:
            errors += self._compare_decks(expected.get_cards_in_deck(e), actual.get_cards_in_deck(a))

        return errors

    def execute_interaction_tests(
        self,
        tests: list[InteractionTest],
        llm_interactor: LLMInteractor,
        task_llm: AbstractLLM,
        default_max_messages: int,
        default_max_errors: int,
        max_stream_messages_per_chunk: int,
        max_stream_errors_per_chunk: int,
        verbose_task_execution: bool,
        print_progress: bool = False,
        log_file_path: str = None,
    ):
        if log_file_path is not None:
            if os.path.exists(log_file_path):
                raise ValueError(f"Log file '{log_file_path}' already exists.")
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        res: list[TestInfo] = []

        try:
            for test_nr, test in enumerate(tests):
                if print_progress:
                    print(f"Test {test_nr} out of {len(tests)} ({100.0 * test_nr / len(tests):.2f}%): {test.name}")

                fcm = test.environment.copy()
                llm_communicator = LLMCommunicator(task_llm)
                llm_interactor.change_flashcard_manager(fcm)
                task_executor = TaskExecutor(
                    llm_interactor,
                    llm_communicator,
                    default_max_messages,
                    default_max_errors,
                    max_stream_messages_per_chunk,
                    max_stream_errors_per_chunk,
                    verbose_task_execution,
                )

                try:
                    task_executor.execute_prompts(test.queries)
                    evaluation = self._compare_srs(test.expected_result, fcm)
                    res.append(
                        TestInfo(
                            passed=len(evaluation) == 0,
                            crashed=False,
                            error_messages=evaluation,
                            queries=test.queries,
                            name=test.name,
                            log_messages=task_executor.log,
                        )
                    )
                except Exception as e:
                    res += [
                        TestInfo(
                            crashed=True,
                            passed=False,
                            queries=test.queries,
                            name=test.name,
                            error_messages=[str(e)],
                            log_messages=task_executor.log,
                        )
                    ]
        except KeyboardInterrupt:
            return res

        if log_file_path is not None:
            with open(log_file_path, "w", encoding="utf-8") as f:
                json.dump([asdict(item) for item in res], f, ensure_ascii=False, indent=4)
        return res
