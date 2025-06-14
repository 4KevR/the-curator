import json
import os
import re
import traceback
from dataclasses import asdict, dataclass

from typeguard import typechecked

from src.backend.modules.ai_assistant.llm_interactor.llm_interactor import LLMInteractor
from src.backend.modules.ai_assistant.task_executor import TaskExecutor
from src.backend.modules.asr.abstract_asr import AbstractASR
from src.backend.modules.evaluation.load_test_data.load_test_data import (
    EvaluationTests,
    InteractionTest,
    QuestionAnsweringTest,
)
from src.backend.modules.evaluation.run_tests.LLMSimilarityJudge import LLMSimilarityJudge
from src.backend.modules.evaluation.run_tests.SRSComparator import SRSComparator
from src.backend.modules.helpers.string_util import remove_block
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.llm.llm_communicator import LLMCommunicator


@dataclass(frozen=True)
class TestEvalResult:
    passed: bool
    crashed: bool
    name: str
    original_queries: list[str]
    transcribed_queries: list[str]
    error_messages: list[str]
    log_messages: list[str]

    def pretty_print(self, skip_thinking=False):
        header = f"Test {self.name} " + ("PASSED" if self.passed else ("CRASHED" if self.crashed else "FAILED")) + "."
        queries = "\n".join(
            f"Original:    {o}\nTranscribed: {t}\n" for (o, t) in zip(self.original_queries, self.transcribed_queries)
        )

        log = []
        for role, message in self.log_messages:
            if skip_thinking:
                message = remove_block(message, "think")
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


class EvaluationPipeline:

    def __init__(
        self,
        asr: AbstractASR,
        llm_interactor: LLMInteractor,
        task_llm: AbstractLLM,
        fuzzy_matching_llm: AbstractLLM,
        llm_judge: AbstractLLM,
        audio_recording_dir_path: str,
        default_max_messages: int,
        default_max_errors: int,
        max_stream_messages_per_chunk: int,
        max_stream_errors_per_chunk: int,
        verbose_task_execution: bool,
        print_progress: bool = False,
        log_file_path: str = None,
    ) -> None:
        self.asr = asr
        self.llm_interactor = llm_interactor
        self.task_llm = task_llm
        self.llm_for_fuzzy_matching = fuzzy_matching_llm
        self.llm_judge = LLMSimilarityJudge(llm_judge)
        self.audio_recording_dir_path = audio_recording_dir_path
        self.default_max_messages = default_max_messages
        self.default_max_errors = default_max_errors
        self.max_stream_messages_per_chunk = max_stream_messages_per_chunk
        self.max_stream_errors_per_chunk = max_stream_errors_per_chunk
        self.verbose_task_execution = verbose_task_execution
        self.print_progress = print_progress
        self.log_file_path = log_file_path

        if log_file_path is not None:
            if os.path.exists(log_file_path):
                raise ValueError(f"Log file '{log_file_path}' already exists.")
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    def _evaluate_test(self, test: InteractionTest | QuestionAnsweringTest) -> TestEvalResult:
        fcm = test.environment.copy()
        llm_communicator = LLMCommunicator(self.task_llm)
        self.llm_interactor.change_flashcard_manager(fcm)

        task_executor = TaskExecutor(
            self.llm_interactor,
            llm_communicator,
            self.default_max_messages,
            self.default_max_errors,
            self.max_stream_messages_per_chunk,
            self.max_stream_errors_per_chunk,
            self.verbose_task_execution,
        )

        evaluation = []
        prompts = []
        try:
            audio_files = [
                os.path.join(self.audio_recording_dir_path, sound_file_name + ".wav")
                for sound_file_name in test.sound_file_names
            ]

            prompts = [self.asr.transcribe_wav_file(afp) for afp in audio_files]

            eval_res = task_executor.execute_prompts(prompts)

            # Now find out if the test passed -> different for q_a or interaction test.
            if isinstance(test, QuestionAnsweringTest):
                if eval_res is None:
                    evaluation = ["LLM did not return an answer to a question."]
                    passed = False
                else:
                    passed = self.llm_judge.judge_answer_similarity(test.expected_answer, eval_res)
                    if not passed:
                        evaluation = [
                            f"LLM judge thought that answer was not close enough.\n"
                            f"Expected: {test.expected_answer}\nActual: {eval_res}"
                        ]
            else:
                comparator = SRSComparator(self.llm_for_fuzzy_matching, self.llm_judge)
                evaluation = comparator.compare_srs(test.expected_result, fcm)
                passed = len(evaluation) == 0

            res = TestEvalResult(
                passed=passed,
                crashed=False,
                error_messages=evaluation,
                original_queries=test.queries,
                transcribed_queries=prompts,
                name=test.name,
                log_messages=task_executor.log,
            )
            return res
        except Exception as e:
            error = f"Exception raised: {e}.\n\nStack trace:\n{traceback.format_exc()}\n"
            return TestEvalResult(
                passed=False,
                crashed=True,
                error_messages=evaluation + [error],
                original_queries=test.queries,
                transcribed_queries=prompts,
                name=test.name,
                log_messages=task_executor.log,
            )

    def _evaluate_tests(self, tests: list[InteractionTest | QuestionAnsweringTest]) -> list[TestEvalResult]:
        """This method mainly exists to make error handling easier."""
        res = []
        try:
            for nr, test in enumerate(tests):
                if self.print_progress:
                    print(
                        f"Total test {nr} out of {len(tests)}"
                        f" ({100.0 * nr / len(tests):.2f}%):"
                        f" {type(test)} {test.name}"
                    )

                res += [self._evaluate_test(test)]

        except Exception as e:
            print(f"Error during evaluation. Still returning partial results.\n{e}\n{traceback.format_exc()}")
            return res

        return res

    def _log_to_file(self, res: list[TestEvalResult]):
        if self.log_file_path is not None:
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                json.dump([asdict(item) for item in res], f, ensure_ascii=False, indent=4)

    @typechecked
    def evaluate(self, tests: EvaluationTests) -> list[TestEvalResult]:
        res = self._evaluate_tests(tests.interaction + tests.question_answering)
        self._log_to_file(res)
        return res

    @typechecked
    def evaluate_individual_tests(self, tests: list[InteractionTest | QuestionAnsweringTest]) -> list[TestEvalResult]:
        res = self._evaluate_tests(tests)
        self._log_to_file(res)
        return res
