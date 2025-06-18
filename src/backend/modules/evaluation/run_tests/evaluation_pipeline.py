import json
import os
import time
import traceback
from dataclasses import asdict

from typeguard import typechecked

from src.backend.modules.ai_assistant.state_manager import StateManager
from src.backend.modules.asr.abstract_asr import AbstractASR
from src.backend.modules.evaluation.load_test_data.load_test_data import (
    EvaluationTests,
    InteractionTest,
    QuestionAnsweringTest,
)
from src.backend.modules.evaluation.run_tests.llm_similarity_judge import LLMSimilarityJudge
from src.backend.modules.evaluation.run_tests.srs_comparator import SRSComparator
from src.backend.modules.evaluation.run_tests.test_eval_result import TestEvalResult
from src.backend.modules.llm.abstract_llm import AbstractLLM


class EvaluationPipeline:

    def __init__(
        self,
        asr: AbstractASR,
        task_llm: AbstractLLM,
        fuzzy_matching_llm: AbstractLLM,
        llm_judge: AbstractLLM,
        audio_recording_dir_path: str | None = None,
        verbose_task_execution: bool = False,
        print_progress: bool = False,
        log_file_path: str = None,
    ) -> None:
        self.asr = asr
        self.task_llm = task_llm
        self.llm_for_fuzzy_matching = fuzzy_matching_llm
        self.llm_judge = LLMSimilarityJudge(llm_judge)
        self.audio_recording_dir_path = audio_recording_dir_path
        self.verbose_task_execution = verbose_task_execution
        self.print_progress = print_progress
        self.log_file_path = log_file_path

        if log_file_path is not None:
            if os.path.exists(log_file_path):
                raise ValueError(f"Log file '{log_file_path}' already exists.")
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    def _evaluate_test(self, test: InteractionTest | QuestionAnsweringTest) -> TestEvalResult:
        start_time = time.time()

        fcm = test.environment.copy()

        evaluation = []
        prompts = []
        sm = StateManager(self.task_llm, fcm)

        if self.audio_recording_dir_path is not None:
            audio_files = [
                os.path.join(self.audio_recording_dir_path, sound_file_name + ".wav")
                for sound_file_name in test.sound_file_names
            ]

            # do all required audio files exist?
            all_files_exist = all(os.path.exists(a_f) for a_f in audio_files)
        else:
            audio_files = []
            all_files_exist = False

        try:
            if all_files_exist:
                prompts = [self.asr.transcribe_wav_file(afp) for afp in audio_files]
            else:
                prompts = list(test.queries)

            if len(prompts) != 1:  # TODO remove!!!
                return TestEvalResult(
                    passed=True,
                    crashed=False,
                    asr_name=self.asr.get_description(),
                    task_llm_name=self.task_llm.get_description(),
                    fuzzy_matching_llm_name=self.llm_for_fuzzy_matching.get_description(),
                    llm_judge_name=self.llm_judge.judge_llm.get_description(),
                    time_taken_s=time.time() - start_time,
                    name=test.name,
                    audio_files_available=all_files_exist,
                    error_messages=test.queries,
                    original_queries=prompts,
                    transcribed_queries=None,
                    question_answer=None,
                    task_finish_message=None,
                    state_history=["SKIPPED"],
                    log_messages=[[("user", "SKIPPED")]],
                )

            eval_res = sm.run(prompts[0], self.verbose_task_execution)

            # Now find out if the test passed -> different for q_a or interaction test.
            if isinstance(test, QuestionAnsweringTest):
                if eval_res.question_answer is None:
                    evaluation = ["LLM did not return an answer to a question."]
                    passed = False
                else:
                    passed = self.llm_judge.judge_answer_similarity(test.expected_answer, eval_res.question_answer)
                    if not passed:
                        evaluation = [
                            f"LLM judge thought that answer was not close enough.\n"
                            f"Expected: {test.expected_answer}\nActual: {eval_res.question_answer}"
                        ]
            else:
                comparator = SRSComparator(self.llm_for_fuzzy_matching, self.llm_judge)
                evaluation = comparator.compare_srs(test.expected_result, fcm)
                passed = len(evaluation) == 0

            res = TestEvalResult(
                passed=passed,
                crashed=False,
                asr_name=self.asr.get_description(),
                task_llm_name=self.task_llm.get_description(),
                fuzzy_matching_llm_name=self.llm_for_fuzzy_matching.get_description(),
                llm_judge_name=self.llm_judge.judge_llm.get_description(),
                time_taken_s=time.time() - start_time,
                name=test.name,
                audio_files_available=all_files_exist,
                error_messages=evaluation,
                original_queries=test.queries,
                transcribed_queries=prompts if all_files_exist else None,
                question_answer=eval_res.question_answer,
                task_finish_message=eval_res.task_finish_message,
                state_history=eval_res.state_history,
                log_messages=eval_res.llm_history,
            )
            return res
        except Exception as e:
            error = f"Exception raised: {e}.\n\nStack trace:\n{traceback.format_exc()}\n"
            return TestEvalResult(
                passed=False,
                crashed=True,
                asr_name=self.asr.get_description(),
                task_llm_name=self.task_llm.get_description(),
                fuzzy_matching_llm_name=self.llm_for_fuzzy_matching.get_description(),
                llm_judge_name=self.llm_judge.judge_llm.get_description(),
                time_taken_s=time.time() - start_time,
                name=test.name,
                audio_files_available=all_files_exist,
                error_messages=evaluation + [error],
                original_queries=test.queries,
                transcribed_queries=prompts if all_files_exist else None,
                question_answer=None,
                task_finish_message=None,
                state_history=sm.state_history,
                log_messages=sm.logging_llm.get_log(),
            )

    def _evaluate_tests(self, tests: list[InteractionTest | QuestionAnsweringTest]) -> list[TestEvalResult]:
        """This method mainly exists to make error handling easier."""
        res = []
        try:
            last_print_len = 0
            for nr, test in enumerate(tests):
                if self.print_progress:
                    s = (
                        f"\rTotal test {nr} out of {len(tests)}"
                        f" ({100.0 * nr / len(tests):.2f}%):"
                        f" {test.__class__.__name__} {test.name}"
                    )
                    print(s + (last_print_len - len(s)) * " ", end="")
                    last_print_len = len(s)

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
