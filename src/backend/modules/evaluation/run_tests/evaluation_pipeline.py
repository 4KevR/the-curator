import json
import os
import re
import traceback
from dataclasses import asdict, dataclass

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
from src.backend.modules.helpers.string_util import remove_block
from src.backend.modules.llm.abstract_llm import AbstractLLM


@dataclass(frozen=True)
class TestEvalResult:
    passed: bool
    crashed: bool
    name: str
    audio_files_available: bool
    original_queries: list[str]
    transcribed_queries: list[str]
    question_answer: str | None
    task_finish_message: str | None
    state_history: list[str]
    error_messages: list[str]
    log_messages: list[list[tuple[str, str]]]

    def pretty_print(self, skip_thinking=False):
        header = f"Test {self.name} " + ("PASSED" if self.passed else ("CRASHED" if self.crashed else "FAILED")) + "."
        header += f" Audio file available: {self.audio_files_available}"

        queries = "\n".join(
            f"Original:    {o}\nTranscribed: {t}\n" for (o, t) in zip(self.original_queries, self.transcribed_queries)
        )

        if self.question_answer is not None:
            response = f"Question Answer:\n{self.question_answer}\n"
        else:
            if self.task_finish_message is not None:
                response = f"Task Finish Message:\n{self.task_finish_message}\n"
            else:
                response = ""

        log = []
        for group in self.log_messages:
            log.append(
                "=============================================================================================\n"
            )
            for role, message in group:
                if skip_thinking:
                    message = remove_block(message, "think")
                    message = re.sub("\n\n+", "\n", message)
                    message = message.strip()

                log.append(
                    "------------------------------------- {role + ' ':>18}-------------------------------------"
                    f"\n{message}\n\n"
                )

        log_msgs = "\n".join(log)

        history = "\n     -------------------------------------------------------------------------      \n".join(
            self.state_history
        )

        if len(self.error_messages) == 0:
            errors = "No errors!"
        else:
            errors = "\n".join("\t" + it for it in self.error_messages)

        s = (
            f"##############################################################################################\n"
            f"{header}\n\n"
            f"####################################### Queries ##############################################\n"
            f"{queries}\n\n"
            f"####################################### Response #############################################\n"
            f"{response}\n\n"
            f"####################################### History ##############################################\n"
            f"{history}\n\n"
            f"####################################### Logs #################################################\n"
            f"{log_msgs}\n\n"
            f"####################################### Errors ###############################################\n"
            f"{errors}\n\n"
            f"##############################################################################################\n"
        )
        print(s)

    def to_markdown(self, skip_thinking=False) -> str:
        a = f"""
## Test {self.name} {("✅ passed" if self.passed else ("⚡ crashed" if self.crashed else "❌ failed"))}
{'There were no audio files available.' if not self.audio_files_available else 'Audio files were available.'}
"""
        q = "### Queries\n" + "\n\n".join(
            f"**`original   `**: {o} \n\n**`transcribed`**: {t}"
            for (o, t) in zip(self.original_queries, self.transcribed_queries)
        )

        if self.question_answer is not None:
            b = f"### Response\n`Expected:` {self.original_queries[0]}\n`Actual  :` {self.question_answer}\n"
        elif self.task_finish_message is not None:
            b = f"### Task Finish Message\n{self.task_finish_message}\n"
        else:
            b = ""

        log = []
        for group in self.log_messages:
            log.append("\n____________________________\n")
            for role, message in group:
                if skip_thinking:
                    message = remove_block(message, "think")
                    message = re.sub("\n\n+", "\n", message)
                    message = message.strip()

                log.append(f"**{role}:**\n{message}\n\n")

        c = "### Interaction Log\n" + "\n\n".join(log)

        d = "### State History\n" + "\n\n".join(
            f" 1. {str(it).replace('<', '').replace('>', '')}" for it in self.state_history
        )

        if len(self.error_messages) == 0:
            e = "### Errors\nNo errors!"
        else:
            e = "### Errors\n" + "\n_______________\n".join("\t" + it for it in self.error_messages)

        return f"{a}\n{q}\n{b}\n{c}\n{d}\n{e}\n"


class EvaluationPipeline:

    def __init__(
        self,
        asr: AbstractASR,
        task_llm: AbstractLLM,
        fuzzy_matching_llm: AbstractLLM,
        llm_judge: AbstractLLM,
        audio_recording_dir_path: str,
        verbose_task_execution: bool,
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
        fcm = test.environment.copy()

        evaluation = []
        prompts = []
        sm = StateManager(self.task_llm, fcm)

        audio_files = [
            os.path.join(self.audio_recording_dir_path, sound_file_name + ".wav")
            for sound_file_name in test.sound_file_names
        ]

        # do all required audio files exist?
        all_files_exist = all(os.path.exists(audio_file) for audio_file in audio_files)

        try:
            if all_files_exist:
                prompts = [self.asr.transcribe_wav_file(afp) for afp in audio_files]
            else:
                prompts = list(test.queries)

            if len(prompts) != 1:  # TODO remove!!!
                return TestEvalResult(
                    True,
                    False,
                    test.name,
                    all_files_exist,
                    test.queries,
                    prompts,
                    None,
                    None,
                    [],
                    ["SKIPPED"],
                    [[("user", "SKIPPED")]],
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
                name=test.name,
                audio_files_available=all_files_exist,
                error_messages=evaluation,
                original_queries=test.queries,
                transcribed_queries=prompts,
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
                name=test.name,
                audio_files_available=all_files_exist,
                error_messages=evaluation + [error],
                original_queries=test.queries,
                transcribed_queries=prompts,
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
