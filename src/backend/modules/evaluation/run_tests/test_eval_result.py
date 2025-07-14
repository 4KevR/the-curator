import re
from dataclasses import dataclass

from src.backend.modules.helpers.string_util import remove_block
from src.backend.modules.llm.types import TokenUsage


@dataclass(frozen=True)
class TestEvalResult:
    passed: bool
    crashed: bool
    name: str
    asr_name: str
    task_llm_name: str
    fuzzy_matching_llm_name: str
    llm_judge_name: str
    max_levenshtein_distance: int | None
    max_levenshtein_factor: float | None
    time_taken_s: float
    audio_files_available: bool
    original_queries: list[str]
    transcribed_queries: list[str] | None
    question_answer: str | None
    expected_answer: str | None
    task_finish_message: str | None
    state_history: list[str]
    error_messages: list[str]
    log_messages: list[list[tuple[str, str]]]
    token_usage: TokenUsage | None

    def pretty_print(self, skip_thinking=False):
        header = f"Test {self.name} " + ("PASSED" if self.passed else ("CRASHED" if self.crashed else "FAILED")) + "."
        header += f" Audio file available: {self.audio_files_available}."
        header += f" Time taken: {self.time_taken_s:.2f} s."
        header += f" Token usage: {self.token_usage.prompt_tokens} prompt, {self.token_usage.completion_tokens} completion, {self.token_usage.total_tokens} total."

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
        if self.max_levenshtein_distance is None and self.max_levenshtein_factor is None:
            levenshtein = "No, exact matching."
        else:
            levenshtein = f"distance: {self.max_levenshtein_distance}, factor: {self.max_levenshtein_factor:.3f}"

        header = f"""
## Test {self.name} {("✅ passed" if self.passed else ("⚡ crashed" if self.crashed else "❌ failed"))}
Audio files available: {'No' if not self.audio_files_available else 'Yes'}

ASR: {self.asr_name}

Task LLM: {self.task_llm_name}

Fuzzy Matching LLM: {self.fuzzy_matching_llm_name}

LLM Judge: {self.llm_judge_name}

Time taken: {self.time_taken_s:.2f} s.

Levenshtein matching: {levenshtein}

Token usage: {self.token_usage.prompt_tokens} prompt, {self.token_usage.completion_tokens} completion, {self.token_usage.total_tokens} total.
"""
        if self.transcribed_queries is not None:
            queries = "### Queries\n" + "\n\n".join(
                f"**`original   `**: {o} \n\n**`transcribed`**: {t}"
                for (o, t) in zip(self.original_queries, self.transcribed_queries)
            )
        else:
            queries = "### Queries\n" + "\n\n".join(f"**`original   `**: {o}" for o in self.original_queries)

        if self.question_answer is not None:
            response = f"### Response\nActual: {self.question_answer}\n\nExpected: {self.expected_answer}\n"
        elif self.task_finish_message is not None:
            response = f"### Task Finish Message\n{self.task_finish_message}\n"
        else:
            response = ""

        log = []
        for group in self.log_messages:
            log.append("\n____________________________\n")
            for role, message in group:
                if skip_thinking:
                    message = remove_block(message, "think")
                    message = re.sub("\n\n+", "\n", message)
                    message = message.strip()

                log.append(f"**{role}:**\n{message}\n\n")

        interaction_log = "### Interaction Log\n" + "\n\n".join(log)

        history = "### State History\n" + "\n\n".join(
            f" 1. {str(it).replace('<', '').replace('>', '')}" for it in self.state_history
        )

        if len(self.error_messages) == 0:
            errors = "### Errors\nNo errors!"
        else:
            errors = "### Errors\n" + "\n_______________\n".join("\t" + it for it in self.error_messages)

        return f"{header}\n{queries}\n{response}\n{errors}\n{history}\n{interaction_log}\n"
