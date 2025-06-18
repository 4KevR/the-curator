import re
from dataclasses import dataclass

from src.backend.modules.helpers.string_util import remove_block


@dataclass(frozen=True)
class TestEvalResult:
    passed: bool
    crashed: bool
    name: str
    asr_name: str
    task_llm_name: str
    fuzzy_matching_llm_name: str
    llm_judge_name: str
    time_taken_s: float
    audio_files_available: bool
    original_queries: list[str]
    transcribed_queries: list[str] | None
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
Audio files available: {'No' if not self.audio_files_available else 'Yes'}

ASR: {self.asr_name}

Task LLM: {self.task_llm_name}

Fuzzy Matching LLM: {self.fuzzy_matching_llm_name}

LLM Judge: {self.llm_judge_name}

Time taken: {self.time_taken_s:.2f} s.
"""
        if self.transcribed_queries is not None:
            q = "### Queries\n" + "\n\n".join(
                f"**`original   `**: {o} \n\n**`transcribed`**: {t}"
                for (o, t) in zip(self.original_queries, self.transcribed_queries)
            )
        else:
            q = "### Queries\n" + "\n\n".join(f"**`original   `**: {o}" for o in self.original_queries)

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
