from src.backend.modules.llm.abstract_llm import AbstractLLM


class LoggingLLM(AbstractLLM):
    def __init__(self, llm: AbstractLLM):
        self._llm = llm
        self._log: list[list[tuple[str, str]]] = []
        self._last_messages: list[tuple[str, str]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        # llm proxy
        response = self._llm.generate(messages, temperature, max_tokens)

        messages_w_r = messages + [{"role": "assistant", "content": response}]

        # logging
        if (
            len(self._log) != 0
            and len(self._last_messages) <= len(messages)
            and self._last_messages == messages[0 : len(self._last_messages)]
        ):
            # add to last group
            self._log[-1] += [(m["role"], m["content"]) for m in messages_w_r[len(self._last_messages) :]]
        else:
            # create new group
            self._log.append([(m["role"], m["content"]) for m in messages_w_r])

        self._last_messages = messages_w_r
        return response

    def get_description(self) -> str:
        return "Logging " + self._llm.get_description()

    def get_log(self):
        return self._log
