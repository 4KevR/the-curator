from src.backend.modules.llm.abstract_llm import AbstractLLM


class LoggingLLM(AbstractLLM):
    def __init__(self, llm: AbstractLLM):
        self._llm = llm
        # self.known_messages = {}
        self._log: list[list[tuple[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        # recognize duplicate messages:
        form_messages = [(message["role"], message["content"]) for message in messages]

        # new_messages = [((msg[0], "known") if msg in self.known_messages else msg) for msg in form_messages]
        # if len(new_messages) == len(form_messages):
        #     self.known_messages = set()
        # else:
        #     self.known_messages.update(new_messages)

        response = self._llm.generate(messages)

        form_messages += [("assistant", response)]
        self._log.append(form_messages)

        return response

    def get_log(self):
        return self._log
