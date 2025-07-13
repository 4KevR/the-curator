import re

from src.backend.modules.ai_assistant.states import AbstractActionState
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.llm.llm_communicator import LLMCommunicator
from src.backend.modules.search.llama_index import LlamaIndexExecutor


class StateQuestion(AbstractActionState):
    _prompt_template = (
        "You are an assistant of a flashcard management system. You assist the user in answering questions about the"
        " content of the flashcards in the system. The user asked the following question:\n"
        "{user_input}\n"
        "The following cards in the system fit the question:\n"
        "{cards}\n"
        "**Using only the information in the cards above, answer the question.**\n"
        "If the question cannot be answered using the cards above, respond accordingly."
        "Answer with a single, short sentence, without any additional information."
    )
    _MAX_CARDS_FOR_LLM = 10

    def __init__(self, user_prompt: str, llm: AbstractLLM, llama_index_executor: LlamaIndexExecutor):
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.llama_index_executor = llama_index_executor

    def act(self) -> AbstractActionState | None:
        fitting_nodes = self.llama_index_executor.search_cards(self.user_prompt)
        fitting_nodes = sorted(fitting_nodes, key=lambda x: x[1], reverse=True)
        fitting_nodes = fitting_nodes[: self._MAX_CARDS_FOR_LLM]
        fitting_nodes = "\n".join(
            re.sub(r"\nA: ", " - ", re.sub(r"^Q:", "", fn[0])).replace("\n", " ") for fn in fitting_nodes
        )

        message = self._prompt_template.format(user_input=self.user_prompt, cards=fitting_nodes)
        response = self.llm_communicator.send_message(message)
        return StateAnswer(response)


class StateAnswer(AbstractActionState):
    def __init__(self, answer: str):
        self.answer = answer

    def act(self) -> None:
        return None  # final state
