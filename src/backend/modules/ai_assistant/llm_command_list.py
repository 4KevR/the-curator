import inspect
from typing import Callable

from src.backend.modules.helpers.string_util import replace_many


class LLMCommandList:
    """
    A list of LLM commands. It is filled using the @llm_command decorator.
    The commands can then be described using the describe_llm_commands method for tool
    use in a llm.
    """
    llm_commands: dict[str, Callable]

    def __init__(self, substitutions: dict[str, str]):
        """
        Params:
         - substitutions: A dictionary of substitutions to apply to the command descriptions. Can be used to replace
                          class names, e.g. TestDeck -> Deck.
        """
        self.llm_commands = {}
        self.substitutions = substitutions

    def add_command(self, func: Callable):
        """Adds a function to the list. If a command with the same name already exists, a ValueError is raised."""
        if func.__name__ in self.llm_commands:
            raise ValueError(f"Command {func.__name__} already exists.")

        self.llm_commands[func.__name__] = func

    @staticmethod
    def annotation_to_string(annotation) -> str:
        """Converts a type annotation to a string. Also handles generic types."""
        if annotation is None: return "None"

        # generic?
        if not hasattr(annotation, "__origin__"):
            return getattr(annotation, '__name__', annotation)

        origin = annotation.__origin__
        args = annotation.__args__
        type_str = f"{origin.__name__}[{', '.join(arg.__name__ for arg in args)}]"
        return type_str

    def describe_llm_commands(self) -> str:
        """Returns a string describing all commands in the list, including their signatures and docstrings."""
        res = []
        for cmnd_name, llm_command in self.llm_commands.items():
            params = []
            sig = inspect.signature(llm_command)
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                params += [f"{name}: {param.annotation.__name__}"]

            if sig.return_annotation is None:
                return_type = "None"
            else:
                return_type = sig.return_annotation

            signature = f"{cmnd_name}({", ".join(params)}) -> {self.annotation_to_string(return_type)}"
            signature = signature.replace("_empty", "<unspecified>")
            docs = llm_command.__doc__.strip("\n")
            res += [f"{signature}\n{docs}"]

        s = "\n\n".join(res)
        s = s.replace("__main__.", "")  # remove unnecessary main references
        s = replace_many(s, self.substitutions)
        return s


def llm_command(llm_command_list):
    """A decorator factory that adds a function to the given LLMCommandList."""

    def decorator(func):
        llm_command_list.add_command(func)
        return func

    return decorator
