import ast
import inspect
import re
import traceback
from collections import Counter
from dataclasses import dataclass

from src.backend.modules.ai_assistant.chunked_card_stream import ChunkedCardStream
from src.backend.modules.ai_assistant.llm_interactor.llm_interactor import LLMInteractor
from src.backend.modules.llm.llm_communicator import LLMCommunicator


@dataclass
class _ParsedLLMCommand:
    func_name: str
    args: list
    kwargs: dict


class TaskExecutor:
    """
    See execute_prompts for usage.
    """

    def __init__(
        self,
        llm_interactor: LLMInteractor,
        llm_communicator: LLMCommunicator,
        default_max_errors: int = 5,
        default_max_messages: int = 10,
        max_stream_messages_per_chunk: int = 3,
        max_stream_errors_per_chunk: int = 3,
        verbose: bool = False,
    ):
        """
        Parameters:
            default_max_errors: Maximum number of errors before aborting execution. Can be overwritten in individual calls.
            default_max_messages: Maximum number of messages before aborting execution. Can be overwritten in individual calls.
            max_stream_messages_per_chunk: Maximum number of messages before aborting a card stream.
              The count is reset after each chunk of cards.
            max_stream_errors_per_chunk: Maximum number of errors per chunk before aborting a card stream.
              The count is reset after each chunk of cards.
        """
        self.log = []
        self.llm_interactor = llm_interactor
        self.llm_commands = self.llm_interactor.command_list
        self.llm_communicator = llm_communicator
        self.default_max_errors = default_max_errors
        self.default_max_messages = default_max_messages
        self.max_stream_messages_per_chunk = max_stream_messages_per_chunk
        self.max_stream_errors_per_chunk = max_stream_errors_per_chunk
        self.verbose = verbose
        self.__agent_instructions = None

    def execute_prompts(self, user_prompts: list[str], max_errors: int | None = None, max_messages: int | None = None):
        """
        Sends the given prompts to the LLM and executes the responses using the classes llm_interactor.

        Stops execution if:
          * encountered errors exceed max_errors, or
          * total messages exceed max_messages.
        """
        max_errors = max_errors if max_errors is not None else self.default_max_errors
        max_messages = max_messages if max_messages is not None else self.default_max_messages

        error_count = 0
        message_count = 0

        self.llm_communicator.set_system_prompt(self._get_system_prompt())

        for user_prompt in user_prompts:
            message_to_send = user_prompt

            while True:  # cannot loop forever since message_count is capped by max_messages.
                try:
                    message_count += 1
                    self._add_log_entry("user", message_to_send)
                    answer = self.llm_communicator.send_message(message_to_send)
                    self._add_log_entry("assistant", answer)

                    commands = self._parse_llm_response(answer)
                    results = self._execute_llm_response(commands)

                    if len(results) == 0:
                        return
                    else:
                        if any(isinstance(it, ChunkedCardStream) for it in results):
                            if len(results) == 1:
                                message_to_send = self._handle_card_stream(results[0])
                            else:
                                raise Exception(
                                    "If you want to call a method that returns a stream, you may not call"
                                    " any other function in the same message."
                                )
                        else:
                            message_to_send = self._deep_to_string(results)
                except Exception as e:
                    error_count += 1
                    self._add_log_entry("error", f"Exception raised: {e}.\n\nStack trace:\n{traceback.format_exc()}\n")
                    message_to_send = f"""An error occurred: {e} Please try again!"""
                if error_count > max_errors:
                    self._add_log_entry("error", f"Too many errors. Abort execution.")
                    raise RuntimeError("Too many errors. Abort execution.")
                if message_count > max_messages:
                    self._add_log_entry("error", f"Too many messages. Abort execution.")
                    raise RuntimeError("Too many messages. Abort execution.")

    def _add_log_entry(self, category: str, content: str):
        """Add a log entry to the log, and print it if verbose is True."""
        new_message = (category, content)
        self.log += [new_message]
        if self.verbose:
            print(f"## {category}\n{content}\n")

    def _get_system_prompt(self):
        """Create the system prompt for the LLM. It is only created once and then cached."""
        if self.__agent_instructions is not None:
            return self.__agent_instructions

        cl = self.llm_interactor.command_list
        template = f"""You are an assistant for a flashcard learning system.

## The flashcard system.
The flashcard system contains decks, and each deck is a collection of cards.

About Cards:
{cl.card_type.__doc__.strip("\n")}

About Decks:
{cl.deck_type.__doc__.strip("\n")}

About Temporary Collections:
{cl.temp_collection_type.__doc__.strip("\n")}

## Available Functions for Interaction with the Flashcard System
You can interact with the system by calling specific Python functions, each of which performs an action. The available actions are:
{cl.describe_llm_commands()}

## Execution Details

First, you **have to** think about what the user wants, which information you need, and make a rough plan of your actions. Almost always you will need further information (e.g., deck ids, card ids, or card content). In this case, you will request the information using the functions at your disposal.
However, please **be concise while thinking**. Do not spend much time.
**Only** execute what the user asks you to. Do not perform further tasks. If you are tasked to create a deck, you create a deck, and not anything else like additional cards.
After reasoning, output the steps you want to execute **now** in the following format:

<execute>
* function_call_1(arguments)
* function_call_2(arguments)
...
</execute>

The system will then execute your commands, and return an python array of results:
["result_of_call_1", "result_of_call_2", ...]

If no further actions are needed, please return an empty execute block:

<execute>
</execute>

## Further Instructions
There are a few special cases:
* If the prompt does not specify which deck to operate on, please check (by listing the decks) if only one deck exists. In this case, please use this deck. In this case, do not generate a default deck.
* Some prompts may contain questions and answers in quotation marks, some prompts may not. Please remove those quotation marks. E.g. if the user wants you to:
 Create a card with question "What is the most common pet in Germany?" and answer "Dogs".
 You should **not** include these quotation marks in the card.
* If the user asks you to find cards about a topic, use the right function. If you have a given or obvious keyword, use functions like search_for_substring. If the user asks about an concept, please use methods like search_for_content instead, that actually evaluate the content of the query and the cards.
* If you performed a substring search and found no cards, please always try a fuzzy search! So single-letter mistakes or punctuation do not prevent you from finding the correct card.
* If your job is to edit/delete all cards that are related to a specific topic / have a certain keyword, your steps are:
  * 1. Create a temporary collection of the relevant cards using search_by_substring or search_by_content. Fuzzy-search is your friend; use it!
  * 2. List the cards in the temporary collection to get a card stream.
  * 3. You will be provided with chunks of cards; now it is your job to work with these chunks: Delete, edit, etc. the card according to your task.
  
If you are not sure what to do, and you are sure that the user forgot to specify some specifics, please call the above-mentioned function to request further information from the user.

## An Example
For example, if the user prompt was:
"Create a new deck with the name Astrology and add What is the largest planet in our solar system? and Jupiter to it. Flag it as Purple."
Your steps should be:
* Create a new deck with the name Astrology. Wait for the output to get the id of this new deck.
* Add a card with the given question, answer and flag to the deck.

So the first execution plan would be:
<execute>
* create_deck("Astrology")
</execute>

The system then would answer you the information of the newly created deck, e.g.:
["Deck 'Astrology' (id: deck_9874_2787)"]

Then, the next execution plan would be:
<execute>
* add_card(deck_id_str="deck_9874_2787", question="What is the largest planet in our solar system?", answer="Jupiter", state="new", flag="purple")
</execute>

The system then provides you with an empty response.
Then, you have achieved your task, and return:

<execute>
</execute>
"""

        self.__agent_instructions = template
        return self.__agent_instructions

    def _handle_card_stream(self, chunked_cards: ChunkedCardStream) -> str:
        """Handle a card stream, passing the chunked cards to the LLM."""

        stream_info = """You are currently in a card stream. You will be provided with groups of cards. **Now** you can work with these cards; you can use the card ids to call edit, delete, add operations.
Once you are done with the current chunk, and you want to continue to the next chunk, please return an empty <execute>...</execute> block.
You will **not** be able to see the previous chunk and the messages you sent in the previous chunks.
To end the stream early (before all cards are processed), please call the function "abort_card_stream(reason: str)". Only call this if there is an error, as you usually have to see all cards in the stream!!
        """
        self.llm_communicator.add_message(stream_info)
        self.llm_communicator.start_visibility_block()

        next_chunk = chunked_cards.next_chunk()
        message_to_send = "The next messages are:\n" + "\n\n".join(str(it) for it in next_chunk)

        error_count = 0
        message_count = 0
        command_counter = Counter()

        while True:
            try:
                message_count += 1
                self._add_log_entry("user-stream", message_to_send)
                answer = self.llm_communicator.send_message(message_to_send)
                self._add_log_entry("assistant-stream", answer)

                commands = TaskExecutor._parse_llm_response(answer)
                if any(c.func_name == "abort_card_stream" for c in commands):
                    if len(commands) == 1:
                        self.llm_communicator.end_visibility_block()
                        args_str = ", ".join(commands[0].args)
                        kw_args = str(commands[0].kwargs) if len(commands[0].kwargs) > 0 else ""
                        return f"You decided to exit the stream early for the following reason: {args_str} {kw_args}"
                    raise Exception(
                        "If you want to exit the card stream, you may not call any other function in the same message."
                        " **None** of your commands from the last message have been executed."
                    )

                if any(self._llm_function_return_type(c.func_name) == ChunkedCardStream for c in commands):
                    raise Exception("You are already in a card stream. Exit this stream before entering a new one.")

                command_counter.update(it.func_name for it in commands)

                if len(commands) > 0:
                    results = self._execute_llm_response(commands)
                    message_to_send = self._deep_to_string(results)
                else:
                    self.llm_communicator.end_visibility_block()

                    if not chunked_cards.has_next():
                        self.llm_communicator.end_visibility_block()
                        func_call_times = "\n".join(
                            f"{it[0]:<26}: {it[1]:>5}"
                            for it in sorted(command_counter.items(), key=lambda x: x[1], reverse=True)
                        )
                        return (
                            f"The stream containing {len(chunked_cards.items)} cards has been fully processed."
                            f" You have left the card stream, and can **not** call abort_card_stream(message) any "
                            f"more. **YOU ACHIEVED ALL YOUR TASKS THAT YOU WANTED TO DO WITH THE STREAM**. In 99 %"
                            f" of all cases, you are done now and can just send an empty <execute></execute> block"
                            f" to finish this session.\n\n"
                            f"You called the following functions (with frequency):\n{func_call_times}"
                        )

                    next_chunk = chunked_cards.next_chunk()
                    error_count, message_count = 0, 0
                    self.llm_communicator.start_visibility_block()
                    message_to_send = "The next messages are:\n" + "\n\n".join(str(it) for it in next_chunk)
            except Exception as e:
                error_count += 1
                self._add_log_entry("exception", f"Exception raised: {e}.\n\nStack trace:\n{traceback.format_exc()}\n")
                message_to_send = (
                    f"Exception raised: {e}. **The card stream is still active.** Remember to call the function"
                    f" 'abort_card_stream()' to abort the card stream prematurely if really necessary."
                )
            if error_count > self.max_stream_errors_per_chunk:
                self._add_log_entry("error", f"Too many errors. Abort execution.")
                raise RuntimeError("Too many errors. Abort execution.")
            if message_count > self.max_stream_messages_per_chunk:
                self._add_log_entry("error", f"Too many messages. Abort execution.")
                raise RuntimeError("Too many messages. Abort execution.")

    def _llm_function_return_type(self, function_name: str):
        """Get the return type of an @llm_command"""
        function = self.llm_commands.llm_commands.get(function_name, None)
        if function is None:
            raise ValueError(f"Function {function_name} not known.")

        sig = inspect.signature(function)
        return type(None) if sig.return_annotation is None else sig.return_annotation

    def _execute_llm_response(self, commands: list[_ParsedLLMCommand]):
        """Execute the given parsed commands and return the results as a list."""
        for command in commands:
            if command.func_name not in self.llm_commands.llm_commands:
                raise ValueError(f"Unknown function name {command.func_name}.")

        return [
            self.llm_commands.llm_commands[command.func_name](self.llm_interactor, *command.args, **command.kwargs)
            for command in commands
        ]

    @staticmethod
    def _deep_to_string(obj):
        """Convert an object to a string recursively, handling lists, tuples, sets, and dicts."""
        if obj is None:
            return "None"
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, (int, float)):
            return str(obj)
        elif isinstance(obj, dict):
            items = []
            for key, value in obj.items():
                items.append(f"{TaskExecutor._deep_to_string(key)}: {TaskExecutor._deep_to_string(value)}")
            return "{" + ", ".join(items) + "}"
        elif isinstance(obj, (list, tuple)):
            elements = [TaskExecutor._deep_to_string(e) for e in obj]
            if isinstance(obj, list):
                return "[" + ", ".join(elements) + "]"
            else:
                return "(" + ", ".join(elements) + ")"
        elif isinstance(obj, set):
            elements = [TaskExecutor._deep_to_string(e) for e in obj]
            elements.sort()
            return "{" + ", ".join(elements) + "}"
        else:
            return str(obj)

    @staticmethod
    def _parse_function_call(call_str) -> _ParsedLLMCommand:
        """Parse a string containing a function call into a _ParsedLLMCommand."""
        # Parse the string into an AST node
        try:
            tree = ast.parse(call_str, mode="eval")
        except SyntaxError:
            raise ValueError(f"The string\n\n{call_str}\n\nis not a ast-parsable Python expression.")

        # Ensure it's a function call
        if not isinstance(tree.body, ast.Call):
            raise ValueError(f"The string\n\n{call_str}\n\n is not a function call.")

        call_node = tree.body

        # Get function name as string
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            # Handles cases like module.function()
            func_name = ast.unparse(call_node.func)
        else:
            raise ValueError(f"Unsupported function name format in {call_str}.")

        # Evaluate positional arguments safely
        args = [ast.literal_eval(arg) for arg in call_node.args]

        # Evaluate keyword arguments safely
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call_node.keywords}

        return _ParsedLLMCommand(func_name, args, kwargs)

    @staticmethod
    def _parse_llm_response(response: str) -> list[_ParsedLLMCommand]:
        """Parses the 'execute' block in the LLM response into a list of _ParsedLLMCommand objects."""
        match = re.search(r"^ *<execute>(.*?)</execute>", response, re.DOTALL + re.MULTILINE)
        if not match:
            raise ValueError(
                "No execute block found in response. Remember to use <execute>...</execute> to mark your execution"
                " plan, and send an empty block to indicate that you do not wish to take any further action."
            )
        plan = match.group(1)
        return [TaskExecutor._parse_function_call(line[1:].strip()) for line in plan.splitlines() if line.strip()]
