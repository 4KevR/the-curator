# =============================================================
# On line 110 of this file there is a "FlashcardManager"
# Now it should be a class of Anki or a subclass of Anki.
# It depends on how you want to modify
# =============================================================
# =============================================================

import re
import traceback
import ast
import inspect
from dataclasses import dataclass
from typing import Optional

from .llm_cmd_registration import llm_commands
from .llm_controller_for_anki import LLMInteractor, LLMCommunicator, ChunkedCardStream

class TaskExecutor:
    __first_message: str = None
    function_map = llm_commands

    def __init__(self):
        self.log = []

    @staticmethod
    def get_system_prompt():
        if TaskExecutor.__first_message is not None:
            return TaskExecutor.__first_message

        available_functions = get_llm_commands()

        template = f"""You are an assistant for a flashcard learning system.

## The flashcard system.
The flashcard system contains decks, and each deck is a collection of cards.

About Cards:
{Card.__doc__.strip("\n")}

About Decks:
{Deck.__doc__.strip("\n")}

About Virtual Decks:
{VirtualDeck.__doc__.strip("\n")}

## Available Functions for Interaction with the Flashcard System
You can interact with the system by calling specific Python functions, each of which performs an action. The available actions are:
{available_functions}

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

        TaskExecutor.__first_message = template
        return TaskExecutor.__first_message

    def execute_prompt(self, flashcard_manager: FlashcardManager, user_prompt: str, verbose: bool):
        llm_interactor = LLMInteractor(flashcard_manager)
        llm_communicator = LLMCommunicator("qwen3-8b", 0.8)
        # llm_communicator = LLMCommunicator("qwen2.5-14b-instruct", 0.8)
        # llm_communicator = LLMCommunicator("meta-llama-3.1-8b-instruct", 0.8)

        error_count = 0
        message_count = 0

        llm_communicator.set_system_prompt(TaskExecutor.get_system_prompt())
        message_to_send = user_prompt

        current_stream: Optional[ChunkedCardStream] = None

        while True:
            try:
                self.log += [("user", message_to_send)]
                if verbose:
                    print("\n=========== REQUEST ===========:")
                    print(message_to_send)

                message_count += 1
                answer = llm_communicator.send_message(message_to_send)

                self.log += [("answer", answer)]
                if verbose:
                    print("\n=========== RESPONSE ==========")
                    print(answer)

                commands = TaskExecutor.parse_llm_response(answer)
                results = TaskExecutor.execute_llm_response(llm_interactor, commands)
                if len(results) == 0:
                    return
                else:
                    if any(isinstance(it, ChunkedCardStream) for it in results):
                        if len(results) == 1:
                            message_to_send = f"The stream containing {len(results[0].items)} cards has been fully processed. You have left the card stream, and can **not** call abort_card_stream(message) any more. **YOU ACHIEVED ALL YOUR TASKS THAT YOU WANTED TO DO WITH THE STREAM**. In 99 % of all cases, you are done now and can just send an empty <execute></execute> block to finish this session.\n\n"
                            message_to_send += self.handle_card_stream(results[0], llm_communicator, llm_interactor, verbose)
                        else:
                            raise Exception(
                                "If you want to call a method that returns a stream, you may not call any other function in the same message.")
                    else:
                        message_to_send = self.deep_to_string(results)
                pass # debug opportunity
            except Exception as e:
                if verbose:
                    print(f"\nException raised: {e}.\n\nStack trace:\n{traceback.format_exc()}\n")
                self.log += [("exception", f"\nException raised: {e}.\n\nStack trace:\n{traceback.format_exc()}\n")]
                error_count += 1
                message_to_send = f"""An error occured: {e} Please try again!"""
            if error_count >= 5:
                raise RuntimeError("Too many errors. Abort execution.")
            if message_count >= 10:
                raise RuntimeError("Too many messages. Abort execution.")

    # my god is this ugly, make llm_communicator and llm_interactor class properties you idiot
    def handle_card_stream(self, chunked_cards: ChunkedCardStream, llm_communicator: LLMCommunicator,
                           llm_interactor: LLMInteractor, verbose: bool) -> str:

        stream_info = """You are currently in a card stream. You will be provided with groups of cards. You can use these cards to achieve your task.
If you want to continue to the next chunk, please return an empty <execute>...</execute> block.
You will **not** be able to see the previous chunk and the messages you sent in the previous chunks.
To end the stream early (before all cards are processed), please call the function "abort_card_stream(reason: str)". Only call this if there is an error, as you usually have to see all cards in the stream!!
        """
        llm_communicator.add_message(stream_info)
        llm_communicator.start_visibility_block()

        next_chunk = chunked_cards.next_chunk()
        message_to_send = "The next messages are:\n" + "\n\n".join(str(it) for it in next_chunk)

        all_commands = {}

        while True:
            try:
                self.log += [("user-stream", message_to_send)]
                if verbose:
                    print("\n=========== REQUEST (STREAM) ===========:")
                    print(message_to_send)
                answer = llm_communicator.send_message(message_to_send)
                self.log += [("answer", answer)]
                if verbose:
                    print("\n=========== RESPONSE (STREAM) ===========:")
                    print(answer)


                commands = TaskExecutor.parse_llm_response(answer)
                if any(c.func_name == "abort_card_stream" for c in commands):
                    if len(commands) == 1:
                        llm_communicator.end_visibility_block()
                        args_str = ", ".join(commands[0].args)
                        kw_args = str(commands[0].kwargs) if len(commands[0].kwargs) > 0 else ""
                        return f"You decided to exit the stream early for the following reason: {args_str}{kw_args}"
                    raise Exception("If you want to exit the card stream, you may not call any other function in the same message. **None** of your commands from the last message have been executed.")

                if any(self.llm_function_return_type(c.func_name) == ChunkedCardStream for c in commands):
                    raise Exception("You are already in a card stream. Exit this stream before entering a new one.")


                # command stats
                for command in commands:
                    if command.func_name not in all_commands:
                        all_commands[command.func_name] = 1
                    else:
                        all_commands[command.func_name] += 1

                if len(commands) > 0:
                    results = TaskExecutor.execute_llm_response(llm_interactor, commands)
                    message_to_send = self.deep_to_string(results)
                else:
                    llm_communicator.end_visibility_block()

                    if not chunked_cards.has_next():
                        llm_communicator.end_visibility_block()
                        func_call_times = sorted(all_commands.items(), key=lambda x: x[1], reverse=True)
                        return f"The card stream was fully processed. Because you have a limited context window, I cannot show you everything you did. You called the following functions (with frequency): {func_call_times}"

                    next_chunk = chunked_cards.next_chunk()
                    llm_communicator.start_visibility_block()
                    message_to_send = "The next messages are:\n" + "\n\n".join(str(it) for it in next_chunk)
            except Exception as e:
                self.log += [("exception", f"\nException raised: {e}.\n\nStack trace:\n{traceback.format_exc()}\n")]
                message_to_send = (
                    f"Exception raised: {e}. **The card stream is still active.** Remember to call the function 'abort_card_stream()' to abort the card stream prematurely if really necessary.")

    @staticmethod
    def llm_function_return_type(function_name: str):
        if function_name not in TaskExecutor.function_map:
            raise ValueError(f"Function {function_name} not known.")

        function = TaskExecutor.function_map[function_name]
        sig = inspect.signature(function)
        if sig.return_annotation is None:
            return type(None)
        else:
            return sig.return_annotation

    @staticmethod
    def deep_to_string(obj):
        if obj is None:
            return "None"
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, (int, float)):
            return str(obj)
        elif isinstance(obj, dict):
            items = []
            for key, value in obj.items():
                items.append(
                    f"{TaskExecutor.deep_to_string(key)}: {TaskExecutor.deep_to_string(value)}"
                )
            return "{" + ", ".join(items) + "}"
        elif isinstance(obj, (list, tuple)):
            elements = [TaskExecutor.deep_to_string(e) for e in obj]
            if isinstance(obj, list):
                return "[" + ", ".join(elements) + "]"
            else:
                return "(" + ", ".join(elements) + ")"
        elif isinstance(obj, set):
            elements = [TaskExecutor.deep_to_string(e) for e in obj]
            elements.sort()
            return "{" + ", ".join(elements) + "}"
        else:
            return str(obj)

    @dataclass
    class ParsedLLMCommand:
        func_name: str
        args: list
        kwargs: dict

    @staticmethod
    def parse_function_call(call_str):
        # Parse the string into an AST node
        try:
            tree = ast.parse(call_str, mode='eval')
        except SyntaxError as e:
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
        kwargs = {
            kw.arg: ast.literal_eval(kw.value)
            for kw in call_node.keywords
            if kw.arg is not None
        }

        return func_name, args, kwargs

    @staticmethod
    def parse_llm_response(response: str) -> list["TaskExecutor.ParsedLLMCommand"]:
        # Extract the execution plan block
        match = re.search(r"^ *<execute>(.*?)<\/execute>", response, re.DOTALL + re.MULTILINE)
        if not match:
            raise ValueError(
                "No execute block found in response. Remember to use <execute>...</execute> to mark your execution plan, and send an empty block to indicate that you do not wish to take any further action.")
        plan = match.group(1)

        commands: list[TaskExecutor.ParsedLLMCommand] = []
        for line in plan.splitlines():
            line = line.strip()
            if not line: continue
            func_name, args, kwargs = TaskExecutor.parse_function_call(line[1:].strip())
            commands += [TaskExecutor.ParsedLLMCommand(func_name, args, kwargs)]
        return commands

    @staticmethod
    def execute_llm_response(llm_interactor: LLMInteractor, commands: list["TaskExecutor.ParsedLLMCommand"]) -> list[
        str]:
        results = []
        for command in commands:
            if command.func_name not in TaskExecutor.function_map:
                raise ValueError(f"Unknown function name {command.func_name}.")
            result = TaskExecutor.function_map[command.func_name](llm_interactor, *command.args,
                                                                  **command.kwargs)  # self as first argument
            results.append(result)
        return results