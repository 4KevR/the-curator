from src.backend.domain import get_action_registry


def extract_intent_and_parameters_prompt(transcription: str) -> str:
    registry = get_action_registry()
    intents_with_parameters = "\n".join(
        f"""- {intent}: {
        ", ".join(
            [
                f"{param} ({type_})"
                for param, type_ in registry_entry.parameters.items()
            ]
        )
        }"""
        for intent, registry_entry in registry.items()
    )
    return f"""
        You are an assistant that extracts intents and parameters from user requests.
        Given the transcription below, identify the action to perform in the Anki
        environment and the relevant parameters. Return the result in JSON format with
        'intent' and 'parameters'.\n
        Intents and their parameters are as follows:
        {intents_with_parameters}
        Here is an example for the JSON:
        {{
            "intent": "action_name",
            "parameters": {{
                "param1": "value1",
            }}
        }}
        Transcription: {transcription}
        Stop your respone by writing "User:"
    """.strip()
