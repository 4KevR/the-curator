import json
import logging
import re

from src.backend.domain import (
    AbstractAnki,
    AbstractLLM,
    extract_intent_and_parameters_prompt,
    get_action_registry_entry,
)


class ActionService:
    def __init__(self, llm: AbstractLLM, anki: AbstractAnki):
        self.llm = llm
        self.anki = anki
        self.logger = logging.getLogger(__name__)

    def process_transcription(self, transcription: str):
        """
        Process the transcribed string to determine the action and parameters.
        """
        self.logger.debug(f"Processing transcription: {transcription}")

        # Use LLM to extract intent and parameters
        prompt = extract_intent_and_parameters_prompt(transcription)
        response = self.llm.generate([{"role": "user", "content": prompt}])

        try:
            json_match = re.search(r"\{.*?\{.*?\}.*?\}", response, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON object found in LLM response.")

            action_data = json.loads(json_match.group(0))
            intent = action_data.get("intent")
            parameters = action_data.get("parameters", {})
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {response}, Error: {e}")
            return {"error": "Failed to parse LLM response."}

        self.logger.debug(f"Extracted intent: {intent}, parameters: {parameters}")

        # Map intent to Anki adapter function
        return self._execute_action(intent, parameters)

    def _execute_action(self, intent: str, parameters: dict):
        """
        Execute the corresponding Anki adapter function based on the intent.
        """
        try:
            registry_entry = get_action_registry_entry(intent)
            if not registry_entry:
                self.logger.warning(f"Unknown intent: {intent}")
                return {"error": "Unknown intent."}

            # Call the registered function with the parameters
            return registry_entry.function(self.anki, **parameters)

        except Exception as e:
            self.logger.error(f"Error executing action for intent {intent}: {e}")
            return {"error": f"Failed to execute action: {e}"}
