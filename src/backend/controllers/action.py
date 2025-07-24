import logging
import os

from flask import Blueprint, jsonify, request

from src.backend.modules.ai_assistant import StateManager
from src.backend.modules.llm import KitLLM, LMStudioLLM
from src.backend.modules.search.llama_index import LlamaIndexExecutor
from src.backend.modules.srs.anki_module import AnkiSRS

logger = logging.getLogger(__name__)

action_blueprint = Blueprint("action", __name__)

# Initialize adapters
anki_srs_adapters: dict[str, AnkiSRS] = dict()
llama_index_executors: dict[str, LlamaIndexExecutor] = dict()

# LLM
if os.getenv("LLM_TO_USE").lower() == "local":
    llm = LMStudioLLM("meta-llama-3.1-8b-instruct", 0.001, 1000)
elif os.getenv("LLM_TO_USE").lower() == "hosted":
    llm = KitLLM(0.001, 1000)
else:
    raise ValueError("LLM_TO_USE environment variable must be set to 'local' or 'hosted'.")


@action_blueprint.route("/action", methods=["POST"])
def perform_action():
    try:
        # Get transcription and user from request
        data: dict = request.get_json()
        transcription = data.get("transcription")
        user_name = data.get("user")

        if not transcription:
            return jsonify({"error": "Transcription is required."}), 400

        if not user_name:
            return jsonify({"error": "User is required."}), 400

        logger.info(f"Processing transcription for user '{user_name}': {transcription}")

        # Initialize Anki adapter with the provided user
        if not anki_srs_adapters.get(user_name):
            anki_srs_adapters[user_name] = AnkiSRS(user_name)
        anki_adapter = anki_srs_adapters[user_name]

        if not llama_index_executors.get(user_name):
            llama_index_executors[user_name] = LlamaIndexExecutor(user_name)
        llama_index_executor = llama_index_executors[user_name]

        # Process transcription
        result = StateManager(llm, anki_adapter, llama_index_executor).run(transcription, True)
        logger.info(f"Result for user '{user_name}': {result.question_answer}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(e)
        return jsonify({"error": str(e)}), 500
