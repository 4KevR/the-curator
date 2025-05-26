from flask import Blueprint, jsonify, request

from src.backend.adapter.anki import Anki
from src.backend.adapter.kit_llm import KitLLM
from src.backend.service.action_service import ActionService

action_blueprint = Blueprint("action", __name__)

# Initialize adapters
llm_adapter = KitLLM()


@action_blueprint.route("/action", methods=["POST"])
def perform_action():
    try:
        # Get transcription and user from request
        data = request.get_json()
        transcription = data.get("transcription", "")
        user_name = data.get("user", "default_user")

        if not transcription:
            return jsonify({"error": "Transcription is required."}), 400

        if not user_name:
            return jsonify({"error": "User is required."}), 400

        # Initialize Anki adapter with the provided user
        anki_adapter = Anki(user_name=user_name)
        action_service = ActionService(llm=llm_adapter, anki=anki_adapter)

        # Process transcription
        result = action_service.process_transcription(transcription)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
