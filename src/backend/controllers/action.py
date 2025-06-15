import nltk
from flask import Blueprint, jsonify, request

nltk.download("punkt_tab")

action_blueprint = Blueprint("action", __name__)

# Initialize adapters
temporary_user_data: dict[str, tuple[str]] = dict()


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

        temporary_user_data[user_name] = (
            transcription if not temporary_user_data.get(user_name) else temporary_user_data[user_name] + transcription
        )

        # Check if transcription contains a complete sentence
        sentences = nltk.tokenize.sent_tokenize(temporary_user_data[user_name])
        complete_sentences = [sentence for sentence in sentences if sentence.endswith((".", "!", "?"))]
        if not complete_sentences:
            print("No complete sentence found in transcription.")
            return jsonify({"message": "Waiting for a complete sentence."}), 200

        print(f"Processing transcription for user '{user_name}': {complete_sentences[0]}")

        # Initialize Anki adapter with the provided user
        # anki_adapter = AnkiSRS(user_name=user_name)

        # TODO integrate new setup

        # Process transcription
        # result = action_service.process_transcription(complete_sentences[0])
        # temporary_user_data[user_name] = temporary_user_data[user_name][len(complete_sentences[0]) :].strip()

        return jsonify(""), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
