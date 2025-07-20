import atexit
import base64
import logging
import os
import tempfile
import threading
import uuid
from collections import defaultdict

import nltk
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from src.backend.controllers.action import action_blueprint
from src.backend.controllers.speech import speech_blueprint
from src.backend.modules.ai_assistant.conversation_manager import ConversationManager
from src.backend.modules.ai_assistant.progress_callback import ProgressCallback
from src.backend.modules.ai_assistant.state_manager import StateFinishedSingleLearnStep
from src.backend.modules.ai_assistant.task_states import StateFinishedDueToMissingInformation
from src.backend.modules.asr.cloud_lecture_translator import CloudLectureTranslatorASR
from src.backend.modules.asr.local_whisper_asr import LocalWhisperASR
from src.backend.modules.llm import LMStudioLLM
from src.backend.modules.search.llama_index import LlamaIndexExecutor
from src.backend.modules.srs.anki_module import AnkiSRS

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3050"}})
app.register_blueprint(action_blueprint)
app.register_blueprint(speech_blueprint)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3050")

llm = LMStudioLLM("meta-llama-3.1-8b-instruct", 0.001, 1000)
whisper_asr = LocalWhisperASR("openai/whisper-medium")
lecture_translator_asr: dict[str, CloudLectureTranslatorASR] = {}
anki_srs_adapters: dict[str, AnkiSRS] = {}
llama_index_executors: dict[str, LlamaIndexExecutor] = {}
temporary_user_data: dict[str, str] = defaultdict(str)
user_conversations: dict[str, ConversationManager] = {}

# Global locks for AnkiSRS and LlamaIndexExecutor initialization
anki_srs_init_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
llama_index_init_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

# For managing sentence processing queue
user_sentence_queues: dict[str, list[str]] = defaultdict(list)
user_processing_flags: dict[str, bool] = defaultdict(bool)


# Cleanup function to close Anki collections on application exit
def _close_anki_collections():
    for user, anki_srs in anki_srs_adapters.items():
        try:
            anki_srs.close()
            logger.info(f"Anki collection for user '{user}' closed during shutdown.")
        except Exception as e:
            logger.error(f"Error closing Anki collection for user '{user}': {e}")


atexit.register(_close_anki_collections)


class TaskStartClassifier:
    _prompt_template = """
You are an AI assistant for a flashcard management system with decks and cards.

Given a user's spoken transcription, decide if it should be processed as a flashcard manager task. Tasks include: asking questions, modifying flashcards or decks, or adding information to them.

- If the transcription contains such a task or question, respond with "yes".
- If it only contains filler words, non-logical statements, or is incomplete (e.g., unfinished thoughts or general introductions), respond with "no".

Transcription:
{transcription}

Respond with "yes" or "no" only.
"""

    def __init__(self):
        self.llm = llm

    def classify(self, transcription: str) -> bool:
        prompt = self._prompt_template.format(transcription=transcription)
        response = self.llm.generate([{"role": "user", "content": prompt}])
        if response.strip().lower() == "yes":
            return True
        elif response.strip().lower() == "no":
            return False
        else:
            raise ValueError(f"Unexpected response from LLM: {response}")


class SocketIOProgressCallback(ProgressCallback):
    def handle(self, message, is_srs_action=False):
        emit("action_progress", {"message": message, "is_srs_action": is_srs_action})


task_start_classifier = TaskStartClassifier()
socketio_progress_callback = SocketIOProgressCallback()


def get_conversation_manager(user_name):
    if user_name in user_conversations:
        return user_conversations[user_name]

    # Ensure AnkiSRS initialization is thread-safe
    with anki_srs_init_locks[user_name]:
        if user_name not in anki_srs_adapters:
            logger.info(f"Creating new AnkiSRS adapter for user: {user_name}")
            anki_srs_adapters[user_name] = AnkiSRS(user_name)

    # Ensure LlamaIndexExecutor initialization is thread-safe
    with llama_index_init_locks[user_name]:
        if user_name not in llama_index_executors:
            llama_index_executors[user_name] = LlamaIndexExecutor(user_name)

    user_conversations[user_name] = ConversationManager(
        llm, anki_srs_adapters[user_name], llama_index_executors[user_name], socketio_progress_callback
    )
    return user_conversations[user_name]


def _process_next_sentence(user: str):
    if not user_sentence_queues[user]:
        user_processing_flags[user] = False
        return

    user_processing_flags[user] = True
    full_sentences = " ".join(user_sentence_queues[user])

    try:
        if not task_start_classifier.classify(full_sentences):
            logger.info(f"Query for user {user} classified as not a valid task: {full_sentences}")
            # If not a valid task, still process the next one in queue
            _process_next_sentence(user)
            return

        emit("received_complete_sentence", {"sentence": full_sentences})
        logger.info(f"Received complete sentence: {full_sentences}")
        user_sentence_queues[user].clear()
        conversation_manager = get_conversation_manager(user)
        result = conversation_manager.process_query(full_sentences)
        if (
            type(result.finish_state) is StateFinishedSingleLearnStep
            or type(result.finish_state) is StateFinishedDueToMissingInformation
        ):
            emit_event = "action_single_result"
        else:
            emit_event = "action_result"
        emit(
            emit_event,
            {"task_finish_message": result.task_finish_message, "question_answer": result.question_answer},
        )
    except Exception as e:
        emit("action_error", {"error": str(e)})
    finally:
        # After processing, check for next sentence in queue
        _process_next_sentence(user)  # Recursively call to process next if available


@socketio.on("submit_action")
def handle_submit_action(data):
    user = data.get("user")
    transcription = data.get("transcription")
    if not user or not transcription:
        emit("action_error", {"error": "User and transcription required."})
        return
    if not task_start_classifier.classify(transcription):
        emit("action_error", {"error": "Query does not contain a valid task."})
        return
    conversation_manager = get_conversation_manager(user)
    try:
        result = conversation_manager.process_query(transcription)
        if (
            type(result.finish_state) is StateFinishedSingleLearnStep
            or type(result.finish_state) is StateFinishedDueToMissingInformation
        ):
            emit_event = "action_single_result"
        else:
            emit_event = "action_result"
        emit(
            emit_event,
            {"task_finish_message": result.task_finish_message, "question_answer": result.question_answer},
        )
    except Exception as e:
        emit("action_error", {"error": str(e)})


@socketio.on("submit_action_file")
def handle_submit_action_file(data):
    logger.info("Received file action request")
    user = data.get("user")
    file_b64 = data.get("file_b64")
    if not user or not file_b64:
        emit("action_error", {"error": "User and file required."})
        return
    # Save file to temp and transcribe
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(base64.b64decode(file_b64))
        tmp_path = tmp.name
    try:
        emit("action_progress", {"message": "Starting transcription..."})
        transcription = whisper_asr.transcribe_wav_file(tmp_path)
        emit("action_progress", {"message": "Transcription completed."})
        emit("action_progress", {"message": f"User message: {transcription}"})
        os.remove(tmp_path)
        if not task_start_classifier.classify(transcription):
            emit("action_error", {"error": "Query does not contain a valid task."})
            return
        conversation_manager = get_conversation_manager(user)
        result = conversation_manager.process_query(transcription)
        if (
            type(result.finish_state) is StateFinishedSingleLearnStep
            or type(result.finish_state) is StateFinishedDueToMissingInformation
        ):
            emit_event = "action_single_result"
        else:
            emit_event = "action_result"
        emit(
            emit_event,
            {"task_finish_message": result.task_finish_message, "question_answer": result.question_answer},
        )
    except Exception as e:
        emit("action_error", {"error": str(e)})
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@socketio.on("start_audio_streaming")
def handle_start_audio_streaming(data):
    user = data.get("user")
    lecture_translator_asr[user] = CloudLectureTranslatorASR()
    lecture_translator_asr[user]._send_white_noise()
    emit("acknowledged_stream_start", {"user": user})
    logger.info(f"Started audio streaming for user: {user}")
    temporary_user_data[user] = ""


@socketio.on("submit_stream_batch")
def handle_submit_stream_batch(data):
    user = data.get("user")
    print(f"Received stream batch for user: {user}")
    b64_pcm = data.get("b64_pcm")
    duration = data.get("duration", 0)
    if not user or not b64_pcm:
        emit("action_error", {"error": "User and audio data required."})
        return
    lecture_translator_asr[user]._send_audio(encoded_audio=b64_pcm, duration=duration)
    read_data = lecture_translator_asr[user]._read_from_queue()
    if read_data:
        emit("streamed_sentence_part", {"part": read_data})
        temporary_user_data[user] += read_data
    sentences = nltk.tokenize.sent_tokenize(temporary_user_data[user])
    complete_sentences = [sentence for sentence in sentences if sentence.endswith((".", "!", "?"))]
    if complete_sentences:
        # Add all newly completed sentences to the queue
        for sentence in complete_sentences:
            user_sentence_queues[user].append(sentence)

        # Clear temporary data for the sentences that were just added to the queue
        temporary_user_data[user] = ""

        # If no processing is currently active for this user, start processing the queue
        if not user_processing_flags[user]:
            _process_next_sentence(user)


@socketio.on("new_conversation")
def handle_new_conversation(data):
    user = data.get("user")
    if not user:
        emit("action_error", {"error": "User required."})
        return
    conversation_manager = get_conversation_manager(user)
    conversation_manager.history_manager.clear_history()


@app.route("/api/anki/decks/<user_name>", methods=["GET"])
def get_anki_decks(user_name):
    try:
        if user_name not in anki_srs_adapters:
            anki_srs_adapters[user_name] = AnkiSRS(user_name)
        anki_srs = anki_srs_adapters[user_name]
        decks = anki_srs.get_all_decks()
        deck_names = [deck.name for deck in decks]
        return jsonify({"decks": deck_names})
    except Exception as e:
        logger.error(f"Error getting Anki decks for user {user_name}: {e}")
        return jsonify({"error": str(e)}), 500


# Directory to store uploaded/exported Anki files
ANKI_FILE_STORAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/anki_files"))
if not os.path.exists(ANKI_FILE_STORAGE_DIR):
    os.makedirs(ANKI_FILE_STORAGE_DIR)


@app.route("/api/anki/upload", methods=["POST"])
def upload_anki_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = f"{uuid.uuid4()}.apkg"
        filepath = os.path.join(ANKI_FILE_STORAGE_DIR, filename)
        file.save(filepath)
        return jsonify({"file_id": filename}), 200
    return jsonify({"error": "File upload failed"}), 500


@app.route("/api/anki/download/<file_id>", methods=["GET"])
def download_anki_file(file_id):
    filepath = os.path.join(ANKI_FILE_STORAGE_DIR, file_id)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=file_id)
    return jsonify({"error": "File not found"}), 404


@socketio.on("import_anki_collection")
def handle_import_anki_collection(data):
    user = data.get("user")
    file_id = data.get("file_id")
    if not user or not file_id:
        emit("action_error", {"error": "User and file ID required for import."})
        return

    filepath = os.path.join(ANKI_FILE_STORAGE_DIR, file_id)
    if not os.path.exists(filepath):
        emit("action_error", {"error": f"File with ID {file_id} not found on server."})
        return

    try:
        anki_srs_adapters[user].import_deck_from_apkg(filepath)
        emit("action_progress", {"message": "Anki deck imported", "is_srs_action": True})
    except Exception as e:
        emit("action_error", {"error": f"Failed to import Anki deck: {str(e)}"})
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@socketio.on("export_anki_collection")
def handle_export_anki_collection(data):
    user = data.get("user")
    deck_name = data.get("deck_name", "Default")

    if not user:
        emit("action_error", {"error": "User required for export."})
        return

    tmp_filepath = ""
    try:
        filename = f"{uuid.uuid4()}.apkg"
        tmp_filepath = os.path.join(ANKI_FILE_STORAGE_DIR, filename)

        anki_srs = anki_srs_adapters[user]
        deck_to_export = anki_srs.get_deck_by_name_or_none(deck_name)

        if not deck_to_export:
            emit("action_error", {"error": f"Deck '{deck_name}' not found for export."})
            return

        anki_srs.export_deck_to_apkg(deck_to_export, tmp_filepath)

        emit("anki_collection_exported", {"file_id": filename})
        emit(
            "action_progress",
            {"message": f"Anki deck '{deck_name}' exported", "is_srs_action": True},
        )
    except Exception as e:
        emit("action_error", {"error": f"Failed to export Anki deck: {str(e)}"})


@socketio.on("reset_anki_collection")
def handle_reset_anki_collection(data):
    user = data.get("user")
    if not user:
        emit("action_error", {"error": "User required for reset."})
        return
    try:
        anki_srs_adapters[user].clear_collection()
        emit("action_progress", {"message": "Anki collection reset successfully!", "is_srs_action": True})
    except Exception as e:
        emit("action_error", {"error": f"Failed to reset Anki collection: {str(e)}"})
