import base64
import os
import tempfile

from flask import Flask
from flask_socketio import SocketIO, emit

from src.backend.controllers.action import action_blueprint
from src.backend.controllers.speech import speech_blueprint
from src.backend.modules.ai_assistant import StateManager
from src.backend.modules.asr.local_whisper_asr import LocalWhisperASR
from src.backend.modules.llm.kit_llm_req import KitLLMReq
from src.backend.modules.search.llama_index import LlamaIndexExecutor
from src.backend.modules.srs.anki_module import AnkiSRS

app = Flask(__name__)
app.register_blueprint(action_blueprint)
app.register_blueprint(speech_blueprint)
socketio = SocketIO(app)

llm = KitLLMReq(os.getenv("LLM_URL"), 0.05, 2048)
whisper_asr = LocalWhisperASR("openai/whisper-medium")
anki_srs_adapters = {}
llama_index_executors = {}


def get_adapters(user_name):
    if user_name not in anki_srs_adapters:
        anki_srs_adapters[user_name] = AnkiSRS(user_name)
    if user_name not in llama_index_executors:
        llama_index_executors[user_name] = LlamaIndexExecutor(user_name)
    return anki_srs_adapters[user_name], llama_index_executors[user_name]


@socketio.on("submit_action")
def handle_submit_action(data):
    user = data.get("user")
    transcription = data.get("transcription")
    if not user or not transcription:
        emit("action_error", {"error": "User and transcription required."})
        return
    anki_adapter, llama_executor = get_adapters(user)

    def progress_callback(msg: str, is_srs_action: bool = False):
        emit("action_progress", {"message": msg, "is_srs_action": is_srs_action})

    try:
        result = StateManager(llm, anki_adapter, llama_executor, progress_callback=progress_callback).run(
            transcription, True
        )
        emit("action_result", {"result": result.__dict__})
    except Exception as e:
        emit("action_error", {"error": str(e)})


@socketio.on("submit_action_file")
def handle_submit_action_file(data):
    print("Received file action request")
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
        anki_adapter, llama_executor = get_adapters(user)

        def progress_callback(msg: str, is_srs_action: bool = False):
            emit("action_progress", {"message": msg, "is_srs_action": is_srs_action})

        result = StateManager(llm, anki_adapter, llama_executor, progress_callback=progress_callback).run(
            transcription, True
        )
        emit("action_result", {"result": result.__dict__})
    except Exception as e:
        emit("action_error", {"error": str(e)})
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
