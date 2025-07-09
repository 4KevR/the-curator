import base64
import logging
import os
import tempfile
from collections import defaultdict

import nltk
from flask import Flask
from flask_socketio import SocketIO, emit

from src.backend.controllers.action import action_blueprint
from src.backend.controllers.speech import speech_blueprint
from src.backend.modules.ai_assistant import StateManager
from src.backend.modules.ai_assistant.progress_callback import ProgressCallback
from src.backend.modules.ai_assistant.state_manager import StateFinishedSingleLearnStep
from src.backend.modules.asr.cloud_lecture_translator import CloudLectureTranslatorASR
from src.backend.modules.asr.local_whisper_asr import LocalWhisperASR
from src.backend.modules.llm.kit_llm_req import KitLLMReq
from src.backend.modules.search.llama_index import LlamaIndexExecutor
from src.backend.modules.srs.anki_module import AnkiSRS

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(action_blueprint)
app.register_blueprint(speech_blueprint)
socketio = SocketIO(app)

llm = KitLLMReq(os.getenv("LLM_URL"), 0.05, 2048)
whisper_asr = LocalWhisperASR("openai/whisper-medium")
lecture_translator_asr: dict[str, CloudLectureTranslatorASR] = {}
anki_srs_adapters: dict[str, AnkiSRS] = {}
llama_index_executors: dict[str, LlamaIndexExecutor] = {}
temporary_user_data: dict[str, str] = defaultdict(str)


def get_adapters(user_name):
    if user_name not in anki_srs_adapters:
        anki_srs_adapters[user_name] = AnkiSRS(user_name)
    if user_name not in llama_index_executors:
        llama_index_executors[user_name] = LlamaIndexExecutor(user_name)
    return anki_srs_adapters[user_name], llama_index_executors[user_name]


class SocketIOProgressCallback(ProgressCallback):
    def handle(self, message, is_srs_action=False):
        emit("action_progress", {"message": message, "is_srs_action": is_srs_action})


@socketio.on("submit_action")
def handle_submit_action(data):
    user = data.get("user")
    transcription = data.get("transcription")
    if not user or not transcription:
        emit("action_error", {"error": "User and transcription required."})
        return
    anki_adapter, llama_executor = get_adapters(user)
    try:
        result = StateManager(llm, anki_adapter, llama_executor, progress_callback=SocketIOProgressCallback()).run(
            transcription
        )
        if type(result.finish_state) is StateFinishedSingleLearnStep:
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
        anki_adapter, llama_executor = get_adapters(user)
        result = StateManager(llm, anki_adapter, llama_executor, progress_callback=SocketIOProgressCallback()).run(
            transcription
        )
        if type(result.finish_state) is StateFinishedSingleLearnStep:
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
    data = {
        "b64_pcm": b64_pcm,
        "duration": duration,
    }
    lecture_translator_asr[user]._send_audio(encoded_audio=data["b64_pcm"], duration=data["duration"])
    read_data = lecture_translator_asr[user]._read_from_queue()
    if read_data:
        emit("streamed_sentence_part", {"part": read_data})
        temporary_user_data[user] += read_data
    sentences = nltk.tokenize.sent_tokenize(temporary_user_data[user])
    complete_sentences = [sentence for sentence in sentences if sentence.endswith((".", "!", "?"))]
    if complete_sentences:
        emit("received_complete_sentence", {"sentence": complete_sentences[0]})
        temporary_user_data[user] = ""
        anki_adapter, llama_executor = get_adapters(user)
        result = StateManager(llm, anki_adapter, llama_executor, progress_callback=SocketIOProgressCallback()).run(
            complete_sentences[0]
        )
        if type(result.finish_state) is StateFinishedSingleLearnStep:
            emit_event = "action_single_result"
        else:
            emit_event = "action_result"
        emit(
            emit_event,
            {"task_finish_message": result.task_finish_message, "question_answer": result.question_answer},
        )
