from flask import Blueprint, jsonify, request

from src.backend.adapter.lecture_translator import LectureTranslatorASR

speech_blueprint = Blueprint("speech", __name__)
translator = LectureTranslatorASR()


@speech_blueprint.route("/transcribe", methods=["POST"])
def transcribe_audio():
    request_json = request.get_json()
    if "audio" in request.files:
        audio_file = request.files["audio"]
        audio_data = audio_file.read()
    else:
        audio_data = request_json["b64_pcm"]
    duration = request_json["duration"]
    try:
        transcription = translator.transcribe(audio_data, duration)
    except TimeoutError:
        return jsonify({"error": "Timeout error during transcription"}), 500
    return jsonify({"transcription": transcription})
