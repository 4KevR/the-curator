import json
import logging

import requests
from sseclient import SSEClient

from src.backend.modules.asr import CloudLectureTranslatorASR

logger = logging.getLogger(__name__)


class LocalLectureTranslatorASR(CloudLectureTranslatorASR):
    def _read_text(self):
        logging.debug("Starting SSEClient")
        messages = SSEClient(self.url + "/" + self.api + "/stream?channel=" + self.session_id)

        for msg in messages:
            if len(msg.data) == 0:
                break
            try:
                data = json.loads(msg.data)
                if "markup" in data:
                    continue
                if "seq" in data:
                    logging.debug(f"Received data: {data}")
                    transcription = data["seq"].replace("<br><br>", "")
                    data = {
                        "transcription": transcription,
                        "user": "test_user",
                    }
                    logger.info(f"Sending: {transcription}")
                    requests.post(url="http://127.0.0.1:5000/action", json=data)

            except json.decoder.JSONDecodeError:
                logging.debug(
                    """WARNING: json.decoder.JSONDecodeError(this may happen
                      when running tts system but no video generation)""",
                )
                continue
