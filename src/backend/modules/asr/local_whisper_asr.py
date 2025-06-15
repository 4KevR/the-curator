import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline

from src.backend.modules.asr.abstract_asr import AbstractASR

logger = logging.getLogger(__name__)


def _create_pipeline(model_name: str, only_cpu: bool = False) -> Pipeline:
    device = "cuda:0" if not only_cpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device {device}.")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )


class LocalWhisperASR(AbstractASR):
    def __init__(self, model: str, only_cpu: bool = False):
        self.pipeline = _create_pipeline(model, only_cpu)

    def transcribe(self, audio_chunk: str, duration: int) -> str:
        raise NotImplementedError  # TODO: I do not even understand what this method should do.

    def transcribe_wav_file(self, audio_file_path: str) -> str:
        return self.pipeline(audio_file_path)["text"]
