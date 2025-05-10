import os
import sys

import sounddevice as sd
from TTS.api import TTS

_tts_model = None


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# I have encapsulated the initialization in a separate function so that
# we can initialize the TTS model at the beginning of the main program later.
def initialize_tts_model() -> None:
    # Initialize the TTS model
    global _tts_model
    if _tts_model is None:
        _tts_model = TTS(
            model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False
        )


def tts_and_play(text: str) -> None:
    """Synthesize English speech from text and play

    :param text: A piece of English text

    :return: None
    """
    with HiddenPrints():
        initialize_tts_model()
        check_illegal_chars(text)

        # Generate audio (return numpy array)
        audio = _tts_model.tts(text)

        # Play
        sd.play(
            audio, samplerate=_tts_model.synthesizer.output_sample_rate, blocking=True
        )
        sd.wait()  # Block until finished

        # # Save the audio to a WAV file
        # unique_id = uuid.uuid4().hex
        # file_name = f"{unique_id}.wav"
        # write(file_name, _tts_model.synthesizer.output_sample_rate, np.array(audio))


ALLOWED_CHARS = set(
    "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\n"
)


def check_illegal_chars(text: str):
    illegal = set(text) - ALLOWED_CHARS
    if illegal:
        raise ValueError(
            f"Input contains illegal characters: {''.join(sorted(illegal))}"
        )


if __name__ == "__main__":
    text = "Let's begin your interactive learning session."
    tts_and_play(text)
