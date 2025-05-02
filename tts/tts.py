from TTS.api import TTS
import sounddevice as sd

_tts_model = None

#I have encapsulated the initialization in a separate function so that 
# we can initialize the TTS model at the beginning of the main program later.
def initialize_tts_model() -> None:
    # Initialize the TTS model
    global _tts_model
    if _tts_model is None:
        _tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)


def tts_and_play(text: str) -> None:
    """Synthesize English speech from text and play

    :param text: A piece of English text
    
    :return: None
    """
    initialize_tts_model()
    check_illegal_chars(text)

    # Generate audio (return numpy array)
    audio = _tts_model.tts(text)

    # Play
    sd.play(audio, samplerate=_tts_model.synthesizer.output_sample_rate)
    sd.wait()  # Block until finished


ALLOWED_CHARS = set("_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\n")

def check_illegal_chars(text: str):
    illegal = set(text) - ALLOWED_CHARS
    if illegal:
        raise ValueError(f"Input contains illegal characters: {''.join(sorted(illegal))}")




if __name__ == "__main__":
    text = "Let's begin your interactive learning session."
    tts_and_play(text)
