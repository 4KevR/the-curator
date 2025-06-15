import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cli.tts import tts_and_play  # noqa: E402

# Recommendation:
# Do not use sentences that are too long.
# Do not use multiple punctuation marks together.

if __name__ == "__main__":
    # Normal
    text1 = "Hello, world!"
    text2 = "Hello, world!\n"
    # Sound stuck or model error
    text3 = "Hello, world?!?!?!?!?!?!"
    # Illegal character
    text4 = "Hello, @world!"
    # It will be divided into multiple small sentences
    text5 = """Hello, world! Hello, world! Hello, world! Hello, world! Hello, world!
    Hello, world!Hello, world! Hello, world! Hello, world! Hello, world! Hello, world!
    Hello, world! Hello, world!"""
    # It doesn't work if the sentence is too long
    text6 = """Hello, this is tts, this is tts, this is tts, this is tts,
    this is tts, this is tts, this is tts, this is tts, this is tts."""

    tts_and_play(text1)
    tts_and_play(text2)
