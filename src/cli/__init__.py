from dotenv import load_dotenv

from src.backend.modules.helpers import check_for_environment_variables
from src.backend.modules.recording.portaudio_stream_adapter import NoAudioDeviceException, PortaudioStream

load_dotenv(".env.local")

required_vars = [
    "AUDIO_DEVICE",
]

try:
    check_for_environment_variables(required_vars)
except EnvironmentError:
    try:
        PortaudioStream()
    except NoAudioDeviceException:
        print("No audio device found. Please set the AUDIO_DEVICE environment variable correctly.")
        exit(1)
