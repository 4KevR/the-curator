import os

from dotenv import load_dotenv

load_dotenv(".env.local")

required_vars = [
    "AUDIO_DEVICE",
]
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
