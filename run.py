import logging

from src.backend.application import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
