import logging

from src.backend.application import app

if __name__ == "__main__":
    logging.getLogger().info("Running Flask app...")
    app.run(host="127.0.0.1", port=5000)
