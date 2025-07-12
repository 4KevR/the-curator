import logging

from src.backend.application import app, socketio

if __name__ == "__main__":
    logging.getLogger().info("Running Flask-SocketIO app...")
    socketio.run(app, host="127.0.0.1", port=5000, allow_unsafe_werkzeug=True)
