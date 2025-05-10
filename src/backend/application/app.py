from flask import Flask

from src.backend.controller import action_blueprint, speech_blueprint

app = Flask(__name__)
app.register_blueprint(action_blueprint)
app.register_blueprint(speech_blueprint)
