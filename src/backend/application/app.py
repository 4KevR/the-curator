from flask import Flask

from src.backend.controllers.action import action_blueprint
from src.backend.controllers.speech import speech_blueprint

app = Flask(__name__)
app.register_blueprint(action_blueprint)
app.register_blueprint(speech_blueprint)
