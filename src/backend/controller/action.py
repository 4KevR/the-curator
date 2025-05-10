from flask import Blueprint

action_blueprint = Blueprint("action", __name__)


@action_blueprint.route("/action", methods=["POST"])
def perform_action():
    return "Action performed successfully", 200
