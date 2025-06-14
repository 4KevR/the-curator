from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from typing import Optional

from src.backend.modules.srs.abstract_srs import AbstractCard, AbstractDeck


class UserState(Enum):
    IN_COLLECTION = auto()
    IN_DECK = auto()
    IN_LEARN = auto()


@dataclass
class UserContext:
    state: UserState = UserState.IN_COLLECTION
    current_deck: Optional[AbstractDeck] = None
    current_card: Optional[AbstractCard] = None


def allowed_in_states(*allowed_states: UserState):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.user_context.state not in allowed_states:
                raise PermissionError(
                    f"Operation '{func.__name__}' not allowed in state '{self.user_context.state.name}'"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def _get_allowed_commands_by_state(self, state: UserState) -> list[str]:
    allowed_cmds = []
    for attr in dir(self):
        func = getattr(self, attr)
        if callable(func) and hasattr(func, "_llm_command") and hasattr(func, "_allowed_states"):
            if state in func._allowed_states:
                allowed_cmds.append(func._llm_command["name"])
    return allowed_cmds
