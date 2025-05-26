import inspect
from dataclasses import dataclass


@dataclass
class RegistryEntry:
    function: callable
    parameters: dict


ACTION_REGISTRY: dict[str, RegistryEntry] = {}


def register_action(intent_name: str):
    """
    Decorator to register a function as an action for a specific intent.
    """

    def decorator(func):
        signature = inspect.signature(func)
        parameters = {
            name: str(param.annotation)
            if param.annotation != inspect.Parameter.empty
            else "Any"
            for name, param in signature.parameters.items()
            if name != "self"
        }
        ACTION_REGISTRY[intent_name] = RegistryEntry(
            function=func,
            parameters=parameters,
        )
        return func

    return decorator


def get_action_registry() -> dict[str, RegistryEntry]:
    """
    Returns the action registry.
    """
    return ACTION_REGISTRY


def get_action_registry_entry(intent_name: str) -> RegistryEntry | None:
    """
    Returns the action registry entry for a specific intent.
    """
    return ACTION_REGISTRY.get(intent_name)
