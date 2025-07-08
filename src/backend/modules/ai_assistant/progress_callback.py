from abc import ABC, abstractmethod

from overrides import override


class ProgressCallback(ABC):
    @abstractmethod
    def handle(self, message: str, is_srs_action: bool = False):
        raise NotImplementedError()


class NoProgressCallback(ProgressCallback):

    @override
    def handle(self, message: str, is_srs_action: bool = False):
        pass
