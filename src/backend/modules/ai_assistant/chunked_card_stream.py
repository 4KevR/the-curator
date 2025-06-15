# TODO: THIS CLASS IS NOW UNUSED! DELETE!

import math

from src.backend.modules.srs.abstract_srs import AbstractCard


class ChunkedCardStream:
    """A class presenting a possibly large number of cards as a stream of chunks."""

    def __init__(self, items: list[AbstractCard], chunk_size: int = 5):
        self.items = items
        self.chunk_size = chunk_size
        self.current_index = 0
        self.is_finished = False

    def remaining_chunks(self):
        """How many chunks are left. The last chunk may be shorter than chunk_size."""
        return math.ceil((len(self.items) - self.current_index) / self.chunk_size)

    def has_next(self):
        """Whether there are more chunks to be read."""
        return self.current_index < len(self.items)

    def next_chunk(self):
        """Returns the next chunk of cards. If there are no more chunks, returns an empty list."""
        if not self.has_next():
            return []
        res = self.items[self.current_index : self.current_index + self.chunk_size]
        self.current_index += self.chunk_size
        return res
