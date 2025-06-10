# =============================================================
# I modified the four classes: Flag, Card, Deck, CardState->CardType,
# I didn't make any change to the class VirtualDeck.
# =============================================================
# The following 2 classes not only implement Anki-related operations,
# but also actually simulate the functions of the Anki software,
# which should be what you use for testing.

# Now we should be able to use the Anki class in adapter\anki_module.py instead
# But there is no feature for virtual decks in Anki class!!!
# Please add those features.
class FlashcardManager:
    pass


class LLMInteractor:
    pass


# =============================================================
# =============================================================

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import openai
import rapidfuzz

from src.backend.modules.srs.anki.dataclasses import CardInfo, DeckCardsInfo


class Flag(Enum):
    NONE = "none"
    RED = "red"
    ORANGE = "orange"
    GREEN = "green"
    BLUE = "blue"
    PINK = "pink"
    CYAN = "cyan"
    PURPLE = "purple"

    @staticmethod
    def from_str(s: str):
        s = s.lower()
        for flag in Flag:
            if flag.value == s:
                return flag
        raise ValueError(f"{s} is not a valid flag.")


class CardType(Enum):
    NEW = "New"
    LEARN = "Learn"
    REVIEW = "Review"
    RELEARN = "Relearn"

    @staticmethod
    def from_str(s: str):
        # s = s.lower()
        for state in CardType:
            if state.value == s:
                return state
        raise ValueError(f"{s} is not a valid state.")


@dataclass(frozen=False)
class Card:
    """A Card is a representation of a flashcard, containing a question and an
    answer. The card is uniquely identified by the id.

    Properties:
      id (int): The id uniquely identifies the card.
          The id is the only way to identify a card.
      deck_id (int): The deck this card belongs to.
      cardInfo (CardInfo): Detailed information of the card and return a dataclass containing:
        - card_id, note_id, deck_id, template_index
        - card type (new card/studying/reviewing/relearning)
        - queue type (queue number and name)
        - due (due), interval (ivl)
        - factor (ease), review/study times (reps, lapses, left)
        - flags (flags), tags (tags) and note field content (fields)
    """

    id: int
    deck_id: int
    cardInfo: CardInfo

    def __str__(self):
        s = f"""Card {self.id} from the deck {self.deck_id}
Question:
{self.cardInfo.fields[0]}

Answer:
{self.cardInfo.fields[1]}

Flag: {self.cardInfo.flags}
Card Type: {self.cardInfo.type["name"]}"""
        return s


@dataclass(frozen=False)
class Deck:
    """A Deck represents a collection of flashcards.

    Properties:
       id (int): The id uniquely identifies the deck. The id is the only way to identify a deck.
       name (str): The name of the deck. This is **not** the id, and is **not** sufficient to address decks.
       cards (int, List[int]): The cards contained in the deck. The order has no meaning.
    """

    id: int
    name: str
    cards: DeckCardsInfo

    def __str__(self):
        s = f"""Deck '{self.name}' (id: {self.id}) containing {self.cards.total_cards} cards"""
        return s

    # self.copy() is deleted, because Anki does not allow decks with the same name.


@dataclass(frozen=False)
class VirtualDeck:
    """A Virtual Deck represents a collection of flashcards. However, the
    flashcards themselves are part of another deck; a virtual deck is a
    temporary collection of flashcards. Any changes to the cards in the virtual
    deck will also change the cards in their 'normal' deck. Virtual Decks are
    e.g. used to represent the result of search queries. Virtual Decks do not
    have names.

    Properties:
       id (str): The id uniquely identifies the virtual deck. It is represented as "virt_deck_xxxx_xxxx", with x being hexadecimal digits.
          The id is the only way to identify a virtual deck. It is assigned randomly, there is no way to guess it!
       description (str): A description that may explain how this deck was created. Optional, may be left blank.
       cards (List[Card]): The cards contained in the virtual deck. The order has no meaning.
    """

    id: int
    description: str
    cards: List[Card]

    def __str__(self):
        hex_str = f"{self.id:08x}"  # pad to 8 hex digits
        hex_id = f"virt_deck_{hex_str[:4]}_{hex_str[4:]}"
        s = f"""Virtual Deck (id: {hex_id}) containing {len(self.cards)} cards."""
        if self.description.strip():
            s += "\nDescription: " + self.description
        return s


CARD_STREAM_CHUNK_SIZE = 5


class ChunkedCardStream:
    def __init__(self, items: List[Card], chunk_size: int = CARD_STREAM_CHUNK_SIZE):
        self.items = items
        self.chunk_size = chunk_size
        self.current_index = 0
        self.is_finished = False

    def remaining_chunks(self):
        return math.ceil((len(self.items) - self.current_index) / self.chunk_size)

    def has_next(self):
        return self.current_index < len(self.items)

    def next_chunk(self):
        if not self.has_next():
            return []
        res = self.items[self.current_index: self.current_index + self.chunk_size]
        self.current_index += self.chunk_size
        return res


class SearchBySubstring:
    def __init__(
            self,
            search_substring: str,
            search_in_question: bool,
            search_in_answer: bool,
            case_sensitive: bool,
            fuzzy: Optional[float],
    ):
        self.search_substring = (
            search_substring if not case_sensitive else search_substring.lower()
        )
        self.search_in_question = search_in_question
        self.search_in_answer = search_in_answer
        self.case_sensitive = case_sensitive
        if not (fuzzy is None or 0.0 <= fuzzy <= 1.0):
            raise ValueError("If fuzzy is set, it must be between 0 and 1.")
        self.fuzzy = fuzzy

    def __include_card(self, question, answer):
        if self.fuzzy is None:
            return self.__include_card_hard(question, answer)
        else:
            return self.__include_card_fuzzy(question, answer)

    def __include_card_hard(self, question, answer) -> bool:
        if self.search_in_question:
            search_question = question if self.case_sensitive else question.lower()
            if self.search_substring in search_question:
                return True

        if self.search_in_answer:
            search_answer = answer if self.case_sensitive else answer.lower()
            if self.search_substring in search_answer:
                return True

        return False

    def __fuzzy_search(self, text: str) -> bool:
        return (
                rapidfuzz.fuzz.partial_ratio(self.search_substring, text)
                >= self.fuzzy * 100.0
        )

    def __include_card_fuzzy(self, question, answer) -> bool:
        if self.search_in_question:
            search_question = question if self.case_sensitive else question.lower()
            if self.__fuzzy_search(search_question):
                return True

        if self.search_in_answer:
            search_answer = answer if self.case_sensitive else answer.lower()
            if self.__fuzzy_search(search_answer):
                return True

        return False

    def search_by_substring(self, cards: List[Card]) -> List[Card]:
        """Can use "*" for all decks."""
        if self.fuzzy is None:
            return [c for c in cards if self.__include_card_hard(c.question, c.answer)]
        else:
            return [c for c in cards if self.__include_card_fuzzy(c.question, c.answer)]


class SearchByContent:
    client = openai.OpenAI(api_key="lm-studio", base_url="http://localhost:1234/v1")

    @staticmethod
    def fuzzy_match(
            search_prompt: str, question: Optional[str], answer: Optional[str]
    ) -> bool:
        if question is not None and answer is not None:
            prompt = f"""Please evaluate if the following flash card fits the search prompt.
Question: {question}
Answer: {answer}
Search prompt: {search_prompt}

Please return true if it fits, and else false.
/no_think
"""
        elif question is not None and answer is None:
            prompt = f"""Please evaluate if the following question of a flash card fits the search prompt.
Question: {question}
Search prompt: {search_prompt}

Please return true if it fits, and else false.
/no_think
"""
        elif answer is not None and question is None:
            prompt = f"""Please evaluate if the following answer of a flash card fits the search prompt.
Answer: {answer}
Search prompt: {search_prompt}

Please return true if it fits, and else false.
/no_think
"""
        else:
            raise ValueError("At least one of question or answer must be specified.")

        response = SearchByContent.client.chat.completions.create(
            # model="qwen2.5-14b-instruct"
            model="qwen3-8b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.lower()

        if "false" in result:
            return False
        if "true" in result:
            return True

        raise ValueError(f"Unexpected llm Studio response: {result!r}")


class LLMCommunicator:
    messages: list[dict[str, str]]
    all_messages: list[dict[str, str]]
    model: str
    temperature: float
    max_tokens: Optional[int]
    visibility_block_beginning: Optional[int]

    def __init__(
            self, model: str, temperature: float, max_tokens: Optional[int] = None
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            api_key="lm-studio", base_url="http://localhost:1234/v1"
        )
        self.messages = []
        self.all_messages = []
        self.visibility_block_beginning = None

    def set_system_prompt(self, message: str) -> None:
        request_message = {"role": "system", "content": message}
        self.messages.append(request_message)
        self.all_messages.append(request_message)

    def send_message(self, message: str) -> str:
        self.add_message(message)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,  # add request message here even if it is a hidden message
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        msg = response.choices[0].message
        response_message = {"role": msg.role, "content": msg.content}
        self.messages.append(response_message)
        self.all_messages.append(response_message)
        return msg.content

    def add_message(self, message: str, role="user"):
        request_message = {"role": role, "content": message}
        self.messages.append(request_message)
        self.all_messages.append(request_message)

    # does not cancel a previous block; recalling it doesnt do anything
    def start_visibility_block(self):
        if self.visibility_block_beginning is None:
            self.visibility_block_beginning = len(self.messages)

    def end_visibility_block(self):
        if self.visibility_block_beginning is None:
            return
        self.messages = self.messages[
                        : self.visibility_block_beginning
                        ]  # cut all messages in the visibility block
        self.visibility_block_beginning = None
