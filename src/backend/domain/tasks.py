from enum import Enum


class AnkiTasks(Enum):
    CREATE_DECK = "create_deck"
    MODIFY_DECK = "modify_deck"
    DELETE_DECK = "delete_deck"
    GET_DECK_STATISTICS = "get_deck_statistics"
    ADD_CARD = "add_card"
    MODIFY_CARD = "modify_card"
    DELETE_CARD = "delete_card"
    LEARN_DECK = "learn_deck"
