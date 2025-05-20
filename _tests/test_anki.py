import logging
import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.adapter import Anki

if __name__ == "__main__":
    load_dotenv(".env")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    username = "test_user"
    myAnki = Anki(user_name=username)
    myAnki.add_deck("test_deck")
    myAnki.add_note("test_deck", "1", "1", "Basic")
    myAnki.add_note("test_deck", "2", "2", "Basic")
    myAnki.add_note("test_deck", "3", "3", "Basic (and reversed card)")

    print(myAnki.list_all_notes())
    print(myAnki.list_cards_in_deck("test_deck"))

    myAnki.activate_preview_cards("test_deck")
    print(myAnki.count_cards_due_today("test_deck"))
    # myAnki.delete_deck("test_deck")
