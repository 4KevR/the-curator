import logging
import os
import sys
import datetime

from dotenv import load_dotenv

from src.backend.modules.srs.abstract_srs import CardID
from src.backend.modules.srs.anki.anki import AnkiSRS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    load_dotenv(".env")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    username = "test_user"
    myAnki = AnkiSRS(anki_directory=username)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Current time: {current_time}")

    myDeck = myAnki.add_deck(f"test_deck {current_time}")
    myAnki.add_note(myDeck, "front: 1", "back: eins", AnkiSRS.NoteType.BASIC)
    myAnki.add_note(myDeck, "front: 2", "back: zwei", AnkiSRS.NoteType.BASIC)
    res = myAnki.add_note(myDeck, "front: 3", "back: drei", AnkiSRS.NoteType.BASIC_REVERSED)

    print(myAnki.get_card(CardID(res.cards[0].id)))
    print(myAnki.get_card(CardID(res.cards[1].id)))

    print(myAnki.list_all_notes())
    print(myAnki.get_cards_in_deck(myDeck))

    myAnki.activate_preview_cards(myDeck.name)
    print(myAnki.count_cards_due_today(myDeck.name))
    # myAnki.delete_deck("test_deck")
