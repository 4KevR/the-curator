import logging
import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.modules.srs.anki_module.anki_srs import AnkiSRS  # noqa: E402

if __name__ == "__main__":
    load_dotenv(".env")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    username = "test_user"
    myAnki = AnkiSRS(anki_user=username)

    # Deck
    print("\nDeck" + "-" * 80)
    deck_name = "test_deck"
    if myAnki.get_deck_by_name_or_none(deck_name) is None:
        myAnki.add_deck(deck_name)

    myDeck = myAnki.get_deck_by_name(deck_name)
    print(myAnki.deck_exists(myDeck))
    print(myDeck.id, myDeck.name)
    print(myAnki.get_deck_by_id_or_none(myDeck.id))

    myAnki.rename_deck(myDeck, "renemed deck")  # After this, myDeck is invalid.
    myAnki.delete_deck(myAnki.get_deck_by_name("renemed deck"))

    # All decks
    print("\nAll Decks" + "-" * 80)
    all_decks = myAnki.get_all_decks()
    print([(deck.id, deck.name) for deck in all_decks])
    default_Deck = myAnki.get_deck_by_name("Default")
    # In our setup, 'Default' deck is undeletable.
    # When Default is empty, it will be automatically hidden in the main interface;
    myAnki.delete_deck(default_Deck)

    # Card
    print("\nCard" + "-" * 80)
    myCard = myAnki.add_card(myAnki.get_deck_by_name("Default"), "q", "a")
    myAnki.card_exists(myCard)
    print(myAnki.get_card_or_none(myCard.id))
    print(myAnki.get_deck_of_card(myCard).name)

    print("-" * 80)
    myDeck = myAnki.add_deck(deck_name)
    myCard = myAnki.change_deck_of_card(myCard, myDeck)
    print(myAnki.get_card_or_none(myCard.id))

    print("-" * 80)
    myCard = myAnki.get_cards_in_deck(myDeck)[0]
    myCard = myAnki.edit_card_question(myCard, new_question="new q")
    myCard = myAnki.edit_card_answer(myCard, new_answer="new a")
    print(myCard)

    # Learn
    print("\nLearn" + "-" * 80)
    myAnki.set_memory_grade(myCard.id.numeric_id, "hard")
    myAnki.set_flag(myCard.id.numeric_id, "red")
    myAnki.set_flag(myCard.id.numeric_id, 2)  # overwrite flag = orange

    print("-" * 80)
    print(myAnki.count_cards_due_today(myDeck.name))
    myAnki.activate_preview_cards(myDeck.name)
    print(myAnki.count_cards_due_today(myDeck.name))

    # Note
    print("\nNote" + "-" * 80)
    print(myAnki.list_all_notes())
    print(myAnki.list_notes_for_cards_in_deck(myDeck.name))
    nid = myAnki.get_note_id_by_card_id(myCard.id.numeric_id)
    myAnki.list_card_ids_from_note(nid)
    myAnki.delete_notes_by_ids([nid])

    print("-" * 80)
    myAnki.export_deck_to_apkg(myDeck)
    myAnki.delete_deck(myDeck)
    myAnki.import_deck_from_apkg("path")
