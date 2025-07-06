import logging
import os
import sys

from dotenv import load_dotenv

# test that the current working dir is "the-curator"
if os.path.basename(os.path.abspath(".")) != "the-curator":
    raise RuntimeError("This script must be run from the 'the-curator' directory.")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.backend.modules.srs.abstract_srs import CardState, Flag  # noqa: E402
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
    assert myAnki.deck_exists(myDeck)
    assert myDeck.name == deck_name
    assert myAnki.get_deck_by_id_or_none(myDeck.id) is not None

    new_name = "renamed deck"
    myAnki.rename_deck(myDeck, new_name)  # After this, myDeck is invalid.
    assert myAnki.get_deck_by_name_or_none(new_name) is not None
    assert myAnki.get_deck_by_name_or_none(deck_name) is None

    myAnki.delete_deck(myAnki.get_deck_by_name(new_name))
    assert myAnki.get_deck_by_name_or_none(new_name) is None

    # All decks
    print("\nAll Decks" + "-" * 80)
    all_decks = myAnki.get_all_decks()
    print([(deck.id, deck.name) for deck in all_decks])
    default_Deck = myAnki.get_deck_by_name("Default")

    # In our setup, 'Default' deck is undeletable.
    # When Default is empty, it will be automatically hidden in the main interface;
    try:
        myAnki.delete_deck(default_Deck)
        raise RuntimeError("Exception expected!")
    except ValueError as e:
        print(e)

    # Card
    myCard = myAnki.add_card(myAnki.get_deck_by_name("Default"), "q", "a", flag=Flag.NONE, state=CardState.NEW)
    # myCard = myAnki.add_card(myAnki.get_deck_by_name("Default"), "q", "a", flag=Flag.PINK, state=CardState.LEARNING)
    assert myAnki.card_exists(myCard)
    assert myAnki.get_card_or_none(myCard.id).to_hashable() == myCard.to_hashable()
    assert myAnki.get_deck_of_card(myCard).name == "Default"

    myDeck = myAnki.add_deck(deck_name)
    myCard = myAnki.change_deck_of_card(myCard, myDeck)
    assert myAnki.get_deck_of_card(myCard).name == deck_name

    myCard = myAnki.get_cards_in_deck(myDeck)[0]
    myCard = myAnki.edit_card_question(myCard, new_question="new q")
    myCard = myAnki.edit_card_answer(myCard, new_answer="new a")
    assert myCard.question == "new q"
    assert myCard.answer == "new a"

    for flag in Flag:
        myCard = myAnki.edit_card_flag(myCard, flag)
        assert myCard.flag == flag

    for card_state in CardState:
        myCard = myAnki.edit_card_state(myCard, card_state)
        assert myCard.state == card_state

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
    myAnki.import_deck_from_apkg("my_path")
