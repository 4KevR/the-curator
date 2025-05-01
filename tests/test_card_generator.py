import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from card_generator import read_pdf, create_anki_cards

if __name__ == "__main__":
    path = 'card_generator/test.pdf' # 6 pages, the last page is blank
    text = read_pdf(path)
    cards = create_anki_cards(text)
    print(cards)
