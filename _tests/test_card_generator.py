import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.adapter.kit_llm import KitLLM
from src.backend.adapter.pypdf2_reader import PyPDF2Reader
from src.backend.service import CardGeneratorService

if __name__ == "__main__":
    path = "data/test.pdf"  # 6 pages, the last page is blank
    pdf_reader = PyPDF2Reader()
    llm_client = KitLLM()
    card_generator = CardGeneratorService(pdf_reader, llm_client)
    cards = card_generator.create_anki_cards_from_pdf(path)
    print(cards)
    text_content = {
        1: "Named Entity Recognition (NER) is an NLP task that identifies and "
        "classifies named entities in text into categories such as person names, "
        "organizations, locations, dates, and more.",
        2: "Machine learning is a branch of artificial intelligence that allows "
        "computers to automatically improve their performance through data.",
    }
    cards = card_generator.create_anki_cards(text_content)
    print(cards)
