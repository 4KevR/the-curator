from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.pdf_to_cards.abstract_pdf_reader import AbstractPDFReader


def create_card_generation_prompt(max_cards: int, content: str) -> str:
    return f"""
        Please generate Anki flashcards.
        Requirements:
        1. Use a concise question-and-answer format; each card should include a clear question and an accurate answer;
        2. Questions should be as specific as possible, avoiding vague or broad topics;
        3. Generate no more than {max_cards} cards;
        4. The output format should be as follows:
        Q: ...\nA: ...\n\nQ: ...\nA: ...
        Content:\n{content}
    """.strip()


class CardGeneratorService:
    def __init__(self, pdf_reader: AbstractPDFReader, llm_client: AbstractLLM):
        self.pdf_reader = pdf_reader
        self.llm_client = llm_client

    def create_anki_cards_from_pdf(self, pdf_path: str) -> dict:
        pdf_text_content = self.pdf_reader.read(pdf_path)
        return self.create_anki_cards(pdf_text_content)

    def create_anki_cards(
        self,
        page_content: dict,
        max_cards: int = 3,
    ) -> dict:
        """Create Anki flashcards from a PDF file.
        :param page_content: Dictionary with page numbers and text content.
        :param max_cards: Maximum number of cards to generate per page.
        :return: Dictionary with page numbers as keys and lists of cards as values.
        """
        cards = {}

        for idx, content in page_content.items():
            if content == "":
                continue

            user_prompt = create_card_generation_prompt(
                max_cards=max_cards,
                content=content,
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that is good at " "knowledge extraction.",
                },
                {"role": "user", "content": user_prompt},
            ]

            try:
                raw_output = self.llm_client.generate(messages)
                cards[idx] = self.parse_anki_output(raw_output)

            except Exception as e:
                print(f"[ERROR] Generation of page {idx} failed: {e}")
                cards[idx] = []

        return cards

    def parse_anki_output(self, raw_output: str) -> list:
        """Parse LLM output into structured Anki Q&A card list.

        :param raw_output: Raw output from LLM.

        :return: list of dictionaries [{'question': ..., 'answer': ...}, ...]
        """
        cards = []
        # The original text is divided into blocks with two line breaks \n\n,
        # each block represents a card (usually containing a question and an answer)
        blocks = raw_output.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 2 and lines[0].startswith("Q:") and lines[1].startswith("A:"):
                # Skip the prefixes Q: and A:, and remove whitespace.
                question = lines[0][2:].strip()
                answer = lines[1][2:].strip()
                cards.append({"question": question, "answer": answer})

        return cards
