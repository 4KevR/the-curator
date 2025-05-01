# Meta-Llama-3.1-8B-Instruct
# The above model is deployed locally in LM Studio.
# First check the port usage, the default port is 1234.
# Then start the server using "lms server start"

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)


def create_anki_cards(pdf_text_content: dict,
    model: str = "meta-llama-3.1-8b-instruct",
    max_cards: int = 3
) -> dict:
    """Generate Anki cards using local LLaMA model

    :param pdf_text_content: dictionary, Dict{(int)page_number: (str)text_content}
    :param model: model name, you can find it in lm-studio
    :param max_cards: maximum number of cards generated for each page
    
    :return: Dict{(int)page_number: List[Dict{Q&A}]}
    """
    cards = {}

    for idx, content in pdf_text_content.items():
        if content == "": continue

        user_prompt = (
            f"Please generate Anki flashcards.\n\n"
            f"Requirements:\n"
            f"1. Use a concise question-and-answer format; each card should include a clear question and an accurate answer;\n"
            f"2. Questions should be as specific as possible, avoiding vague or broad topics;\n"
            f"3. Generate no more than {max_cards} cards;\n"
            f"4. The output format should be as follows:\n"
            f"Q: ...\nA: ...\n\nQ: ...\nA: ...\n\n"
            f"Content:\n{content}"
        )

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that is good at knowledge extraction."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            raw_output = response.choices[0].message.content
            cards[idx] = parse_anki_output(raw_output)

        except Exception as e:
            print(f"[ERROR] Generation of page {idx} failed: {e}")
            cards[idx] = []

    return cards


def parse_anki_output(raw_output: str) -> list:
    """Parse LLM output into structured Anki Q&A card list.
    
    :param output: raw output text from model

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
            cards.append({
                "question": question,
                "answer": answer
            })

    return cards




if __name__ == '__main__':
    text_content = {
        1: "Named Entity Recognition (NER) is an NLP task that identifies and classifies named entities in text into categories such as person names, organizations, locations, dates, and more.",
        2: "Machine learning is a branch of artificial intelligence that allows computers to automatically improve their performance through data."
    }
    cards = create_anki_cards(text_content)
    print(cards)