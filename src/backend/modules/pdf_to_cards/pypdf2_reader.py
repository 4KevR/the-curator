from typing import Optional

import PyPDF2

from src.backend.modules.pdf_to_cards.AbstractPDFReader import AbstractPDFReader

class PyPDF2Reader(AbstractPDFReader):
    def read(
        self, file_path: str, page_range: Optional[tuple[int, int]] = None
    ) -> dict:
        """Read PDF file.

        :param file_path: PDF file path.
        :param page_range: if None, read all.

        :return: Dict{(int)page_number: (str)text_content}
        """
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                text_content = {}

                # Get page range, first index is 1
                if page_range is None:
                    start, end = 1, num_pages
                else:
                    if page_range[0] > page_range[1]:
                        raise ValueError(
                            "The start page number cannot be "
                            "greater than the end page number"
                        )
                    if page_range[0] > num_pages:
                        raise ValueError(
                            "The starting page number cannot be "
                            "greater than the total number of pages"
                        )
                    else:
                        start = max(1, page_range[0])
                        end = min(page_range[1], num_pages)

                # Extract text content
                for i in range(start, end + 1):
                    text = reader.pages[i - 1].extract_text()
                    # To be improved:
                    # If text == '', maybe this page only has pictures.
                    # Later we may consider calling other models to understand the
                    # pictures and return text descriptions.
                    text_content[i] = text

                return text_content

        except FileNotFoundError:
            print(f"FileNotFound: {file_path}")
            return {}


if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "test.pdf")

    # content = read_pdf(file_path)
    pdf_reader = PyPDF2Reader()
    content = pdf_reader.read_pdf(file_path, (1, 2))
    print(content)
