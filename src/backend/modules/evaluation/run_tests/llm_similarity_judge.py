from src.backend.modules.helpers.string_util import find_substring_in_llm_response
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.srs.testsrs.testsrs import TestCard


class LLMSimilarityJudge:

    def __init__(self, judge_llm: AbstractLLM):
        self.judge_llm = judge_llm

    def judge_answer_similarity(self, expected: str, actual: str) -> bool:
        """
        Use the LLM to judge if the actual answer to a question-answering-test is similar enough to the expected answer.
        """
        prompt = f"""Please evaluate the following two answers, and tell me if they contain the same information.
        Ignore differences in grammar, length, or wording, as long as the answers are semantically equivalent.
        Also similar responses are accepted, as phrasing might be very different.
        If they are similar, end your response with "true", else with "false" (without quotation marks).
        Only the last word of your response will be evaluated.

Expected answer:
{expected}

Actual answer:
{actual}

Remember to only respond with 'true' or 'false'.
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.judge_llm.generate(messages)

        return find_substring_in_llm_response(response, "true", "false")

    def judge_card_similarity(self, expected_card: TestCard, actual_card: TestCard) -> bool:
        """
        Match two cards by their content using a llm as a judge.

        If expected_card.fuzzymatch_question is false, uses hard matching on question.
        If expected_card.fuzzymatch_answer is false, uses hard matching on answer.
        """
        required = [
            expected_card.cardState == actual_card.cardState,
            expected_card.flag == actual_card.flag,
            expected_card.fuzzymatch_question or expected_card.question == actual_card.question,
            expected_card.fuzzymatch_answer or expected_card.answer == actual_card.answer,
        ]
        if not all(required):
            return False

        prompt = f"""Please evaluate the following two flashcards, and tell me if they have the same content.
        It is fine if the spelling, the grammar, the length and the wording differs,
        as long as the cards contain roughly the same information.
        Punctuation or enclosing quotation marks are irrelevant.
        If these cards are quite similar, please end your response with "true",
        else with "false" (without quotation marks). Only the last word of your response will be evaluated.

Card 1:
Question: {expected_card.question}
Answer: {expected_card.answer}

Card 2:
Question: {actual_card.question}
Answer: {actual_card.answer}

Remember to only respond with 'true' or 'false'.
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.judge_llm.generate(messages)

        return find_substring_in_llm_response(response, "true", "false")
