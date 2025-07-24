from src.backend.modules.helpers.string_util import find_substring_in_llm_response, remove_quots
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.srs.testsrs.testsrs import TestCard


class LLMSimilarityJudge:

    def __init__(self, judge_llm: AbstractLLM):
        self.judge_llm = judge_llm

    def judge_answer_similarity(self, expected: str, actual: str) -> bool:
        """
        Use the LLM to judge if the actual answer to a question-answering-test is similar enough to the expected answer.
        """
        prompt = f"""I have two answers to a task.
Expected answer:
{expected}

Actual answer:
{actual}

Please tell me if the actual answer contains – at least – the same information as the expected answer.
It is fine if the actual answer contains a little more information than the expected answer.
Ignore differences in grammar, length, or wording, as long as the answers are semantically equivalent.

If they are similar enough, answer true, else false. **Answer nothing else.**
"""
        response = self.judge_llm.generate_single(prompt)

        return find_substring_in_llm_response(response, "true", "false")

    def judge_card_similarity(self, expected_card: TestCard, actual_card: TestCard) -> bool:
        """
        Match two cards by their content using a llm as a judge.

        If expected_card.fuzzymatch_question is false, uses hard matching on question.
        If expected_card.fuzzymatch_answer is false, uses hard matching on answer.
        """
        required = [
            expected_card.state == actual_card.state,
            expected_card.flag == actual_card.flag,
            expected_card.fuzzymatch_question
            or remove_quots(expected_card.question) == remove_quots(actual_card.question),
            expected_card.fuzzymatch_answer or remove_quots(expected_card.answer) == remove_quots(actual_card.answer),
        ]
        if not all(required):
            return False

        prompt = f"""Please evaluate the following two flashcards, and tell me if they have the same content.
It is fine if the spelling, the grammar, the length and the wording differs, as long as the cards contain roughly the same information.
Punctuation is irrelevant.

Card 1:
Question: {remove_quots(expected_card.question)}
Answer: {remove_quots(expected_card.answer)}

Card 2:
Question: {remove_quots(actual_card.question)}
Answer: {remove_quots(actual_card.answer)}

Answer true if the cards have similar content. Also answer true if Card 2 has more content than card 1.

Only respond with 'true' or 'false'!
"""
        response = self.judge_llm.generate_single(prompt)

        return find_substring_in_llm_response(response, "true", "false")
