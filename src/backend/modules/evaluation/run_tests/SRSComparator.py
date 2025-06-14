from src.backend.modules.asr.cloud_lecture_translator import CloudLectureTranslatorASR
from src.backend.modules.evaluation.run_tests.LLMSimilarityJudge import LLMSimilarityJudge
from src.backend.modules.helpers.matching import match_by_equals, match_by_key
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.srs.testsrs.testsrs import TestCard, TestFlashcardManager


class SRSComparator:
    def __init__(self, llm_for_fuzzy_matching: AbstractLLM, llm_judge=LLMSimilarityJudge):
        self.llm_for_fuzzy_matching = llm_for_fuzzy_matching
        self.llm_judge = llm_judge
        super().__init__(llm_for_fuzzy_matching, CloudLectureTranslatorASR())

    def _compare_decks(self, expected: list[TestCard], actual: list[TestCard]) -> list[str]:
        """
        Compares two decks. The two decks must be of the same type.
        """
        exp_strict = [x for x in expected if not x.fuzzymatch_question and not x.fuzzymatch_answer]
        exp_fuzzy = [x for x in expected if x.fuzzymatch_question or x.fuzzymatch_answer]

        # match exact
        (_, unm_exp, tmp_unm_act) = match_by_key(
            exp_strict,
            actual,
            equals=lambda x, y: x.to_hashable() == y.to_hashable(),
            left_key=lambda x: x.to_hashable(),
            right_key=lambda x: x.to_hashable(),
        )

        (_, unm_exp_fuzzy, final_unm_act) = match_by_equals(
            exp_fuzzy, tmp_unm_act, equals=self.llm_judge.judge_card_similarity
        )

        # Now create the error messages
        errors = []
        for additional_fuzzy in unm_exp_fuzzy:
            errors += [f"The following expected, fuzzy-matching card has not found a partner:\n{additional_fuzzy}"]

        for additional_expected in unm_exp:
            errors += [f"The following expected card has not found a partner:\n{additional_expected}"]

        for additional_actual in final_unm_act:
            errors += [f"The following provided card was not expected:\n{additional_actual}"]
        return errors

    def compare_srs(self, expected: TestFlashcardManager, actual: TestFlashcardManager) -> list[str]:
        (matched, unmatched_expected, unmatched_actual) = match_by_key(
            expected.get_all_decks(),
            actual.get_all_decks(),
            equals=(lambda x, y: x.name == y.name),  # checking by name is sufficient, as names must be unique
            left_key=lambda l: l.name,
            right_key=lambda r: r.name,
        )

        errors: list[str] = []
        for unmatched_expected_deck in unmatched_expected:
            errors += [f"The deck {unmatched_expected_deck.name} was expected, but was not in the actual result."]

        for unmatched_actual_deck in unmatched_actual:
            errors += [f"The deck {unmatched_actual_deck.name} was in the actual result, but was unexpected."]

        for e, a in matched:
            errors += self._compare_decks(expected.get_cards_in_deck(e), actual.get_cards_in_deck(a))

        return errors
