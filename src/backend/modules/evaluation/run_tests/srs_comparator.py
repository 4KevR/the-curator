from rapidfuzz.distance import Levenshtein

from src.backend.modules.evaluation.run_tests.llm_similarity_judge import LLMSimilarityJudge
from src.backend.modules.helpers.matching import match_by_equals, match_by_key, match_by_tolerance
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.srs.testsrs.testsrs import TestCard, TestFlashcardManager


class SRSComparator:
    def __init__(self, llm_for_fuzzy_matching: AbstractLLM, llm_judge=LLMSimilarityJudge):
        self.llm_for_fuzzy_matching = llm_for_fuzzy_matching
        self.llm_judge = llm_judge

    def _compare_decks(
        self,
        expected: list[TestCard],
        actual: list[TestCard],
        levenshtein_distance: int | None,
        levenshtein_factor: float | None,
    ) -> list[str]:
        """
        Compares two decks. The two decks must be of the same type.

        If neither levenshtein_distance nor levenshtein_factor is set, only exact matches are used.
        If both are set, two cards only match if they pass *both* thresholds.

        Parameters:
            levenshtein_distance: If set, the maximum distance between question/answer strings to be considered a match.
            levenshtein_factor: If set, the maximum ratio (levenshtein distance / max(question length, answer length)
                     to be considered a match. Should be in the range [0, 1].
        """
        exp_strict = [x for x in expected if not x.fuzzymatch_question and not x.fuzzymatch_answer]
        exp_fuzzy = [x for x in expected if x.fuzzymatch_question or x.fuzzymatch_answer]

        # match exact
        # since only non-matched cards matter, matches are ignored (_).
        (_, unm_exp, tmp_unm_act) = match_by_key(
            exp_strict,
            actual,
            equals=lambda x, y: x.to_hashable() == y.to_hashable(),
            left_key=lambda x: x.to_hashable(),
            right_key=lambda x: x.to_hashable(),
        )

        # If enabled, use levenshtein distance to match cards
        if levenshtein_distance is not None or levenshtein_factor is not None:

            def tolerance_function(l: TestCard, r: TestCard) -> bool:
                if l.state != r.state or l.flag != r.flag:
                    return False

                dist_question = Levenshtein.distance(l.question, r.question)
                dist_answer = Levenshtein.distance(l.answer, r.answer)

                if levenshtein_distance is not None:
                    if dist_question > levenshtein_distance or dist_answer > levenshtein_distance:
                        return False
                if levenshtein_factor is not None:
                    max_len = max(len(l.question), len(r.question))
                    ratios = [dist_question / max_len > levenshtein_factor, dist_answer / max_len > levenshtein_factor]
                    if any(ratios):
                        return False
                return True

            (_, unm_exp, tmp_unm_act) = match_by_tolerance(unm_exp, tmp_unm_act, tolerance_function)

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

    def compare_srs(
        self,
        expected: TestFlashcardManager,
        actual: TestFlashcardManager,
        levenshtein_distance: int | None,
        levenshtein_factor: float | None,
    ) -> list[str]:
        """
        Compares two SRSs by first matching decks, then comparing cards within each deck.

        If neither levenshtein_distance nor levenshtein_factor is set, only exact matches are used.
        If both are set, two cards only match if they pass *both* thresholds.

        Parameters:
            actual: The actual SRS.
            expected: The expected SRS.
            levenshtein_distance: If set, the maximum distance between question/answer strings to be considered a match.
            levenshtein_factor: If set, the maximum ratio (levenshtein distance / max(question length, answer length)
                     to be considered a match. Should be in the range [0, 1].
        """
        (matched, unmatched_expected, unmatched_actual) = match_by_key(
            expected.get_all_decks(),
            actual.get_all_decks(),
            equals=(lambda x, y: x.name == y.name),  # checking by name is sufficient, as names must be unique
            left_key=lambda l_k: l_k.name,
            right_key=lambda r_k: r_k.name,
        )

        errors: list[str] = []
        for unmatched_expected_deck in unmatched_expected:
            errors += [f"The deck {unmatched_expected_deck.name} was expected, but was not in the actual result."]

        for unmatched_actual_deck in unmatched_actual:
            errors += [f"The deck {unmatched_actual_deck.name} was in the actual result, but was unexpected."]

        for e, a in matched:
            errors += self._compare_decks(
                expected.get_cards_in_deck(e),
                actual.get_cards_in_deck(a),
                levenshtein_distance,
                levenshtein_factor,
            )

        return errors
