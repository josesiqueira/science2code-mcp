"""Attacks on the one property, and the regressions for the four that landed.

The property under attack: a paraphrase, a fabricated quote, or a passage from
a different paper must never be returned in a way a caller could present as
character-identical with the document.

Every test here was written by trying to break that, running the attack against
the real code, and keeping the input that worked. Four attacks worked and are
fixed; the rest did not, and are kept because an attack that fails is the
evidence the property holds, and it stays evidence only while it keeps failing.

Inline fixtures, no corpus on disk, no MCP client, same as the rest of the
suite. The fake corpus is imported from `test_server` rather than copied, so
the two cannot drift apart.
"""

from __future__ import annotations

import unittest

from science2code import envelope as env
from science2code import server
from science2code.anchor import (
    T_LOCATE_EQUIVALENT_RANGE,
    Tier,
    _char_diff,
    locate,
    prepare,
)
from science2code.envelope import EnvelopeViolation, Outcome
from science2code.normalise import match_form, normalise
from tests.test_server import FakeCorpus

# Every non-ASCII character in this file is written as an escape. An attack on
# a normaliser is made of characters that are invisible, or that look exactly
# like another character, so a literal would be unreadable in review and could
# be silently eaten by an editor that trims trailing invisibles.
ZWSP = "\u200b"  # ZERO WIDTH SPACE
SOFT_HYPHEN = "\u00ad"  # SOFT HYPHEN
SUPER_SIX = "\u2076"  # SUPERSCRIPT SIX
SUB_TWO = "\u2082"  # SUBSCRIPT TWO
HALF = "\u00bd"  # VULGAR FRACTION ONE HALF
CYRILLIC_A = "\u0430"  # CYRILLIC SMALL LETTER A, a homoglyph of Latin "a"
RLO = "\u202e"  # RIGHT-TO-LEFT OVERRIDE
LIGATURE_FI = "\ufb01"  # LATIN SMALL LIGATURE FI, which folds safely to "fi"


PAPER_A = (
    "Regulatory sandboxes are established under Article 57 of the Regulation. "
    "They provide a controlled environment for the development and testing of "
    "innovative AI systems prior to their placement on the market. "
    "The model does not improve accuracy on the held-out set. "
    "We report a mean absolute error of 0.412 across 1024 trials. "
    "The study was carried out by Hasan and colleagues in 2016. "
    "Throughput reached 10" + SUPER_SIX + " operations per second. "
    "The reaction was run in H" + SUB_TWO + "O at room temperature. "
    "About " + HALF + " of the sample was discarded before analysis. "
    "The participants were asked to re-sign the consent form after the change. "
    "The " + LIGATURE_FI + "nal section was surveyed in March by the WHO."
)

PAPER_B = (
    "A risk based approach is applied across the whole of the Regulation. "
    "Nothing in this second paper concerns sandboxes at all. "
    "It is about compiler optimisation and register allocation on small "
    "embedded targets with very little memory to spare."
)

TRUE_PASSAGE_OF_A = (
    "They provide a controlled environment for the development and testing of "
    "innovative AI systems"
)

PARAPHRASES_OF_A = {
    "synonym_swap": (
        "They offer a controlled setting for the development and testing of "
        "innovative AI systems"
    ),
    "second_synonym_swap": (
        "They supply a managed environment for the creation and evaluation of "
        "novel AI systems"
    ),
    "clause_reorder": (
        "For the development and testing of innovative AI systems they provide "
        "a controlled environment"
    ),
    "function_words_elided": (
        "They provide controlled environment for development and testing of "
        "innovative AI systems"
    ),
    "tense_shift": (
        "They provided a controlled environment for the development and "
        "testing of innovative AI systems"
    ),
    "nominalised": (
        "A controlled environment is provided by them for developing and "
        "testing innovative AI systems"
    ),
}

NEAR_MISSES_OF_A = {
    "one_digit_changed": "We report a mean absolute error of 0.512 across 1024 trials",
    "trial_count_changed": "We report a mean absolute error of 0.412 across 1025 trials",
    "negation_dropped": "The model does improve accuracy on the held-out set",
    "negation_added": "The provider does not retain responsibility for the sandbox",
    "author_flipped": "The study was carried out by Akhigbe and colleagues in 2016",
    "year_flipped": "The study was carried out by Hasan and colleagues in 2019",
}

IDENTITY_TIERS = (Tier.T1_EXACT, Tier.T2_RELAXED)
IDENTITY_OUTCOMES = {o.value for o in env.VERBATIM_OUTCOMES}

#: 0.00, 0.05, ... 1.00. The two identity tiers are threshold free, so no value
#: here may change whether one is reached. Sweeping is how that is checked
#: rather than assumed.
THRESHOLDS = tuple(round(i * 0.05, 2) for i in range(21))


class AdversarialTestCase(unittest.TestCase):
    """A two paper corpus, restored around every test."""

    DOCS = {"paper-a": PAPER_A, "paper-b": PAPER_B}

    def setUp(self):
        self._real_load = server._load_corpus
        self._real_env = dict(server.os.environ)
        self.use_corpus(FakeCorpus(dict(self.DOCS)))
        server._PREPARED_CACHE.clear()

    def tearDown(self):
        server._load_corpus = self._real_load
        server.os.environ.clear()
        server.os.environ.update(self._real_env)
        server._PREPARED_CACHE.clear()

    def use_corpus(self, corpus):
        server._load_corpus = lambda root: corpus

    def tiers_over_the_threshold_sweep(self, document: str, quote: str) -> set[Tier]:
        prepared = prepare(document)
        return {locate(prepared, quote, t_locate=t).tier for t in THRESHOLDS}


# ---------------------------------------------------------------------------
# attacks that did NOT break it, kept as the evidence that it holds
# ---------------------------------------------------------------------------


class TestParaphraseCannotReachCharacterIdentity(AdversarialTestCase):
    """Attack: rewrite a real passage and try to get it called identical."""

    def test_no_paraphrase_reaches_an_identity_tier_at_any_threshold(self):
        for name, quote in PARAPHRASES_OF_A.items():
            with self.subTest(paraphrase=name):
                tiers = self.tiers_over_the_threshold_sweep(PAPER_A, quote)
                self.assertEqual(
                    set(),
                    tiers & set(IDENTITY_TIERS),
                    "%s reached an identity tier somewhere in 0.0 to 1.0" % name,
                )

    def test_the_true_passage_still_does_reach_one(self):
        # A guard on the test above: a check that nothing can pass is worthless
        # if the thing that should pass does not.
        tiers = self.tiers_over_the_threshold_sweep(PAPER_A, TRUE_PASSAGE_OF_A)
        self.assertEqual({Tier.T1_EXACT}, tiers)

    def test_a_paraphrase_comes_back_from_the_tool_as_a_refusal(self):
        for name, quote in PARAPHRASES_OF_A.items():
            with self.subTest(paraphrase=name):
                result = server.verify_quote(quote, paper_id="paper-a")
                self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES, name)

    def test_a_paraphrase_is_never_a_find_passage_hit(self):
        for name, quote in PARAPHRASES_OF_A.items():
            with self.subTest(paraphrase=name):
                result = server.find_passage(quote)
                self.assertEqual(0, result["hit_count"], name)


class TestCrossDocumentAttribution(AdversarialTestCase):
    """Attack: a real passage of paper A, asked about paper B."""

    def test_a_real_passage_of_one_paper_is_not_identical_with_another(self):
        tiers = self.tiers_over_the_threshold_sweep(PAPER_B, TRUE_PASSAGE_OF_A)
        self.assertEqual(set(), tiers & set(IDENTITY_TIERS))

    def test_the_tool_refuses_it_against_the_other_paper(self):
        result = server.verify_quote(TRUE_PASSAGE_OF_A, paper_id="paper-b")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        self.assertNotIn("document_text", result)

    def test_find_passage_names_only_the_paper_that_holds_it(self):
        result = server.find_passage(TRUE_PASSAGE_OF_A)
        self.assertEqual(["paper-a"], [h["paper_id"] for h in result["hits"]])

    def test_a_paraphrase_of_one_paper_is_not_located_in_the_other(self):
        for name, quote in PARAPHRASES_OF_A.items():
            with self.subTest(paraphrase=name):
                result = server.verify_quote(quote, paper_id="paper-b")
                self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES, name)


class TestAdversarialNearMisses(AdversarialTestCase):
    """Attack: high similarity, opposite meaning. The dangerous ones."""

    def test_a_near_miss_is_relocated_and_never_identical(self):
        for name, quote in NEAR_MISSES_OF_A.items():
            with self.subTest(near_miss=name):
                result = server.verify_quote(quote, paper_id="paper-a")
                self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES, name)

    def test_a_near_miss_that_relocates_always_carries_the_diff(self):
        for name, quote in NEAR_MISSES_OF_A.items():
            with self.subTest(near_miss=name):
                result = server.verify_quote(quote, paper_id="paper-a")
                if result["outcome"] != Outcome.PASSAGE_RELOCATED_QUOTE_DIFFERS.value:
                    continue
                self.assertIn("char_diff", result)
                self.assertIn("document_text", result)
                self.assertIn("your_text", result)
                self.assertNotEqual(result["document_text"], result["your_text"])

    def test_a_dropped_negation_shows_the_missing_word_in_the_diff(self):
        result = server.verify_quote(
            NEAR_MISSES_OF_A["negation_dropped"], paper_id="paper-a"
        )
        self.assertEqual(
            Outcome.PASSAGE_RELOCATED_QUOTE_DIFFERS.value, result["outcome"]
        )
        self.assertIn("not", result["char_diff"])
        self.assertIn("does not improve", result["document_text"])

    def test_a_changed_digit_is_not_absorbed(self):
        result = server.verify_quote(
            NEAR_MISSES_OF_A["one_digit_changed"], paper_id="paper-a"
        )
        self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES)
        self.assertIn("0.412", result["document_text"])

    def test_a_flipped_author_name_is_not_absorbed(self):
        result = server.verify_quote(
            NEAR_MISSES_OF_A["author_flipped"], paper_id="paper-a"
        )
        self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES)
        self.assertIn("Hasan", result["document_text"])


class TestHomoglyphsAndInvisibles(AdversarialTestCase):
    """Attack: two strings that look identical and are not, and the reverse."""

    def test_a_cyrillic_homoglyph_does_not_reach_an_identity_tier(self):
        quote = "Regulatory sandboxes are established under Article 57".replace(
            "a", CYRILLIC_A
        )
        tiers = self.tiers_over_the_threshold_sweep(PAPER_A, quote)
        self.assertEqual(set(), tiers & set(IDENTITY_TIERS))
        result = server.verify_quote(quote, paper_id="paper-a")
        self.assertEqual(
            Outcome.PASSAGE_RELOCATED_QUOTE_DIFFERS.value, result["outcome"]
        )

    def test_a_right_to_left_override_does_not_reach_an_identity_tier(self):
        quote = "Regulatory sandboxes are " + RLO + "established under Article 57"
        result = server.verify_quote(quote, paper_id="paper-a")
        self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES)

    def test_a_ligature_the_extractor_left_in_still_reaches_identity(self):
        # The safe direction of the same fold, kept so a future tightening
        # cannot quietly break what NFKC is actually there for.
        result = server.verify_quote(
            "The final section was surveyed in March", paper_id="paper-a"
        )
        self.assertIn(result["outcome"], IDENTITY_OUTCOMES)
        self.assertIn(LIGATURE_FI, result["document_text"])


class TestDegenerateAndEmptyInputs(AdversarialTestCase):
    """Attack: nothing, whitespace, one character, and more than everything."""

    def test_an_empty_string_is_not_locatable(self):
        result = server.verify_quote("", paper_id="paper-a")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        self.assertIsNone(result["char_interval"])

    def test_whitespace_only_is_not_locatable_however_long(self):
        result = server.verify_quote(" \t\n" * 40, paper_id="paper-a")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])

    def test_a_single_character_is_not_locatable(self):
        result = server.verify_quote("a", paper_id="paper-a")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])

    def test_a_quote_longer_than_the_document_never_reaches_identity(self):
        result = server.verify_quote(PAPER_A * 2, paper_id="paper-a")
        self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES)

    def test_the_whole_document_is_identical_with_itself(self):
        result = server.verify_quote(PAPER_A, paper_id="paper-a")
        self.assertEqual(Outcome.VERBATIM_EXACT.value, result["outcome"])

    def test_no_degenerate_input_produces_a_liftable_field(self):
        for label, quote in (
            ("empty", ""),
            ("whitespace", "   "),
            ("one character", "a"),
            ("longer than the document", PAPER_A * 2),
        ):
            with self.subTest(input=label):
                result = server.verify_quote(quote, paper_id="paper-a")
                for path, key, _value in env._walk(result):
                    self.assertNotIn(key, env.QUOTE_FIELD_NAMES, "%s at %s" % (label, path))


class TestFieldNameEvasion(AdversarialTestCase):
    """Attack: get the document's words out under a name that is not banned."""

    def test_no_refusal_from_any_attack_carries_a_liftable_field_name(self):
        quotes = list(PARAPHRASES_OF_A.values()) + list(NEAR_MISSES_OF_A.values())
        for quote in quotes:
            for response in (
                server.verify_quote(quote, paper_id="paper-a"),
                server.verify_quote(quote),
                server.find_passage(quote),
            ):
                for path, key, _value in env._walk(response):
                    self.assertNotIn(key, env.QUOTE_FIELD_NAMES, path)

    def test_a_liftable_field_two_levels_deep_in_a_list_is_still_rejected(self):
        # The ban has to hold on nested structures and on list members, not
        # only on top level keys.
        response = server.find_passage(TRUE_PASSAGE_OF_A)
        response["hits"][0]["neighbours"] = [{"passage": "something to paste"}]
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_not_locatable_still_carries_no_words_of_the_document(self):
        result = server.verify_quote(
            "a sentence that appears in neither of these two papers anywhere",
            paper_id="paper-a",
        )
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        self.assertNotIn("document_text", result)
        self.assertNotIn("char_diff", result)


class TestT2RelaxationIsCaseAndHyphenBlindByDesign(AdversarialTestCase):
    """Not a defect, and pinned here so that stays a decision rather than luck.

    T2 folds case and deletes intra-word hyphens, so "resign" reaches it
    against a document that says "re-sign", and "who" against one that says
    "WHO". Those are real meaning changes and the ladder accepts them, on the
    measured ground that the extractor eats a compound hyphen at a line break
    far more often than a quote flips one. What makes it safe to accept is the
    field the caller is handed back: `document_text` is the document's own
    form, never the caller's, so the string available to paste is the paper's.
    That is the load-bearing part, and it is what these tests pin.
    """

    def test_a_deleted_hyphen_reaches_the_relaxed_tier(self):
        result = server.verify_quote(
            "participants were asked to resign the consent form", paper_id="paper-a"
        )
        self.assertEqual(
            Outcome.VERBATIM_RELAXED_EXTRACTOR_DAMAGE.value, result["outcome"]
        )

    def test_and_hands_back_the_document_own_hyphenated_form(self):
        result = server.verify_quote(
            "participants were asked to resign the consent form", paper_id="paper-a"
        )
        self.assertIn("re-sign", result["document_text"])
        self.assertNotIn("your_text", result)

    def test_a_case_change_reaches_the_relaxed_tier(self):
        result = server.verify_quote(
            "final section was surveyed in march by the who", paper_id="paper-a"
        )
        self.assertEqual(
            Outcome.VERBATIM_RELAXED_EXTRACTOR_DAMAGE.value, result["outcome"]
        )

    def test_and_hands_back_the_document_own_casing(self):
        result = server.verify_quote(
            "final section was surveyed in march by the who", paper_id="paper-a"
        )
        self.assertIn("March", result["document_text"])
        self.assertIn("WHO", result["document_text"])


# ---------------------------------------------------------------------------
# attacks that DID break it. One regression class per defect.
# ---------------------------------------------------------------------------


class TestZeroWidthPaddingCannotBuyAnIdentityOutcome(AdversarialTestCase):
    """Defect: the length floor was measured on the raw string.

    `MIN_LOCATABLE_CHARS` exists because a string under twelve characters
    occurs in too many places to name one of them. It was measured before
    normalisation, and the normaliser deletes zero-width characters and soft
    hyphens outright, so padding a three character string with twenty ZERO
    WIDTH SPACE characters walked straight past the floor and came back as
    VERBATIM_EXACT with a character interval naming one occurrence of the word
    "not" in one paper.
    """

    def test_the_bare_short_string_is_refused(self):
        result = server.verify_quote("not", paper_id="paper-a")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])

    def test_zero_width_padding_does_not_buy_an_identity_outcome(self):
        result = server.verify_quote("not" + ZWSP * 20, paper_id="paper-a")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        self.assertIsNone(result["char_interval"])
        self.assertNotIn("document_text", result)

    def test_soft_hyphen_padding_does_not_either(self):
        result = server.verify_quote("not" + SOFT_HYPHEN * 20, paper_id="paper-a")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])

    def test_whitespace_padding_does_not_either(self):
        result = server.verify_quote("not" + " " * 40 + "x", paper_id="paper-a")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])

    def test_find_passage_applies_the_same_floor(self):
        result = server.find_passage("not" + ZWSP * 20)
        self.assertEqual("NOT_LOCATABLE", result["outcome"])

    def test_the_reason_says_the_floor_is_measured_after_normalisation(self):
        result = server.verify_quote("not" + ZWSP * 20, paper_id="paper-a")
        reasons = " ".join(a["reason"] for a in result["not_available"])
        self.assertIn("invisible characters are removed", reasons)

    def test_a_long_enough_string_is_unaffected(self):
        result = server.verify_quote(
            "Regulatory sandboxes are established" + ZWSP * 5, paper_id="paper-a"
        )
        self.assertEqual(Outcome.VERBATIM_EXACT.value, result["outcome"])


class TestANfkcFoldCannotCreateCharacterIdentity(AdversarialTestCase):
    """Defect: NFKC discards position, and position carries the number.

    The normaliser applies NFKC because that is what makes a formula quotable.
    NFKC folds U+2076 SUPERSCRIPT SIX to "6", so a document reading "10<super
    six> operations per second" and a quote reading "106 operations per second"
    were character-identical after normalisation and the tool said so. They are
    different numbers by six orders of magnitude, and the outcome asserting
    they were the same characters is the exact fabrication this server exists
    to refuse.

    The fix compares the superscript, subscript and fraction characters of the
    quote against those of the located document span before either identity
    tier may be returned. It can only demote.
    """

    def test_a_superscript_folded_away_is_not_character_identity(self):
        result = server.verify_quote(
            "Throughput reached 106 operations per second", paper_id="paper-a"
        )
        self.assertEqual(
            Outcome.PASSAGE_RELOCATED_QUOTE_DIFFERS.value, result["outcome"]
        )
        self.assertIn(SUPER_SIX, result["document_text"])
        self.assertIn(SUPER_SIX, result["char_diff"])

    def test_the_same_attack_from_the_other_side_is_also_refused(self):
        # The document has the plain digit and the quote has the superscript.
        self.use_corpus(FakeCorpus({"paper-p": "Throughput reached 106 ops."}))
        result = server.verify_quote(
            "Throughput reached 10" + SUPER_SIX + " ops.", paper_id="paper-p"
        )
        self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES)

    def test_a_subscript_folded_away_is_not_character_identity(self):
        result = server.verify_quote(
            "The reaction was run in H2O at room temperature", paper_id="paper-a"
        )
        self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES)

    def test_a_vulgar_fraction_folded_away_is_not_character_identity(self):
        result = server.verify_quote(
            "About 1/2 of the sample was discarded before analysis",
            paper_id="paper-a",
        )
        self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES)

    def test_the_guard_demotes_and_never_promotes(self):
        # Quoting the superscript exactly as the document holds it still
        # reaches the identity tier, at every threshold, so the guard has not
        # cost a true match.
        quote = "Throughput reached 10" + SUPER_SIX + " operations per second"
        tiers = self.tiers_over_the_threshold_sweep(PAPER_A, quote)
        self.assertEqual({Tier.T1_EXACT}, tiers)
        result = server.verify_quote(quote, paper_id="paper-a")
        self.assertEqual(Outcome.VERBATIM_EXACT.value, result["outcome"])

    def test_find_passage_does_not_list_a_folded_apart_match_as_a_hit(self):
        result = server.find_passage("Throughput reached 106 operations per second")
        self.assertEqual(0, result["hit_count"])

    def test_no_threshold_restores_the_identity_verdict(self):
        quote = "Throughput reached 106 operations per second"
        tiers = self.tiers_over_the_threshold_sweep(PAPER_A, quote)
        self.assertEqual(set(), tiers & set(IDENTITY_TIERS))


class TestTheCharDiffLegendDescribesTheRealDiff(AdversarialTestCase):
    """Defect: the legend described a notation the producer never emitted.

    The relocated envelope carried a legend saying that "[-...-]" marks
    characters in your_text and "{+...+}" marks characters in the document.
    `_char_diff` has never emitted either marker. A caller who read the legend
    and then searched the diff for those brackets would have found none, and
    the honest reading of "no differences marked" is "the two texts agree",
    which is the blur the relocated outcome exists to prevent.
    """

    def test_the_legend_names_what_the_producer_actually_emits(self):
        diff = _char_diff(
            "the model does improve on set A", "the model did not improve on set B"
        )
        legend = env.CHAR_DIFF_LEGEND
        self.assertIn("minus sign", legend)
        self.assertIn("plus sign", legend)
        # The producer marks each side with a leading sign on its own line.
        signs = {line.strip()[0] for line in diff.splitlines() if line.startswith("    ")}
        self.assertEqual({"-", "+"}, signs)

    def test_the_legend_does_not_describe_a_notation_never_emitted(self):
        diff = _char_diff(
            "the model does improve on set A", "the model did not improve on set B"
        )
        for marker in ("[-", "-]", "{+", "+}"):
            self.assertNotIn(marker, diff)
            self.assertNotIn(marker, env.CHAR_DIFF_LEGEND)

    def test_a_real_relocated_response_carries_that_legend(self):
        result = server.verify_quote(
            NEAR_MISSES_OF_A["negation_dropped"], paper_id="paper-a"
        )
        self.assertEqual(env.CHAR_DIFF_LEGEND, result["char_diff_legend"])

    def test_the_legend_obeys_the_language_rules_it_travels_with(self):
        self.assertTrue(env._verbatim_uses_are_outcome_names(env.CHAR_DIFF_LEGEND))
        for term in env.BANNED_TERMS:
            self.assertNotIn(term, env.CHAR_DIFF_LEGEND.lower())
        # Written as escapes so this file does not itself trip the repo wide
        # check that forbids the two characters.
        self.assertNotIn("\u2014", env.CHAR_DIFF_LEGEND)
        self.assertNotIn("\u2013", env.CHAR_DIFF_LEGEND)


class TestAnIdentifierIsNotAnAssertion(AdversarialTestCase):
    """Defect: a paper identifier was policed as though the server wrote it.

    Rule 4 forbids a system-generated string from containing `demonstrates` and
    six other verbs. A paper identifier comes out of the user's own filename,
    and a corpus root is whatever directory the user pointed at. Both were
    being checked, so a corpus holding `kim-2020-demonstrates-traceability`
    raised EnvelopeViolation on every call that touched it, and any caller
    could raise the same exception by passing that identifier as an argument.
    The promise is a passage or an explicit refusal; a stack trace is neither.
    """

    BANNED_ID = "kim-2020-demonstrates-traceability"
    WORD_ID = "lee-2019-verbatim-anchoring"

    def test_a_held_paper_whose_identifier_holds_a_banned_term_still_answers(self):
        self.use_corpus(FakeCorpus({self.BANNED_ID: PAPER_A}))
        result = server.verify_quote(
            "Regulatory sandboxes are established under Article 57",
            paper_id=self.BANNED_ID,
        )
        self.assertEqual(Outcome.VERBATIM_EXACT.value, result["outcome"])
        self.assertEqual(self.BANNED_ID, result["paper_id"])

    def test_every_banned_term_in_an_identifier_is_survivable(self):
        for term in env.BANNED_TERMS:
            pid = "author-2020-%s-something" % term.replace(" ", "-")
            with self.subTest(term=term):
                self.use_corpus(FakeCorpus({pid: PAPER_A}))
                result = server.verify_quote(
                    "Regulatory sandboxes are established under Article 57",
                    paper_id=pid,
                )
                self.assertEqual(Outcome.VERBATIM_EXACT.value, result["outcome"])

    def test_an_identifier_holding_the_word_is_survivable_too(self):
        self.use_corpus(FakeCorpus({self.WORD_ID: PAPER_A}))
        result = server.verify_quote(
            "Regulatory sandboxes are established under Article 57",
            paper_id=self.WORD_ID,
        )
        self.assertEqual(Outcome.VERBATIM_EXACT.value, result["outcome"])

    def test_a_caller_supplied_identifier_is_refused_and_not_raised(self):
        result = server.verify_quote(
            "Regulatory sandboxes are established", paper_id="a-study-that-demonstrates"
        )
        self.assertEqual("SOURCE_UNKNOWN", result["outcome"])

    def test_find_passage_lists_such_a_paper_as_a_hit(self):
        self.use_corpus(FakeCorpus({self.BANNED_ID: PAPER_A}))
        result = server.find_passage(
            "Regulatory sandboxes are established under Article 57"
        )
        self.assertEqual([self.BANNED_ID], [h["paper_id"] for h in result["hits"]])

    def test_find_passage_names_such_a_paper_in_not_available(self):
        self.use_corpus(FakeCorpus({self.BANNED_ID: PAPER_A}, unheld={self.BANNED_ID}))
        result = server.find_passage(
            "Regulatory sandboxes are established under Article 57"
        )
        self.assertEqual("OK", result["outcome"])
        self.assertIn(
            "document:%s" % self.BANNED_ID, [a["field"] for a in result["not_available"]]
        )

    def test_also_occurs_in_carries_such_an_identifier(self):
        self.use_corpus(FakeCorpus({"paper-a": PAPER_A, self.BANNED_ID: PAPER_A}))
        result = server.verify_quote(
            "Regulatory sandboxes are established under Article 57"
        )
        self.assertIn(self.BANNED_ID, [result["paper_id"], *result.get("also_occurs_in", [])])

    def test_a_corpus_root_holding_a_banned_term_refuses_cleanly(self):
        def explode(root):
            raise OSError("no manifest under %s" % root)

        server.os.environ[server.CORPUS_ROOT_ENV] = "/papers/what-it-demonstrates"
        server._load_corpus = explode
        result = server.verify_quote("Regulatory sandboxes are established")
        self.assertEqual("CORPUS_UNAVAILABLE", result["outcome"])

    def test_the_exemption_did_not_reach_the_server_own_prose(self):
        # The point of the fix is that `reason` stays fully policed. If the
        # exemption had been applied there instead, this would pass vacuously,
        # so it is checked against validate() rather than against a response.
        response = server.verify_quote(
            "a sentence that appears in neither of these two papers anywhere",
            paper_id="paper-a",
        )
        response["not_available"].append(
            {"field": "page", "reason": "the passage demonstrates the claim"}
        )
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_the_exemption_did_not_reach_a_top_level_reason(self):
        response = server.verify_quote(
            "Regulatory sandboxes are established", paper_id="paper-zz"
        )
        response["reason"] = "this document supports the claim"
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_no_reason_string_in_any_scenario_carries_a_banned_term(self):
        self.use_corpus(FakeCorpus({self.BANNED_ID: PAPER_A}, unheld={self.BANNED_ID}))
        responses = [
            server.verify_quote("Regulatory sandboxes are established under Article 57"),
            server.find_passage("Regulatory sandboxes are established under Article 57"),
        ]
        for response in responses:
            for _path, key, value in env._walk(response):
                if key != "reason" or not isinstance(value, str):
                    continue
                for term in env.BANNED_TERMS:
                    self.assertNotIn(term, value.lower())


class TestTheThresholdIsSweptAndSaidOutLoud(AdversarialTestCase):
    """Attack: move t_locate and see what the relocation tier will accept."""

    def test_a_threshold_of_zero_relocates_anything_but_asserts_nothing(self):
        result = server.verify_quote(
            "zzzz qqqq wwww vvvv xxxx yyyy", paper_id="paper-a", t_locate=0.0
        )
        self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES)

    def test_a_threshold_outside_the_measured_band_is_named(self):
        low, high = T_LOCATE_EQUIVALENT_RANGE
        for value in (0.0, round(low - 0.05, 2), round(high + 0.05, 2), 1.0):
            with self.subTest(t_locate=value):
                result = server.verify_quote(
                    "zzzz qqqq wwww vvvv xxxx yyyy", paper_id="paper-a", t_locate=value
                )
                fields = [a["field"] for a in result["not_available"]]
                self.assertIn("t_locate", fields)

    def test_a_threshold_inside_the_measured_band_is_not_flagged(self):
        result = server.verify_quote(
            "Regulatory sandboxes are established under Article 57",
            paper_id="paper-a",
            t_locate=0.65,
        )
        self.assertNotIn("t_locate", [a["field"] for a in result["not_available"]])

    def test_no_threshold_changes_a_character_identity_verdict(self):
        for value in THRESHOLDS:
            with self.subTest(t_locate=value):
                result = server.verify_quote(
                    "Regulatory sandboxes are established under Article 57",
                    paper_id="paper-a",
                    t_locate=value,
                )
                self.assertEqual(Outcome.VERBATIM_EXACT.value, result["outcome"])


class TestTheWholeBatteryObeysTheFourRules(AdversarialTestCase):
    """Every rule, checked against every attack in this file at once.

    The suite's existing rule tests run over benign fixtures. These run the
    same checks over the inputs that were built to break them, including a
    corpus whose identifiers hold every banned term.
    """

    def responses(self):
        self.use_corpus(
            FakeCorpus(
                {
                    "paper-a": PAPER_A,
                    "paper-b": PAPER_B,
                    "author-2020-demonstrates-it": PAPER_A,
                    "author-2021-verbatim-quoting": PAPER_B,
                }
            )
        )
        quotes = [
            TRUE_PASSAGE_OF_A,
            *PARAPHRASES_OF_A.values(),
            *NEAR_MISSES_OF_A.values(),
            "Throughput reached 106 operations per second",
            "not" + ZWSP * 20,
            "",
            "a",
            PAPER_A * 2,
        ]
        out = []
        for quote in quotes:
            out.append(server.verify_quote(quote))
            out.append(server.verify_quote(quote, paper_id="paper-a"))
            out.append(server.find_passage(quote))
        out.append(server.verify_quote(TRUE_PASSAGE_OF_A, paper_id="nothing-held"))
        return out

    def test_rule_one_no_response_carries_a_liftable_field_name(self):
        for response in self.responses():
            for path, key, _value in env._walk(response):
                self.assertNotIn(key, env.QUOTE_FIELD_NAMES, path)

    def test_rule_two_the_word_stays_inside_the_two_outcome_names(self):
        for response in self.responses():
            for path, text in env.iter_strings(response, system_only=True):
                self.assertTrue(env._verbatim_uses_are_outcome_names(text), path)

    def test_rule_three_every_not_locatable_is_explicit(self):
        for response in self.responses():
            if response["outcome"] != "NOT_LOCATABLE":
                continue
            self.assertIn("char_interval", response)
            self.assertIsNone(response["char_interval"])
            self.assertIn("best_score", response)
            self.assertIn("scorer", response)

    def test_rule_four_no_system_string_asserts_what_a_passage_does(self):
        for response in self.responses():
            for path, text in env.iter_strings(response, system_only=True):
                for term in env.BANNED_TERMS:
                    self.assertNotIn(term, text.lower(), path)

    def test_no_response_carries_a_boolean(self):
        for response in self.responses():
            for path, _key, value in env._walk(response):
                self.assertNotIsInstance(value, bool, path)

    def test_every_outcome_stays_inside_the_closed_vocabulary(self):
        for response in self.responses():
            for _path, key, value in env._walk(response):
                if key == "outcome":
                    self.assertIn(value, env.OUTCOME_NAMES)

    def test_no_identity_outcome_is_reached_without_the_document_characters(self):
        for response in self.responses():
            if response["outcome"] not in IDENTITY_OUTCOMES:
                continue
            self.assertIn("document_text", response)
            self.assertTrue(response["document_text"])
            self.assertIsNotNone(response["char_interval"])


class TestRuleTwoAcceptsOnlyWholeOutcomeNames(AdversarialTestCase):
    """Defect: rule 2 accepted any string that STARTED with an outcome name.

    The rule says the word may appear only inside the two outcome names. The
    check asked whether the text at that offset started with one of them, so
    "VERBATIM_EXACTLY what the paper says" passed: it is prose, it begins with
    an outcome name, and prose is exactly what the rule forbids. The check now
    requires the whole name, with something other than a letter, a digit or an
    underscore after it.
    """

    def test_a_prefix_of_an_outcome_name_is_not_a_licence(self):
        for prose in (
            "VERBATIM_EXACTLY what the paper says",
            "VERBATIM_EXACT2",
            "VERBATIM_RELAXED_EXTRACTOR_DAMAGEs in every paper",
            "VERBATIM_EXACT_MATCH",
        ):
            with self.subTest(prose=prose):
                self.assertFalse(env._verbatim_uses_are_outcome_names(prose))

    def test_the_two_outcome_names_themselves_still_pass(self):
        for text in (
            "VERBATIM_EXACT",
            "VERBATIM_RELAXED_EXTRACTOR_DAMAGE",
            "VERBATIM_EXACT: the string occurs literally.",
            "either VERBATIM_EXACT or VERBATIM_RELAXED_EXTRACTOR_DAMAGE.",
        ):
            with self.subTest(text=text):
                self.assertTrue(env._verbatim_uses_are_outcome_names(text))

    def test_the_notice_and_both_descriptions_still_pass(self):
        for text in (
            env.INTERPRETATION_NOTICE,
            env.CHAR_DIFF_LEGEND,
            server.VERIFY_QUOTE_DESCRIPTION,
            server.FIND_PASSAGE_DESCRIPTION,
        ):
            self.assertTrue(env._verbatim_uses_are_outcome_names(text))

    def test_validate_rejects_a_prefix_dressed_as_prose(self):
        response = server.verify_quote(TRUE_PASSAGE_OF_A, paper_id="paper-a")
        response["note"] = "VERBATIM_EXACTLY as printed in the paper"
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)


class TestTheForeignTextExemptionIsNotABackDoor(AdversarialTestCase):
    """The exemption that fixes the identifier defect must not widen further.

    `field`, `detail`, `paper_id` and `also_occurs_in` stop being policed by
    rules 2 and 4, because the strings under them come from a filename, a path
    or the operating system. That is only safe while the server's own prose
    stays somewhere else, so these tests walk every absence the server can
    emit and check that the slug is one of a known, enumerated set rather than
    free text that the exemption would now wave through.
    """

    KNOWN_SLUGS = frozenset(
        {
            "char_diff",
            "char_interval",
            "corpus",
            "corpus_freshness",
            "document",
            "hits",
            "max_hits",
            "page",
            "t_locate",
        }
    )

    def battery(self):
        responses = []
        self.use_corpus(
            FakeCorpus(
                {"paper-a": PAPER_A, "scan-1": "", "bad": PAPER_B, "no-pages": PAPER_B},
                unheld={"scan-1"},
                unreadable={"bad"},
                no_pages={"no-pages"},
                stale=["one file changed"],
            )
        )
        responses.append(server.verify_quote(TRUE_PASSAGE_OF_A))
        responses.append(server.verify_quote("short"))
        responses.append(
            server.verify_quote(TRUE_PASSAGE_OF_A, paper_id="paper-a", t_locate=0.0)
        )
        responses.append(server.verify_quote(TRUE_PASSAGE_OF_A, t_locate="not a number"))
        responses.append(server.find_passage(TRUE_PASSAGE_OF_A, max_hits=999))
        responses.append(server.find_passage(TRUE_PASSAGE_OF_A, paper_ids=["absent"]))
        responses.append(server.find_passage("a phrase held by nobody here at all"))
        return responses

    def test_every_absence_slug_is_a_known_name(self):
        seen = set()
        for response in self.battery():
            for entry in response["not_available"]:
                slug = entry["field"].split(":", 1)[0]
                seen.add(slug)
                self.assertIn(slug, self.KNOWN_SLUGS, entry["field"])
        self.assertTrue(seen)

    def test_every_absence_reason_is_the_server_own_policed_prose(self):
        for response in self.battery():
            for entry in response["not_available"]:
                reason = entry["reason"]
                self.assertTrue(env._verbatim_uses_are_outcome_names(reason), reason)
                for term in env.BANNED_TERMS:
                    self.assertNotIn(term, reason.lower(), reason)

    def test_the_exempt_key_list_has_not_grown(self):
        # A guard on the exemption itself. Widening it is a decision, not a
        # detail, and it should fail a test rather than pass silently.
        self.assertEqual(
            {"also_occurs_in", "detail", "field", "paper_id"},
            set(env.FOREIGN_TEXT_FIELDS),
        )
        self.assertEqual(
            {"char_diff", "document_text", "query", "your_text"},
            set(env.CALLER_OR_DOCUMENT_FIELDS),
        )


class TestAFractionBesideADigitIsADifferentNumber(AdversarialTestCase):
    """The fraction half of the fold guard, which is the least obvious one.

    A vulgar fraction folds to its own value, so on its own the fold keeps the
    meaning. Beside a digit it does not: "2" then U+00BD folds to "21/2", so a
    document saying two and a half and a quote saying twenty-one halves become
    the same characters.
    """

    def test_a_fraction_beside_a_digit_is_not_character_identity(self):
        self.use_corpus(
            FakeCorpus(
                {"paper-f": "The mean was 2" + HALF + " times higher than control."}
            )
        )
        result = server.verify_quote(
            "The mean was 21/2 times higher than control.", paper_id="paper-f"
        )
        self.assertNotIn(result["outcome"], IDENTITY_OUTCOMES)
        self.assertIn(HALF, result["document_text"])

    def test_quoting_the_fraction_as_printed_still_reaches_identity(self):
        self.use_corpus(
            FakeCorpus(
                {"paper-f": "The mean was 2" + HALF + " times higher than control."}
            )
        )
        result = server.verify_quote(
            "The mean was 2" + HALF + " times higher than control.",
            paper_id="paper-f",
        )
        self.assertEqual(Outcome.VERBATIM_EXACT.value, result["outcome"])


class TestASeededFuzzFindsNoIllegitimateIdentityVerdict(AdversarialTestCase):
    """The same attack, at volume, so a hand picked example cannot be lucky.

    Spans of the paper are drawn at random, mutated one to three times, and put
    through the ladder at seven thresholds. Every mutation family that broke
    something by hand is in the generator: a character substituted, deleted or
    inserted, an invisible or a homoglyph or a superscript inserted, a word
    spliced in, the clauses shuffled, the case flipped, the hyphens deleted.

    The invariant is the property itself, checked independently of the ladder
    rather than by asking the ladder what it decided: an identity tier may be
    returned only when the normalised quote really is a substring of the
    normalised document, or its match form really is a substring of the
    document's match form. The seed is fixed so a failure is reproducible.
    """

    SEED = 20260829
    SPANS = 120
    SWEEP = (0.0, 0.3, 0.55, 0.65, 0.72, 0.9, 1.0)
    WORDS = (
        "method", "system", "result", "approach", "provide", "offer", "supply",
        "controlled", "managed", "testing", "evaluation", "not", "does", "did",
        "0.412", "0.512", "1024", "2016", "2019", "Hasan", "Akhigbe",
    )
    CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,-"

    def weird(self):
        return (
            ZWSP, SOFT_HYPHEN, CYRILLIC_A, RLO, SUPER_SIX, SUB_TWO, HALF,
            LIGATURE_FI, "\u2013", "\u2212", "\u00a0", "\u2019", "\u00df",
        )

    def mutate(self, rng, text):
        if not text:
            return text
        kind = rng.randrange(8)
        i = rng.randrange(len(text))
        if kind == 0:
            return text[:i] + rng.choice(self.CHARS) + text[i + 1:]
        if kind == 1:
            return text[:i] + text[i + 1:]
        if kind == 2:
            return text[:i] + rng.choice(self.CHARS) + text[i:]
        if kind == 3:
            return text[:i] + rng.choice(self.weird()) + text[i:]
        if kind == 4:
            return text[:i] + " " + rng.choice(self.WORDS) + " " + text[i:]
        if kind == 5:
            parts = text.split()
            rng.shuffle(parts)
            return " ".join(parts)
        if kind == 6:
            return text.upper() if rng.random() < 0.5 else text.lower()
        return text.replace("-", "") if "-" in text else text + rng.choice(self.WORDS)

    def test_no_mutation_reaches_an_identity_tier_without_being_identical(self):
        import random

        rng = random.Random(self.SEED)
        prepared = prepare(PAPER_A)
        checked = 0
        for _ in range(self.SPANS):
            start = rng.randrange(0, len(PAPER_A) - 40)
            quote = PAPER_A[start:start + rng.randrange(20, 80)]
            for _ in range(rng.randrange(1, 4)):
                quote = self.mutate(rng, quote)
            for threshold in self.SWEEP:
                anchor = locate(prepared, quote, t_locate=threshold)
                checked += 1
                if anchor.tier not in IDENTITY_TIERS:
                    continue
                quote_norm = normalise(quote)[0]
                identical = (
                    quote_norm in prepared.norm
                    or match_form(quote_norm) in prepared.match
                )
                self.assertTrue(
                    identical,
                    "tier %s at t_locate %.2f for a string that is not "
                    "character identical: %r" % (anchor.tier.name, threshold, quote),
                )
        self.assertGreater(checked, 500)

    def test_every_identity_verdict_agrees_with_the_document_own_characters(self):
        import random
        import unicodedata

        rng = random.Random(self.SEED + 1)
        prepared = prepare(PAPER_A)
        verdicts = 0
        for _ in range(self.SPANS):
            start = rng.randrange(0, len(PAPER_A) - 40)
            base = PAPER_A[start:start + rng.randrange(20, 80)]
            for quote in (base, self.mutate(rng, base)):
                anchor = locate(prepared, quote)
                if anchor.tier not in IDENTITY_TIERS:
                    continue
                verdicts += 1
                index_map = prepared.norm_index_map
                end = anchor.offset_norm + anchor.length_norm
                raw_start = index_map[anchor.offset_norm]
                raw_end = (
                    index_map[end] if end < len(index_map) else len(prepared.raw)
                )
                span = prepared.raw[raw_start:raw_end]

                def positional(text):
                    return tuple(
                        ch for ch in text
                        if unicodedata.decomposition(ch).startswith(
                            ("<super>", "<sub>", "<fraction>")
                        )
                    )

                self.assertEqual(
                    positional(quote),
                    positional(span),
                    "a lossy fold reached an identity tier for %r" % (quote,),
                )
        self.assertGreater(verdicts, 20)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
