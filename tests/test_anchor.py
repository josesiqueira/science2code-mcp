"""Tests for the anchoring ladder in science2code/anchor.py.

Stdlib unittest only, no third-party imports:

    python3 -m unittest discover -s tests -p "test_*.py"
    python3 tests/test_anchor.py

Every fixture in this file is a short inline string, and every one of them is
INVENTED. No passage here is quoted from any real publication, and nothing here
opens a corpus file. A test that read a corpus would fail for every reader who
does not hold that corpus, and a fixture copied out of a paper would put
someone else's text into a public repository for no benefit at all.
"""

import dataclasses
import os
import sys
import unittest

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from science2code import anchor as s2c_anchor  # noqa: E402
from science2code.anchor import Anchor, Tier, locate  # noqa: E402
from science2code.normalise import (  # noqa: E402
    FINGERPRINT_UNAVAILABLE,
    fingerprint_file,
    fingerprint_source,
)

# ---------------------------------------------------------------------------
# Fixtures: short inline strings only, all of them invented.
#
# PAPER_DOC is shaped like a passage of extracted body text rather than being
# one: several sentences, a hyphenated compound, a curly apostrophe the agent
# will type as an ASCII one, and an inline citation marker.
# ---------------------------------------------------------------------------

PAPER_DOC = (
    "Sensor drift has been reported in field deployments for three decades.\n"
    "The present authors define drift as the slow divergence between a "
    "probe’s reported value and the quantity it is meant to track, a "
    "divergence that widens whenever recalibration is deferred [14]. They "
    "argue that the cadence of recalibration matters more than the nominal "
    "accuracy of the probe, because an uncorrected low-drift instrument "
    "eventually reports worse figures than a well-tended noisy one.\n"
)

# The compound hyphen of "long-term" has been eaten by the extractor, which is
# what pdftotext does when the compound falls on a line break.
EATEN_HYPHEN_DOC = (
    "The protocol applies to laboratories that report longterm calibration "
    "records, and to the field stations that submit those records for review "
    "before the close of the sampling season.\n"
)

# A character whose casefold EXPANDS (U+00DF folds to "ss") sitting before the
# span to be matched. This is a regression fixture: an index map walked one
# folded character at a time desynchronises here and reports the wrong offset.
EXPANDING_FOLD_DOC = (
    "Die Straße war lang. The committee then discussed longterm "
    "reporting duties for the field stations that joined the review cycle.\n"
)

UNRELATED_QUOTE = (
    "The harbour ferry departs on the hour and carries bicycles at no "
    "additional charge throughout the winter timetable."
)

# A document identifier, not a path and not a real filename. The record format
# treats `source` as an opaque string, so a caller may put a DOI, a hash or a
# local name here.
SOURCE_ID = "example-document-0001.txt"


def norm(text):
    return s2c_anchor.normalise(text)[0]


class LadderTierOneTests(unittest.TestCase):
    def test_exact_hit_returns_t1_at_the_right_offset(self):
        quote = "the cadence of recalibration matters more than the nominal"
        anchor = locate(PAPER_DOC, quote)
        self.assertIs(anchor.tier, Tier.T1_EXACT)
        self.assertEqual(anchor.score, 1.0)
        self.assertIsNone(anchor.diff)
        haystack = norm(PAPER_DOC)
        self.assertEqual(anchor.offset_norm, haystack.find(norm(quote)))
        self.assertEqual(
            haystack[anchor.offset_norm : anchor.offset_norm + anchor.length_norm],
            norm(quote),
        )

    def test_curly_apostrophe_quote_reaches_t1_after_normalisation(self):
        # The document has U+2019, the agent typed U+0027.
        self.assertIn("’", PAPER_DOC)
        quote = "a probe's reported value and the quantity it is meant to track"
        self.assertNotIn(quote, PAPER_DOC)
        anchor = locate(PAPER_DOC, quote)
        self.assertIs(anchor.tier, Tier.T1_EXACT)
        self.assertTrue(anchor.is_verbatim)

    def test_curly_double_quotes_reach_t1_after_normalisation(self):
        doc = "She called it “deferred recalibration” in the report."
        anchor = locate(doc, 'called it "deferred recalibration" in the report')
        self.assertIs(anchor.tier, Tier.T1_EXACT)

    def test_a_quote_carrying_a_citation_marker_still_reaches_t1(self):
        # An inline marker means the passage is an ATTRIBUTED claim rather than
        # an original one. That distinction is the caller's to draw; the ladder
        # only has to locate the characters, marker included.
        quote = "whenever recalibration is deferred [14]"
        self.assertIs(locate(PAPER_DOC, quote).tier, Tier.T1_EXACT)


class LadderTierTwoTests(unittest.TestCase):
    def test_extractor_destroyed_compound_hyphen_reaches_t2(self):
        quote = "laboratories that report long-term calibration records"
        self.assertIn("longterm", EATEN_HYPHEN_DOC)
        anchor = locate(EATEN_HYPHEN_DOC, quote)
        self.assertIs(anchor.tier, Tier.T2_RELAXED)
        self.assertTrue(anchor.is_verbatim)
        self.assertEqual(anchor.score, 1.0)
        haystack = norm(EATEN_HYPHEN_DOC)
        span = haystack[anchor.offset_norm : anchor.offset_norm + anchor.length_norm]
        self.assertEqual(span, "laboratories that report longterm calibration records")
        # The document span is one character shorter than the quote, because
        # the fold deletes the hyphen on the quote side only.
        self.assertEqual(anchor.length_norm, len(norm(quote)) - 1)

    def test_case_only_difference_reaches_t2_not_t1(self):
        quote = "THE PROTOCOL APPLIES TO LABORATORIES"
        anchor = locate(EATEN_HYPHEN_DOC, quote)
        self.assertIs(anchor.tier, Tier.T2_RELAXED)
        haystack = norm(EATEN_HYPHEN_DOC)
        span = haystack[anchor.offset_norm : anchor.offset_norm + anchor.length_norm]
        self.assertEqual(span, "The protocol applies to laboratories")

    def test_t2_span_is_correct_after_an_expanding_casefold(self):
        anchor = locate(EXPANDING_FOLD_DOC, "long-term reporting duties")
        self.assertIs(anchor.tier, Tier.T2_RELAXED)
        haystack = norm(EXPANDING_FOLD_DOC)
        span = haystack[anchor.offset_norm : anchor.offset_norm + anchor.length_norm]
        self.assertEqual(span, "longterm reporting duties")

    def test_a_verbatim_tier_always_returns_a_verified_span(self):
        """The invariant: T1 and T2 never hand back an unverified offset."""
        cases = [
            (PAPER_DOC, "the cadence of recalibration matters more than the nominal"),
            (PAPER_DOC, "a probe's reported value and the quantity"),
            (EATEN_HYPHEN_DOC,
             "laboratories that report long-term calibration records"),
            (EATEN_HYPHEN_DOC, "THE PROTOCOL APPLIES TO LABORATORIES"),
            (EXPANDING_FOLD_DOC, "long-term reporting duties"),
            (EXPANDING_FOLD_DOC, "Die Straße war lang"),
        ]
        for doc, quote in cases:
            with self.subTest(quote=quote[:36]):
                anchor = locate(doc, quote)
                self.assertTrue(anchor.is_verbatim)
                haystack = norm(doc)
                span = haystack[
                    anchor.offset_norm : anchor.offset_norm + anchor.length_norm
                ]
                self.assertEqual(
                    s2c_anchor.match_form(span),
                    s2c_anchor.match_form(norm(quote)),
                )


class LadderTierThreeTests(unittest.TestCase):
    # The document says "and the quantity it is meant to track". The quote has
    # lost one word, which is damage to the QUOTE, not to the extraction.
    QUOTE_ONE_WORD_DROPPED = (
        "the slow divergence between a probe's reported value and the it is "
        "meant to track, a divergence that widens whenever recalibration is "
        "deferred"
    )

    def test_dropped_word_reaches_t3_with_a_diff(self):
        anchor = locate(PAPER_DOC, self.QUOTE_ONE_WORD_DROPPED)
        self.assertIs(anchor.tier, Tier.T3_LOCATED)
        self.assertIsNotNone(anchor.offset_norm)
        self.assertIsNotNone(anchor.diff)
        self.assertIn("char diff", anchor.diff)
        self.assertIn("quantity", anchor.diff)

    def test_t3_is_not_verbatim(self):
        anchor = locate(PAPER_DOC, self.QUOTE_ONE_WORD_DROPPED)
        self.assertFalse(anchor.is_verbatim)
        self.assertFalse(anchor.tier.is_verbatim)

    def test_t3_score_is_between_the_threshold_and_one(self):
        anchor = locate(PAPER_DOC, self.QUOTE_ONE_WORD_DROPPED)
        self.assertGreaterEqual(anchor.score, s2c_anchor.T_LOCATE_DEFAULT)
        self.assertLess(anchor.score, 1.0)

    def test_t3_locates_the_right_passage(self):
        anchor = locate(PAPER_DOC, self.QUOTE_ONE_WORD_DROPPED)
        haystack = norm(PAPER_DOC)
        truth = haystack.find("the slow divergence between")
        self.assertLess(abs(anchor.offset_norm - truth), 40)


class LadderTierFourTests(unittest.TestCase):
    def test_fabricated_passage_from_unrelated_text_returns_t4(self):
        anchor = locate(PAPER_DOC, UNRELATED_QUOTE)
        self.assertIs(anchor.tier, Tier.T4_NOT_LOCATABLE)
        self.assertFalse(anchor.is_verbatim)

    def test_t4_never_carries_an_offset(self):
        anchor = locate(PAPER_DOC, UNRELATED_QUOTE)
        self.assertIsNone(anchor.offset_norm)
        self.assertIsNone(anchor.length_norm)
        self.assertIsNone(anchor.diff)

    def test_t4_still_reports_how_close_it_got(self):
        anchor = locate(PAPER_DOC, UNRELATED_QUOTE)
        self.assertIsNotNone(anchor.score)
        self.assertLess(anchor.score, s2c_anchor.T_LOCATE_DEFAULT)

    def test_empty_quote_is_t4_and_not_an_exact_hit_at_zero(self):
        anchor = locate(PAPER_DOC, "   ")
        self.assertIs(anchor.tier, Tier.T4_NOT_LOCATABLE)
        self.assertIsNone(anchor.offset_norm)

    def test_empty_haystack_is_t4_not_an_exception(self):
        # A source document with no text layer at all lands here for every
        # quote, and that is a boundary, not a failure.
        anchor = locate("", "the cadence of recalibration")
        self.assertIs(anchor.tier, Tier.T4_NOT_LOCATABLE)


class ParaphraseBoundaryTests(unittest.TestCase):
    """The property the whole ladder exists to guarantee."""

    PARAPHRASES = [
        # synonym substitution, same clause order
        "The present authors describe drift as the gradual separation between "
        "a sensor's recorded figure and the property it is supposed to follow, "
        "a separation that grows whenever adjustment is postponed",
        # heavier: reordered and trimmed
        "a neglected instrument with little inherent drift eventually gives "
        "poorer numbers than a carefully maintained noisy one, so how often "
        "recalibration happens counts for more than the rated accuracy",
        # gist only
        "recalibrate often rather than buying a more accurate probe",
    ]

    def test_synonym_paraphrase_never_returns_t1_or_t2(self):
        for text in self.PARAPHRASES:
            with self.subTest(paraphrase=text[:40]):
                anchor = locate(PAPER_DOC, text)
                self.assertNotIn(anchor.tier, (Tier.T1_EXACT, Tier.T2_RELAXED))
                self.assertFalse(anchor.is_verbatim)

    def test_paraphrase_stays_non_verbatim_at_every_threshold(self):
        # T1 and T2 carry no threshold, so moving t_locate cannot promote a
        # paraphrase to "verbatim". It can only move it between T3 and T4.
        for t_locate in (0.0, 0.5, 0.65, 0.9, 1.0):
            for text in self.PARAPHRASES:
                with self.subTest(t=t_locate, paraphrase=text[:30]):
                    anchor = locate(PAPER_DOC, text, t_locate=t_locate)
                    self.assertFalse(anchor.is_verbatim)


class ReturnTypeTests(unittest.TestCase):
    def test_locate_returns_a_tier_not_a_bool(self):
        anchor = locate(PAPER_DOC, "cadence of recalibration matters")
        self.assertIsInstance(anchor, Anchor)
        self.assertIsInstance(anchor.tier, Tier)
        self.assertNotIsInstance(anchor.tier, bool)
        self.assertNotIsInstance(anchor.tier, int)
        self.assertNotIsInstance(anchor.tier, str)

    def test_every_tier_is_reachable_and_distinct(self):
        self.assertEqual(len(set(Tier)), 4)
        self.assertEqual(
            [t.value for t in Tier],
            ["T1_EXACT", "T2_RELAXED", "T3_LOCATED", "T4_NOT_LOCATABLE"],
        )

    def test_locate_never_returns_none_on_a_miss(self):
        self.assertIsNotNone(locate(PAPER_DOC, UNRELATED_QUOTE))

    def test_anchor_is_frozen(self):
        # dataclasses.FrozenInstanceError, named exactly, because a blind
        # assertRaises(Exception) here would also pass if the attribute did
        # not exist, which is the opposite of what this asserts.
        anchor = locate(PAPER_DOC, "cadence of recalibration matters")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            anchor.tier = Tier.T4_NOT_LOCATABLE


class ScorerTests(unittest.TestCase):
    def test_scorer_name_is_recorded_on_the_result(self):
        for quote in (
            "cadence of recalibration matters",  # T1
            "a probe's reported low-drift value",  # hyphen fold path
            UNRELATED_QUOTE,  # T4
        ):
            with self.subTest(quote=quote[:30]):
                anchor = locate(PAPER_DOC, quote)
                self.assertEqual(anchor.scorer, "difflib")

    def test_default_scorer_is_difflib(self):
        self.assertEqual(locate(PAPER_DOC, UNRELATED_QUOTE).scorer, "difflib")

    def test_unknown_scorer_raises(self):
        with self.assertRaises(ValueError):
            locate(PAPER_DOC, "cadence of recalibration", scorer="levenshtein")

    def test_rapidfuzz_is_optional_and_honest_about_it(self):
        if s2c_anchor._rapidfuzz_fuzz is None:
            with self.assertRaises(ValueError):
                locate(PAPER_DOC, "cadence of recalibration", scorer="rapidfuzz")
        else:
            anchor = locate(PAPER_DOC, "cadence of recalibration",
                            scorer="rapidfuzz")
            self.assertEqual(anchor.scorer, "rapidfuzz")

    def test_threshold_is_documented_as_a_range(self):
        low, high = s2c_anchor.T_LOCATE_EQUIVALENT_RANGE
        self.assertLessEqual(low, s2c_anchor.T_LOCATE_DEFAULT)
        self.assertLessEqual(s2c_anchor.T_LOCATE_DEFAULT, high)


class PreparedTextTests(unittest.TestCase):
    def test_prepared_haystack_gives_the_same_answer_as_a_raw_string(self):
        prepared = s2c_anchor.prepare(PAPER_DOC)
        quote = "the cadence of recalibration matters more than the nominal"
        self.assertEqual(locate(prepared, quote), locate(PAPER_DOC, quote))


class AnchorRecordTests(unittest.TestCase):
    QUOTE = "the cadence of recalibration matters more than the nominal"

    def record(self):
        return s2c_anchor.anchor_record(self.QUOTE, PAPER_DOC, source=SOURCE_ID)

    def test_record_carries_w3c_selectors(self):
        record = self.record()
        self.assertEqual(record["@context"], s2c_anchor.W3C_ANNOTATION_CONTEXT)
        self.assertEqual(record["type"], "Annotation")
        selectors = record["target"]["selector"]
        kinds = [s["type"] for s in selectors]
        self.assertIn("TextQuoteSelector", kinds)
        self.assertIn("TextPositionSelector", kinds)

    def test_quote_selector_is_authoritative_and_position_is_a_hint(self):
        record = self.record()
        self.assertEqual(record["refs"]["authoritative_selector"],
                         "TextQuoteSelector")
        self.assertTrue(record["refs"]["position_selector_is_cache"])

    def test_quote_selector_has_exact_prefix_and_suffix(self):
        quote_selector = self.record()["target"]["selector"][0]
        self.assertEqual(quote_selector["exact"], norm(self.QUOTE))
        self.assertTrue(quote_selector["prefix"])
        self.assertTrue(quote_selector["suffix"])
        haystack = norm(PAPER_DOC)
        window = (
            quote_selector["prefix"]
            + quote_selector["exact"]
            + quote_selector["suffix"]
        )
        self.assertIn(window, haystack)

    def test_position_selector_indexes_the_normalised_text(self):
        record = self.record()
        position = record["target"]["selector"][1]
        haystack = norm(PAPER_DOC)
        self.assertEqual(
            haystack[position["start"] : position["end"]], norm(self.QUOTE)
        )

    def test_record_carries_every_version_string_and_the_tier(self):
        refs = self.record()["refs"]
        self.assertEqual(refs["normaliser"], s2c_anchor.NORMALISER_VERSION)
        self.assertEqual(refs["matchform"], s2c_anchor.MATCHFORM_VERSION)
        self.assertEqual(refs["verifier"], s2c_anchor.VERIFIER_VERSION)
        self.assertEqual(refs["scorer"], "difflib")
        self.assertEqual(refs["tier"], "T1_EXACT")
        self.assertEqual(refs["quote_raw"], self.QUOTE)

    def test_not_locatable_record_has_no_position_selector(self):
        record = s2c_anchor.anchor_record(UNRELATED_QUOTE, PAPER_DOC)
        kinds = [s["type"] for s in record["target"]["selector"]]
        self.assertEqual(kinds, ["TextQuoteSelector"])
        self.assertEqual(record["refs"]["tier"], "T4_NOT_LOCATABLE")
        self.assertFalse(record["refs"]["is_verbatim"])

    def test_fresh_record_is_fresh(self):
        self.assertEqual(s2c_anchor.record_status(self.record()),
                         s2c_anchor.STATUS_FRESH)
        self.assertFalse(s2c_anchor.record_is_stale(self.record()))

    def test_normaliser_bump_degrades_to_stale_not_invalid(self):
        record = self.record()
        record["refs"]["normaliser"] = "norm/0.9.0"
        self.assertEqual(s2c_anchor.record_status(record),
                         s2c_anchor.STATUS_STALE)
        self.assertTrue(s2c_anchor.record_is_stale(record))
        # Stale, not invalid: the raw quote is still the primary key.
        self.assertEqual(record["refs"]["quote_raw"], self.QUOTE)

    def test_stale_record_reanchors_from_quote_raw(self):
        record = self.record()
        good_position = dict(record["target"]["selector"][1])
        record["refs"]["normaliser"] = "norm/0.9.0"
        record["target"]["selector"][1]["start"] = 999999
        record["target"]["selector"][1]["end"] = 999999
        self.assertTrue(s2c_anchor.record_is_stale(record))

        rebuilt = s2c_anchor.reanchor_record(record, PAPER_DOC)
        self.assertFalse(s2c_anchor.record_is_stale(rebuilt))
        self.assertEqual(rebuilt["refs"]["tier"], "T1_EXACT")
        self.assertEqual(rebuilt["target"]["selector"][1], good_position)
        self.assertEqual(rebuilt["target"]["source"], SOURCE_ID)

    def test_reanchor_preserves_caller_owned_keys(self):
        record = self.record()
        record["id"] = "urn:example:decision:0042"
        rebuilt = s2c_anchor.reanchor_record(record, PAPER_DOC)
        self.assertEqual(rebuilt["id"], "urn:example:decision:0042")

    def test_reanchor_without_a_quote_raw_is_an_error(self):
        with self.assertRaises(ValueError):
            s2c_anchor.reanchor_record({"refs": {}}, PAPER_DOC)


class RecordFingerprintTests(unittest.TestCase):
    """The backstop against a forgotten version bump, at the record level.

    NORMALISER_VERSION is hand-maintained. If the normaliser's behaviour
    changes and nobody edits that literal, every record written afterwards
    reads FRESH forever while its cached offsets point at the wrong place.
    These tests pin the second, machine-maintained signal that makes such a
    record read STALE anyway.
    """

    QUOTE = "the cadence of recalibration matters more than the nominal"

    def record(self):
        return s2c_anchor.anchor_record(self.QUOTE, PAPER_DOC, source=SOURCE_ID)

    def test_record_carries_the_normaliser_fingerprint(self):
        refs = self.record()["refs"]
        self.assertIn("normaliser_fingerprint", refs)
        self.assertEqual(refs["normaliser_fingerprint"],
                         s2c_anchor.NORMALISER_FINGERPRINT)

    def test_the_record_carries_the_version_and_the_fingerprint_both(self):
        # Either one alone is insufficient. The version is human-readable and
        # says WHY it changed; the fingerprint cannot be forgotten.
        refs = self.record()["refs"]
        self.assertNotEqual(refs["normaliser"], refs["normaliser_fingerprint"])
        self.assertTrue(refs["normaliser"])
        self.assertTrue(refs["normaliser_fingerprint"])

    def test_a_fingerprint_mismatch_alone_degrades_to_stale(self):
        # This is the exact failure the fingerprint exists to catch: every
        # version string is still correct, because nobody bumped one, and the
        # behaviour changed underneath the record anyway.
        record = self.record()
        record["refs"]["normaliser_fingerprint"] = "0123456789ab"
        self.assertEqual(record["refs"]["normaliser"],
                         s2c_anchor.NORMALISER_VERSION)
        self.assertEqual(record["refs"]["matchform"],
                         s2c_anchor.MATCHFORM_VERSION)
        self.assertEqual(record["refs"]["verifier"],
                         s2c_anchor.VERIFIER_VERSION)
        self.assertEqual(s2c_anchor.record_status(record),
                         s2c_anchor.STATUS_STALE)
        self.assertTrue(s2c_anchor.record_is_stale(record))

    def test_a_record_written_before_the_fingerprint_existed_reads_stale(self):
        # It cannot be shown to be fresh, so it is not called fresh. STALE is
        # cheap: it means re-anchor, and re-anchoring always succeeds when the
        # quote is genuine.
        record = self.record()
        del record["refs"]["normaliser_fingerprint"]
        self.assertTrue(s2c_anchor.record_is_stale(record))

    def test_a_fingerprint_stale_record_reanchors_to_fresh(self):
        record = self.record()
        record["refs"]["normaliser_fingerprint"] = "0123456789ab"
        record["target"]["selector"][1]["start"] = 999999
        rebuilt = s2c_anchor.reanchor_record(record, PAPER_DOC)
        self.assertFalse(s2c_anchor.record_is_stale(rebuilt))
        self.assertEqual(rebuilt["refs"]["normaliser_fingerprint"],
                         s2c_anchor.NORMALISER_FINGERPRINT)
        self.assertEqual(rebuilt["refs"]["tier"], "T1_EXACT")

    def test_a_fingerprint_mismatch_never_invalidates_the_quote(self):
        # STALE is a third state, never a verdict on the citation.
        record = self.record()
        record["refs"]["normaliser_fingerprint"] = "0123456789ab"
        self.assertEqual(record["refs"]["quote_raw"], self.QUOTE)
        self.assertNotEqual(s2c_anchor.record_status(record), "INVALID")
        self.assertIn(s2c_anchor.record_status(record),
                      (s2c_anchor.STATUS_FRESH, s2c_anchor.STATUS_STALE))


class VerifierFingerprintTests(unittest.TestCase):
    """The same backstop, for the verifier. VERIFIER_VERSION is also a hand
    maintained literal, and it decides which tier a quote reached, so a silent
    change to it is a citation that stopped being checked the way its record
    says it was.
    """

    QUOTE = "the cadence of recalibration matters more than the nominal"

    def record(self):
        return s2c_anchor.anchor_record(self.QUOTE, PAPER_DOC, source=SOURCE_ID)

    def test_the_verifier_has_a_fingerprint_at_all(self):
        self.assertIsInstance(s2c_anchor.VERIFIER_FINGERPRINT, str)
        self.assertNotEqual(s2c_anchor.VERIFIER_FINGERPRINT,
                            FINGERPRINT_UNAVAILABLE)
        self.assertEqual(12, len(s2c_anchor.VERIFIER_FINGERPRINT))

    def test_it_is_not_the_normalisers_fingerprint(self):
        # Two different modules, two different digests. One standing in for
        # the other would leave whichever it replaced unguarded.
        self.assertNotEqual(s2c_anchor.VERIFIER_FINGERPRINT,
                            s2c_anchor.NORMALISER_FINGERPRINT)

    def test_it_is_the_digest_of_this_modules_own_source(self):
        with open(s2c_anchor.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertEqual(fingerprint_source(source),
                         s2c_anchor.VERIFIER_FINGERPRINT)

    def test_record_carries_the_verifier_fingerprint(self):
        refs = self.record()["refs"]
        self.assertIn("verifier_fingerprint", refs)
        self.assertEqual(refs["verifier_fingerprint"],
                         s2c_anchor.VERIFIER_FINGERPRINT)
        self.assertNotEqual(refs["verifier"], refs["verifier_fingerprint"])

    def test_a_verifier_fingerprint_mismatch_alone_degrades_to_stale(self):
        # Every version literal is still correct, because nobody bumped one,
        # and the ladder changed underneath the record anyway.
        record = self.record()
        record["refs"]["verifier_fingerprint"] = "0123456789ab"
        self.assertEqual(record["refs"]["verifier"], s2c_anchor.VERIFIER_VERSION)
        self.assertEqual(record["refs"]["normaliser_fingerprint"],
                         s2c_anchor.NORMALISER_FINGERPRINT)
        self.assertEqual(s2c_anchor.record_status(record),
                         s2c_anchor.STATUS_STALE)

    def test_a_record_written_before_it_existed_reads_stale(self):
        record = self.record()
        del record["refs"]["verifier_fingerprint"]
        self.assertTrue(s2c_anchor.record_is_stale(record))

    def test_a_verifier_stale_record_reanchors_to_fresh(self):
        record = self.record()
        record["refs"]["verifier_fingerprint"] = "0123456789ab"
        rebuilt = s2c_anchor.reanchor_record(record, PAPER_DOC)
        self.assertFalse(s2c_anchor.record_is_stale(rebuilt))
        self.assertEqual(rebuilt["refs"]["verifier_fingerprint"],
                         s2c_anchor.VERIFIER_FINGERPRINT)

    def test_a_verifier_fingerprint_mismatch_never_invalidates_the_quote(self):
        record = self.record()
        record["refs"]["verifier_fingerprint"] = "0123456789ab"
        self.assertEqual(record["refs"]["quote_raw"], self.QUOTE)
        self.assertIn(s2c_anchor.record_status(record),
                      (s2c_anchor.STATUS_FRESH, s2c_anchor.STATUS_STALE))

    def test_both_modules_use_one_fingerprint_implementation(self):
        # Two implementations would be two chances to disagree about what a
        # fingerprint is. There is one, and it is public.
        self.assertIs(s2c_anchor.fingerprint_file, fingerprint_file)


class PackageSurfaceTests(unittest.TestCase):
    """The import line every other module in this project codes against."""

    def test_the_documented_import_surface_exists(self):
        from science2code import (  # noqa: F401
            Anchor as PackageAnchor,
        )
        from science2code import (
            Tier as PackageTier,
        )
        from science2code import (
            anchor_record,
            prepare,
            reanchor_record,
        )
        from science2code import (
            locate as package_locate,
        )
        from science2code import (
            normalise as package_normalise,
        )
        self.assertIs(PackageTier, Tier)
        self.assertIs(PackageAnchor, Anchor)
        self.assertIs(package_locate, locate)
        self.assertTrue(callable(package_normalise))
        self.assertTrue(callable(anchor_record))
        self.assertTrue(callable(reanchor_record))
        self.assertTrue(callable(prepare))

    def test_the_package_exposes_the_version_and_fingerprint(self):
        import science2code

        self.assertEqual(science2code.NORMALISER_VERSION, "norm/1.1.0")
        self.assertIsInstance(science2code.NORMALISER_FINGERPRINT, str)
        self.assertIsInstance(science2code.__version__, str)

    def test_the_core_imports_without_any_third_party_package(self):
        # anchor.py may import rapidfuzz, but only inside a try block, and the
        # default scorer must never need it.
        with open(s2c_anchor.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("try:", source.split("from rapidfuzz")[0][-40:])
        self.assertEqual(locate(PAPER_DOC, self.__class__.__name__).scorer,
                         "difflib")


class NoDashesInTheSourceTests(unittest.TestCase):
    """This project refuses an en dash or em dash anywhere in a .py file.

    The two characters are written as escapes rather than literals, because a
    test that spelled them out would be the very thing it forbids.
    """

    def test_neither_source_file_contains_a_dash_the_project_refuses(self):
        paths = [s2c_anchor.__file__, os.path.abspath(__file__)]
        for path in paths:
            with self.subTest(path=os.path.basename(path)):
                with open(path, encoding="utf-8") as handle:
                    source = handle.read()
                self.assertNotIn("\u2013", source, "en dash in %s" % path)
                self.assertNotIn("\u2014", source, "em dash in %s" % path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
