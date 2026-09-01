"""Tests for citation-marker detection, region reporting and the citance.

Inline fixtures only. No test here reads the real corpus, and every string a
test needs is written into the test, so a failure names the character sequence
that caused it rather than a filename.

The two long dashes are written as escapes throughout, because a literal one
would break the repo-wide rule that this project refuses them, and the
bracketed-marker pattern has to match them.

WHAT IS BEING TESTED, AND WHY IT IS NOT A THRESHOLD SWEEP
---------------------------------------------------------
Three claims, each with a measurement behind it, and one refusal:

  * bracketed markers, precision 72/72 on the hand-checked sample, so the
    pattern is pinned tightly and every widening has to fail a test;
  * strict author-year, 95.6% precision at 1.000 recall on 113 labelled
    matches, where the running-head filter carries 20 of the 26 naive false
    alarms, so each of the six running-head shapes gets its own test;
  * MARKER_STYLE_UNDETECTABLE, the label for the citation style pdftotext
    destroys, because before it existed such a paper answered "no marker near
    this span", which is a false negative wearing the shape of a finding;
  * and the refusal: nothing here says whose claim a sentence carries.
"""

from __future__ import annotations

import os
import types
import unittest

from science2code import envelope as env
from science2code import markers as M

# ---------------------------------------------------------------------------
# bracketed markers: the near-perfect half
# ---------------------------------------------------------------------------


class TestBracketedMarkers(unittest.TestCase):
    """Square brackets in this literature are essentially never anything else.

    72 of 72 hand-checked markers across 12 papers matched, with no false
    alarm. A pattern with that precision earns the right to stay strict, so
    these tests pin the shapes it does match and the shapes it must not.
    """

    def test_a_bracketed_number(self):
        self.assertEqual(1, M.count_bracketed("oranges are good [12]"))

    def test_a_bracketed_list_is_one_marker(self):
        self.assertEqual(1, M.count_bracketed("as reported [3, 4, 5]"))

    def test_a_bracketed_range_is_one_marker(self):
        self.assertEqual(1, M.count_bracketed("as reported [7-9]"))

    def test_a_range_written_with_an_en_dash_is_one_marker(self):
        # The raw document has not been through the punctuation fold when this
        # runs over raw characters, so both dash forms have to match.
        self.assertEqual(1, M.count_bracketed("as reported [7\u20139]"))

    def test_a_range_written_with_an_em_dash_is_one_marker(self):
        self.assertEqual(1, M.count_bracketed("as reported [7\u20149]"))

    def test_a_semicolon_separated_list_is_one_marker(self):
        self.assertEqual(1, M.count_bracketed("as reported [3; 4]"))

    def test_two_markers_in_one_sentence_are_two(self):
        self.assertEqual(2, M.count_bracketed("this [1] and that [2]"))

    def test_an_array_index_is_not_a_marker(self):
        self.assertEqual(0, M.count_bracketed("the value of x[i] rises"))

    def test_a_four_digit_number_is_not_a_marker(self):
        # Three digits is the ceiling. [2019] is a year in brackets, which is
        # not how anything in this literature cites.
        self.assertEqual(0, M.count_bracketed("published [2019] elsewhere"))

    def test_an_empty_bracket_is_not_a_marker(self):
        self.assertEqual(0, M.count_bracketed("an empty [] pair"))

    def test_prose_with_no_bracket_counts_zero(self):
        self.assertEqual(0, M.count_bracketed("a sentence with no citation"))

    def test_the_spans_index_the_text_they_were_found_in(self):
        text = "oranges are good [12] and so on"
        (start, end), = M.bracketed_spans(text)
        self.assertEqual("[12]", text[start:end])


# ---------------------------------------------------------------------------
# strict author-year: the half where the work is
# ---------------------------------------------------------------------------


class TestStrictAuthorYearAccepts(unittest.TestCase):
    """Recall was measured at 1.000, so every real shape has to match."""

    CASES = (
        "as shown (Smith et al., 2019)",
        "as shown (Smith et al. 2019)",
        "as shown (Lee & See, 2004)",
        "as shown (Goastellec and Pekari 2013)",
        "as shown (Smith, 2019)",
        "as shown (Trow 1962)",
        "as shown (Hong et al., 2023; Wu et al., 2024)",
        "Smith (2019) found",
        "Smith et al. (2019) found",
        "Smith and Jones (2019) found",
        "Smith & Jones (2019) found",
        "McDonald (2019) found",
        "Wang et al. (2023b) investigated",
        "as shown (Republic of Indonesia 2014b)",
    )

    def test_every_real_shape_is_counted(self):
        for text in self.CASES:
            with self.subTest(text=text):
                self.assertGreaterEqual(M.count_author_year(text), 1, text)


class TestStrictAuthorYearRejects(unittest.TestCase):
    """The naive pattern scored 77.0%. These are what it was getting wrong."""

    def test_a_bare_parenthesised_year_is_not_a_marker(self):
        # The single change that separates the strict rule from the naive one:
        # a parenthesis that merely ends in a year is not a citation.
        self.assertEqual(0, M.count_author_year("the study (2019) found"))

    def test_a_lower_case_word_before_the_year_is_not_a_name(self):
        self.assertEqual(0, M.count_author_year("published in the year (2019)"))

    def test_an_all_capitals_acronym_is_not_a_name_token(self):
        # "capitalised name token" means a surname shape. Admitting an acronym
        # turns a monetary aside into a citation.
        self.assertEqual(
            0,
            M.count_author_year(
                "the tower cost (equivalent to USD 120,000 in 2023)"
            ),
        )

    def test_a_commission_document_number_is_not_a_narrative_citation(self):
        self.assertEqual(0, M.count_author_year("Committee of the Regions COM(2019)168"))

    def test_a_grant_identifier_is_not_a_citation(self):
        self.assertEqual(0, M.count_author_year("the (JAES/2024/EVIL-AI) project"))

    def test_a_long_parenthetical_ending_in_a_year_is_not_a_citation(self):
        # Over 60 characters. In this corpus the long ones are table cells and
        # two-column text that pdftotext interleaved, which merely end in a
        # date. The ceiling is inherited from the pattern this replaces.
        self.assertEqual(
            0,
            M.count_author_year(
                "(a parenthetical clause running on at considerable length "
                "about matters unrelated to any citation whatsoever, 2019)"
            ),
        )


class TestTheRunningHeadFilter(unittest.TestCase):
    """20 of the 26 naive false alarms were journal running heads.

    pdftotext writes the running head of a modern journal PDF onto every page,
    so a paper with 40 pages carries 40 copies of a string that ends in a
    parenthesised year beside a capitalised name. That single family is most of
    what separates 77.0% precision from 95.6%, so each of the six shapes the
    filter looks for gets its own test.
    """

    REAL = "Scientific Data |   (2026) 13:936 | https://doi.org/10.1038/s41597"

    def test_the_real_running_head_from_the_corpus_is_rejected(self):
        self.assertEqual(0, M.count_author_year(self.REAL))

    def test_a_pipe_before_the_parenthetical_is_rejected(self):
        self.assertEqual(0, M.count_author_year("Nature Data | (2026) with Smith"))

    def test_a_volume_and_article_number_is_rejected(self):
        self.assertEqual(0, M.count_author_year("Smith (2026) 13:936 onwards"))

    def test_the_word_journal_beside_a_candidate_is_rejected(self):
        self.assertEqual(0, M.count_author_year("Journal of Fruit, Smith (2019)"))

    def test_a_page_range_token_beside_a_candidate_is_rejected(self):
        self.assertEqual(0, M.count_author_year("Smith (2019), pp. 12 to 30"))

    def test_a_doi_beside_a_candidate_is_rejected(self):
        self.assertEqual(0, M.count_author_year("Smith (2019) doi:10.1000/abc"))

    def test_a_url_beside_a_candidate_is_rejected(self):
        self.assertEqual(0, M.count_author_year("Smith (2019) https://example.org/x"))

    def test_a_www_address_beside_a_candidate_is_rejected(self):
        self.assertEqual(0, M.count_author_year("Smith (2019) www.example.org"))

    def test_the_word_journal_far_away_does_not_reject(self):
        # The window is 45 characters either side. A body sentence that happens
        # to use the word further off must still count its own citation, or the
        # filter would eat real markers.
        text = (
            "The journal that published it is of no interest here, and none of "
            "the following words matter at all, but the citation does (Smith "
            "et al., 2019)."
        )
        self.assertEqual(1, M.count_author_year(text))


class TestTheAcknowledgedUndecidableCase(unittest.TestCase):
    """1.8% of labelled matches are not decidable by any character test.

    A named document that carries a year is the same character sequence as a
    citation. This is recorded as a test rather than hidden, because a reader
    of this file should be able to see the residue rather than infer it.
    """

    def test_a_named_act_with_a_year_counts_as_a_marker(self):
        self.assertEqual(1, M.count_author_year("the National Research Act (1974)"))


class TestBothKindsTogether(unittest.TestCase):
    def test_the_two_counts_are_reported_apart_and_summed(self):
        text = "as reported [12] and also (Smith et al., 2019)"
        self.assertEqual(1, M.count_bracketed(text))
        self.assertEqual(1, M.count_author_year(text))
        self.assertEqual(2, M.count_markers(text))

    def test_a_standards_citation_is_counted_by_its_bracket(self):
        # "(ISO/IEC TR 24028:2020 [10])" carries no surname, so the strict
        # author-year rule declines it. It is not lost: the bracketed number
        # beside it is counted, which is why declining it costs no recall.
        text = "trustworthiness (ISO/IEC TR 24028:2020 [10]), and bias"
        self.assertEqual(0, M.count_author_year(text))
        self.assertEqual(1, M.count_bracketed(text))
        self.assertEqual(1, M.count_markers(text))

    def test_overlapping_matches_collapse_to_one_marker(self):
        for start, end in M.marker_spans("Smith (2019) reported [4]"):
            self.assertLess(start, end)
        self.assertEqual(2, M.count_markers("Smith (2019) reported [4]"))


# ---------------------------------------------------------------------------
# the citance
# ---------------------------------------------------------------------------


class TestSentenceSpan(unittest.TestCase):
    """The one primitive with a published precision of 1.00 behind it.

    Sarol et al., Bioinformatics 2024;40(7):btae420: the trivial "return the
    sentence containing the marker" baseline scored P 1.00 / R 0.90 / F1 0.94
    and beat a fine-tuned PubMedBERT. It is strong because it decides nothing.
    """

    TEXT = (
        "The first sentence has no marker. Oranges are good for you [12]. "
        "A third sentence closes the paragraph."
    )

    def sentence(self, text, start, end):
        span = M.sentence_span(text, start, end)
        self.assertIsNotNone(span)
        return text[span[0]:span[1]]

    def test_the_sentence_around_a_span_is_returned_whole(self):
        start = self.TEXT.index("Oranges")
        self.assertEqual(
            "Oranges are good for you [12].",
            self.sentence(self.TEXT, start, start + len("Oranges are good for you")),
        )

    def test_the_sentence_reaches_past_a_quote_that_stops_before_the_marker(self):
        # The case that used to be invisible. The caller located characters
        # that carry no marker; the sentence they sit in carries one.
        start = self.TEXT.index("Oranges")
        end = start + len("Oranges are good for you")
        self.assertNotIn("[12]", self.TEXT[start:end])
        self.assertIn("[12]", self.sentence(self.TEXT, start, end))

    def test_the_first_sentence_of_a_document_starts_at_zero(self):
        self.assertEqual((0, 33), M.sentence_span(self.TEXT, 4, 9))

    def test_et_al_does_not_end_a_sentence(self):
        text = "As Smith et al. showed, oranges are good [12]. Next."
        self.assertEqual(
            "As Smith et al. showed, oranges are good [12].",
            self.sentence(text, text.index("oranges"), text.index("oranges") + 7),
        )

    def test_an_abbreviation_does_not_end_a_sentence(self):
        text = "See Fig. 2 for the result, which is clear [7]. Next one."
        self.assertEqual(
            "See Fig. 2 for the result, which is clear [7].",
            self.sentence(text, text.index("result"), text.index("result") + 6),
        )

    def test_an_initial_does_not_end_a_sentence(self):
        text = "Work by J. A. Smith and others reported this [3]. Next."
        self.assertEqual(
            "Work by J. A. Smith and others reported this [3].",
            self.sentence(text, text.index("others"), text.index("others") + 6),
        )

    def test_a_decimal_point_does_not_end_a_sentence(self):
        text = "The score rose to 0.94 in that run [9]. Next."
        self.assertEqual(
            "The score rose to 0.94 in that run [9].",
            self.sentence(text, text.index("rose"), text.index("rose") + 4),
        )

    def test_a_final_sentence_with_no_terminator_runs_to_the_end(self):
        text = "One sentence ends here. A second one never terminates"
        self.assertEqual(
            "A second one never terminates",
            self.sentence(text, text.index("second"), text.index("second") + 6),
        )

    def test_the_span_never_shrinks_below_what_was_located(self):
        # The point of the sentence is to hand back MORE than was located.
        text = "A. B. C. D. E."
        span = M.sentence_span(text, 3, 11)
        self.assertLessEqual(span[0], 3)
        self.assertGreaterEqual(span[1], 11)

    def test_an_offset_outside_the_text_returns_nothing(self):
        self.assertIsNone(M.sentence_span("short", 99, 120))
        self.assertIsNone(M.sentence_span("", 0, 0))
        self.assertIsNone(M.sentence_span("short", 0, None))


# ---------------------------------------------------------------------------
# the style label, including the one that names a silent failure
# ---------------------------------------------------------------------------

BRACKETED_BODY = "Oranges are good [12] and pears are better [13]."
AUTHOR_YEAR_BODY = "Oranges are good (Smith et al., 2019) and pears (Jones 2020)."

#: A superscript-citing paper as pdftotext renders it. Every citation has been
#: glued onto the word before it, and not one of them is separable from GPT4 or
#: Section3 without the layout the extractor discarded.
SUPERSCRIPT_BODY = (
    "Semantic Units follow the Principles4 laid out earlier. The "
    "hallucination26,27 problem and FAIRness5,9 are treated at length, and "
    "the status25,26 of each entity36 is recorded."
)


class TestMarkerStyle(unittest.TestCase):
    """Five labels, and the one that matters is the absence of the other four."""

    def test_a_bracketed_body_is_numeric(self):
        self.assertEqual(
            M.MARKER_STYLE_NUMERIC_BRACKETED,
            M.marker_style(BRACKETED_BODY, True),
        )

    def test_an_author_year_body_is_author_year(self):
        self.assertEqual(
            M.MARKER_STYLE_AUTHOR_YEAR, M.marker_style(AUTHOR_YEAR_BODY, True)
        )

    def test_a_body_with_both_is_mixed(self):
        self.assertEqual(
            M.MARKER_STYLE_MIXED,
            M.marker_style(BRACKETED_BODY + " " + AUTHOR_YEAR_BODY, True),
        )

    def test_a_superscript_body_with_a_region_is_undetectable(self):
        """The most dangerous state in the system, now named.

        This paper cites 92 times. Not one of those citations survives
        extraction as anything a character test can see, so every count over it
        is zero. Reporting the zero without this label reads as "the author
        claimed this without citing anyone", which is a false negative shaped
        exactly like a positive finding.
        """
        self.assertEqual(0, M.count_markers(SUPERSCRIPT_BODY))
        self.assertEqual(
            M.MARKER_STYLE_UNDETECTABLE, M.marker_style(SUPERSCRIPT_BODY, True)
        )

    def test_the_same_body_with_no_region_is_not_assessed_instead(self):
        # Without a reference region there is no evidence the paper cites
        # anything at all, so the two cases cannot be told apart and the label
        # says so rather than picking one.
        self.assertEqual(
            M.MARKER_STYLE_NOT_ASSESSED, M.marker_style(SUPERSCRIPT_BODY, False)
        )

    def test_every_label_returned_is_in_the_closed_set(self):
        for body in (BRACKETED_BODY, AUTHOR_YEAR_BODY, SUPERSCRIPT_BODY, "", "x"):
            for detected in (True, False):
                with self.subTest(body=body[:20], detected=detected):
                    self.assertIn(M.marker_style(body, detected), M.MARKER_STYLES)

    def test_the_label_is_a_string_and_never_a_count_or_a_flag(self):
        style = M.marker_style(BRACKETED_BODY, True)
        self.assertIsInstance(style, str)
        self.assertNotIsInstance(style, bool)


# ---------------------------------------------------------------------------
# the local-block test
# ---------------------------------------------------------------------------

REFERENCE_BLOCK_DOCUMENT = (
    "This is body prose that carries a single year, 2019, and nothing else "
    "that resembles a bibliography at all.\n"
    "\n"
    "[1] J. A. Smith and K. Jones. A study of oranges. Journal of Fruit, "
    "40(7):936-948, 2019. doi:10.1000/abc\n"
    "[2] L. M. Brown and N. White. Warranted trust. Proceedings of the "
    "Conference on Trust, pp. 12 to 30, 2020.\n"
)


class TestTheLocalBlockTest(unittest.TestCase):
    """The second, independent reference-list signal, and why it is a warning.

    Measured: alone it catches 31 of 36 known reference-list hits, and the
    union with region membership catches 36 of 36. Its false-positive rate over
    1,640 genuine body offsets is 2.4%, on numbered lists and author-email
    footer blocks. A rate that size is worth reporting and far too large to act
    on unseen, which is what makes it a warning and not a verdict.
    """

    def signals(self, offset, text=REFERENCE_BLOCK_DOCUMENT):
        return M.local_block_signals(text, offset)

    def test_an_offset_in_a_reference_block_raises_the_warning(self):
        result = self.signals(REFERENCE_BLOCK_DOCUMENT.index("A study of oranges"))
        self.assertEqual(M.LOCAL_BLOCK_WARNING, result["signal"])
        self.assertGreaterEqual(result["signals_found"], M.LOCAL_BLOCK_MIN_SIGNALS)

    def test_all_four_signals_are_named_when_all_four_are_present(self):
        result = self.signals(REFERENCE_BLOCK_DOCUMENT.index("A study of oranges"))
        self.assertEqual(
            list(M.LOCAL_BLOCK_SIGNAL_NAMES), sorted(result["signals_named"],
                                                     key=M.LOCAL_BLOCK_SIGNAL_NAMES.index)
        )

    def test_body_prose_with_only_a_year_does_not_raise_the_warning(self):
        result = self.signals(REFERENCE_BLOCK_DOCUMENT.index("body prose"))
        self.assertEqual(M.LOCAL_BLOCK_CLEAR, result["signal"])
        self.assertEqual(["year"], result["signals_named"])

    def test_the_blank_line_bounds_the_block(self):
        # The body offset must not reach across the blank line into the
        # reference list, or every document with a bibliography would warn on
        # its last paragraph.
        result = self.signals(0)
        self.assertEqual(M.LOCAL_BLOCK_CLEAR, result["signal"])

    def test_a_single_initial_is_not_enough_to_name_the_signal(self):
        text = "The result is clear. See Fig. A. for the detail, in 2019.\n"
        self.assertNotIn("initials", self.signals(10, text)["signals_named"])

    def test_the_test_reports_that_it_did_not_run_when_there_is_no_text(self):
        result = M.local_block_signals("", None)
        self.assertEqual(M.LOCAL_BLOCK_NOT_ASSESSED, result["signal"])
        self.assertEqual(0, result["signals_found"])

    def test_every_label_returned_is_in_the_closed_set(self):
        for offset in (0, 50, 200, 10_000, -5):
            with self.subTest(offset=offset):
                self.assertIn(
                    self.signals(offset)["signal"], M.LOCAL_BLOCK_LABELS
                )

    def test_the_result_carries_a_count_and_never_a_flag(self):
        result = self.signals(REFERENCE_BLOCK_DOCUMENT.index("A study of oranges"))
        self.assertIsInstance(result["signals_found"], int)
        for value in result.values():
            self.assertNotIsInstance(value, bool)

    def test_the_normalised_rendering_cannot_be_used_for_this_test(self):
        """Why the test reads raw characters, pinned as a test rather than a note.

        The normaliser collapses every run of whitespace to one space, so a
        document that has been normalised has no blank lines left in it and no
        block can be delimited. Running this test on the wrong string would
        silently take the 700-character cap every time.
        """
        from science2code.normalise import normalise_text

        self.assertNotIn("\n\n", normalise_text(REFERENCE_BLOCK_DOCUMENT))
        self.assertIn("\n\n", REFERENCE_BLOCK_DOCUMENT)


# ---------------------------------------------------------------------------
# the region adapter
# ---------------------------------------------------------------------------


class RegionStub:
    """A stand-in for `science2code.region`, installed into sys.modules.

    CLEARLY MARKED AS A STUB. It exists so that the two states the real module
    cannot easily be made to produce, namely "this build does not carry the
    module" and "the module returned a label outside the closed set", are
    reachable from a test. Everything else in this file runs against the real
    module.
    """

    def __init__(self, status_name, start=None, version="region/stub"):
        self.REGION_VERSION = version
        self._status = types.SimpleNamespace(name=status_name)
        self._start = start

    def detect_region(self, text):
        return types.SimpleNamespace(
            start_char=self._start,
            end_char=None,
            status=self._status,
        )

    def classify_offset(self, region, offset):
        return self._status


class RegionAdapterTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = M._region_module

    def tearDown(self):
        M._region_module = self._saved

    def install(self, stub):
        M._region_module = lambda _s=stub: _s

    def absent(self):
        M._region_module = lambda: None


class TestTheRegionAdapter(RegionAdapterTestCase):
    def test_the_real_module_answers_for_a_document_with_a_bibliography(self):
        text = (
            "Body prose about oranges runs for a while here.\n\n"
            "References\n\n"
            "[1] J. A. Smith. A study of oranges. Journal of Fruit, 2019.\n"
            "[2] L. Brown. Warranted trust. Proceedings, pp. 12 to 30, 2020.\n"
        )
        status, start, version = M.region_status(text, 5)
        self.assertIn(status, M.REGION_STATUS_NAMES)
        self.assertIsInstance(version, str)
        if start is not None:
            self.assertIn(
                M.region_status(text, start + 20)[0],
                {
                    "IN_REFERENCE_REGION",
                    "IN_REFERENCE_REGION_UNCERTAIN",
                    "REGION_END_UNCERTAIN",
                },
            )

    def test_a_build_without_the_module_reports_region_unknown(self):
        self.absent()
        self.assertEqual("REGION_MODEL_ABSENT", M.region_available())
        status, start, version = M.region_status("some text", 3)
        self.assertEqual(M.REGION_UNKNOWN, status)
        self.assertIsNone(start)
        self.assertIsNone(version)

    def test_a_label_outside_the_closed_set_degrades_to_region_unknown(self):
        # A tier this server does not know must never fall through to a
        # default, because every available default is one of the wrong merges.
        self.install(RegionStub("SOMETHING_NOBODY_DOCUMENTED", start=10))
        self.assertEqual(M.REGION_UNKNOWN, M.region_status("text", 3)[0])

    def test_a_model_that_did_not_apply_bounds_no_region(self):
        self.install(RegionStub("REGION_MODEL_INAPPLICABLE", start=10))
        _status, start, _version = M.region_status("text", 3)
        self.assertIsNone(start)

    def test_a_stub_reporting_membership_is_passed_through(self):
        self.install(RegionStub("IN_REFERENCE_REGION", start=4))
        status, start, version = M.region_status("some text here", 6)
        self.assertEqual("IN_REFERENCE_REGION", status)
        self.assertEqual(4, start)
        self.assertEqual("region/stub", version)

    def test_a_module_that_raises_does_not_escape(self):
        class Exploding:
            REGION_VERSION = "region/stub"

            def detect_region(self, text):
                raise RuntimeError("the region model fell over")

            def classify_offset(self, region, offset):
                raise RuntimeError("and so did this one")

        self.install(Exploding())
        self.assertEqual(M.REGION_UNKNOWN, M.region_status("text", 3)[0])

    def test_availability_is_a_label_and_never_a_flag(self):
        self.assertIn(
            M.region_available(), {"REGION_MODEL_PRESENT", "REGION_MODEL_ABSENT"}
        )
        self.assertNotIsInstance(M.region_available(), bool)


# ---------------------------------------------------------------------------
# the two blocks
# ---------------------------------------------------------------------------


class TestTheCitationMarkerBlock(unittest.TestCase):
    TEXT = (
        "A first sentence carries nothing. Oranges are good for you [12]. "
        "A third sentence closes it."
    )

    def block(self, start, length, **kwargs):
        return M.citation_marker_block(self.TEXT, start, length, **kwargs)

    def test_a_marker_inside_the_span_is_counted_inside(self):
        start = self.TEXT.index("Oranges")
        block = self.block(start, len("Oranges are good for you [12]"))
        self.assertEqual(1, block["in_located_characters"])
        self.assertEqual(1, block["bracketed_in_located_characters"])
        self.assertEqual(0, block["in_surrounding_characters"])

    def test_a_marker_beside_the_span_is_counted_beside(self):
        start = self.TEXT.index("Oranges")
        block = self.block(start, len("Oranges are good for you"))
        self.assertEqual(0, block["in_located_characters"])
        self.assertEqual(1, block["in_surrounding_characters"])
        self.assertEqual(1, block["bracketed_in_surrounding_characters"])

    def test_the_two_kinds_are_counted_apart(self):
        text = "Oranges are good [12] and pears are better (Smith et al., 2019)."
        block = M.citation_marker_block(text, 0, len(text))
        self.assertEqual(1, block["bracketed_in_located_characters"])
        self.assertEqual(1, block["author_year_in_located_characters"])
        self.assertEqual(2, block["in_located_characters"])

    def test_the_block_carries_the_sentence_under_document_text(self):
        start = self.TEXT.index("Oranges")
        block = self.block(start, len("Oranges are good for you"))
        self.assertEqual(
            "Oranges are good for you [12].", block["sentence"]["document_text"]
        )
        interval = block["sentence"]["char_interval"]
        self.assertEqual("normalised_document", interval["basis"])
        self.assertEqual(
            self.TEXT[interval["start_pos"]:interval["end_pos"]],
            block["sentence"]["document_text"],
        )

    def test_the_sentence_key_is_document_text_so_the_exemption_need_not_widen(self):
        # A real paper may contain a word this server is forbidden to write.
        # Naming the key `document_text` puts the sentence under the exemption
        # that already exists rather than widening it, which is a decision
        # `test_the_exempt_key_list_has_not_grown` guards.
        text = "This paper demonstrates the effect clearly [4]."
        block = M.citation_marker_block(text, 0, len(text))
        self.assertIn("demonstrates", block["sentence"]["document_text"])
        self.assertIn("document_text", env.CALLER_OR_DOCUMENT_FIELDS)

    def test_the_block_names_the_window_it_looked_at(self):
        self.assertEqual(
            env.CITATION_CONTEXT_CHARS, self.block(0, 5)["context_chars"]
        )

    def test_the_style_travels_on_the_block(self):
        block = self.block(0, 5, style=M.MARKER_STYLE_UNDETECTABLE)
        self.assertEqual(M.MARKER_STYLE_UNDETECTABLE, block["marker_style"])

    def test_a_style_outside_the_closed_set_degrades_rather_than_travels(self):
        block = self.block(0, 5, style="MARKER_STYLE_INVENTED_BY_A_CALLER")
        self.assertEqual(M.MARKER_STYLE_NOT_ASSESSED, block["marker_style"])

    def test_a_block_with_no_offset_still_has_every_key(self):
        empty = M.citation_marker_block(self.TEXT, None, None)
        full = self.block(0, 5)
        self.assertEqual(set(full), set(empty))
        self.assertIsNone(empty["sentence"])

    def test_no_value_anywhere_in_the_block_is_a_boolean(self):
        block = self.block(self.TEXT.index("Oranges"), 24)
        for path, value in _walk(block):
            self.assertNotIsInstance(value, bool, path)

    def test_every_count_is_a_whole_number(self):
        block = self.block(self.TEXT.index("Oranges"), 24)
        for key, value in block.items():
            if key.endswith("_characters"):
                self.assertIsInstance(value, int, key)


class TestTheReferenceRegionBlock(unittest.TestCase):
    def block(self, status="OUTSIDE_REFERENCE_REGION", **kwargs):
        kwargs.setdefault("region_version", "region/1.0.0")
        return M.reference_region_block(status, **kwargs)

    def test_the_block_carries_the_status_and_the_vocabulary_it_came_from(self):
        block = self.block("IN_REFERENCE_REGION")
        self.assertEqual("IN_REFERENCE_REGION", block["status"])
        self.assertEqual(sorted(M.REGION_STATUS_NAMES), block["status_vocabulary"])

    def test_a_status_outside_the_vocabulary_degrades_to_region_unknown(self):
        self.assertEqual(M.REGION_UNKNOWN, self.block("MADE_UP")["status"])

    def test_the_two_signals_are_carried_side_by_side_and_never_merged(self):
        block = self.block(
            "OUTSIDE_REFERENCE_REGION",
            local_block=M.local_block_signals(
                REFERENCE_BLOCK_DOCUMENT,
                REFERENCE_BLOCK_DOCUMENT.index("A study of oranges"),
            ),
        )
        # Region membership says outside, the local-block test says
        # reference-like. Both readings survive into the response, because a
        # caller told "reference like: 0.83" cannot tell which one fired.
        self.assertEqual("OUTSIDE_REFERENCE_REGION", block["status"])
        self.assertEqual(M.LOCAL_BLOCK_WARNING, block["local_block"]["signal"])

    def test_the_block_carries_a_local_block_even_when_none_was_given(self):
        self.assertEqual(
            M.LOCAL_BLOCK_NOT_ASSESSED, self.block()["local_block"]["signal"]
        )

    def test_no_value_anywhere_in_the_block_is_a_boolean(self):
        for path, value in _walk(self.block("IN_REFERENCE_REGION")):
            self.assertNotIsInstance(value, bool, path)


# ---------------------------------------------------------------------------
# the refusal, and the language rules
# ---------------------------------------------------------------------------


def _walk(node, path="$"):
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, "%s.%s" % (path, key))
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            yield from _walk(value, "%s[%d]" % (path, i))


class TestNothingHereAssertsAttribution(unittest.TestCase):
    """The ceiling, enforced rather than promised.

    Trained human annotators judging citation accuracy agree at Cohen's kappa
    0.18 to 0.31. The one dedicated attribution study reaches alpha .654
    against a human ceiling of .806. A field naming whose claim a sentence
    carries would therefore be a number nobody can stand behind, printed in a
    shape a caller would act on, so `envelope.validate()` refuses one and this
    tests that both blocks pass it.
    """

    def blocks(self):
        text = "Oranges are good for you [12] and pears are better (Smith 2019)."
        return (
            M.citation_marker_block(text, 0, 29, style=M.MARKER_STYLE_MIXED),
            M.reference_region_block(
                "IN_REFERENCE_REGION",
                region_version="region/1.0.0",
                local_block=M.local_block_signals(REFERENCE_BLOCK_DOCUMENT, 200),
            ),
        )

    def test_no_key_in_either_block_names_the_owner_of_a_claim(self):
        for block in self.blocks():
            for path, _value in _walk(block):
                lowered = path.lower()
                for banned in env.ATTRIBUTION_FIELD_NAMES:
                    self.assertNotIn(banned, lowered, path)

    def test_both_blocks_survive_the_envelope_contract(self):
        marker_block, region_block = self.blocks()
        response = env.verbatim(
            env.Outcome.VERBATIM_EXACT,
            paper_id="paper-a",
            document_text="Oranges are good for you [12]",
            start_pos=0,
            end_pos=29,
            score=1.0,
            scorer="difflib",
            provenance_block=env.provenance("norm/1.1.0", "verify/1", "difflib"),
            citation_marker_block=marker_block,
            reference_region_block=region_block,
        )
        self.assertEqual(marker_block, response["citation_markers"])
        self.assertEqual(region_block, response["reference_region"])

    def test_the_module_docstring_records_the_agreement_numbers(self):
        doc = M.__doc__ or ""
        for fragment in ("0.18 to 0.31", ".654", ".806", "kappa"):
            self.assertIn(fragment, doc, fragment)

    def test_the_ceiling_sentence_travels_on_the_marker_block(self):
        block = M.citation_marker_block("Oranges are good [12].", 0, 22)
        self.assertEqual(M.ATTRIBUTION_CEILING, block["ceiling"])


class TestTheProseObeysTheServersOwnRules(unittest.TestCase):
    """Every string this module ships is held to rules 2 and 4 of the envelope.

    They are the server's own prose, and they ride on a response, so a banned
    verb in one of them would turn a count into an assertion just as surely as
    one in a reason would.
    """

    def strings(self):
        text = "Oranges are good for you [12]."
        blocks = (
            M.citation_marker_block(text, 0, len(text), style=M.MARKER_STYLE_MIXED),
            M.reference_region_block(
                "IN_REFERENCE_REGION",
                region_version="region/1.0.0",
                local_block=M.local_block_signals(REFERENCE_BLOCK_DOCUMENT, 200),
            ),
        )
        for block in blocks:
            for path, value in _walk(block):
                if isinstance(value, str) and not path.endswith(".document_text"):
                    yield path, value

    def test_no_shipped_string_contains_a_banned_term(self):
        for path, value in self.strings():
            for term in env.BANNED_TERMS:
                self.assertNotIn(term, value.lower(), "%s: %r" % (path, term))

    def test_no_shipped_string_uses_the_word_outside_an_outcome_name(self):
        for path, value in self.strings():
            self.assertTrue(env._verbatim_uses_are_outcome_names(value), path)

    def test_the_module_source_carries_neither_dash_the_project_refuses(self):
        with open(M.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("\u2013", source, "en dash in markers.py")
        self.assertNotIn("\u2014", source, "em dash in markers.py")

    def test_this_test_file_carries_neither_dash_either(self):
        with open(os.path.abspath(__file__), encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("\u2013", source)
        self.assertNotIn("\u2014", source)


class TestTheModuleStaysStdlibAndImportable(unittest.TestCase):
    def test_the_module_imports_no_third_party_package(self):
        import inspect

        source = inspect.getsource(M)
        for name in ("rapidfuzz", "fastmcp", "numpy", "regex"):
            self.assertNotIn("import %s" % name, source, name)

    def test_the_region_import_is_lazy_so_a_build_without_it_still_loads(self):
        import inspect

        source = inspect.getsource(M)
        head = source.split("def _region_module")[0]
        self.assertNotIn("import science2code.region", head)
        self.assertIn("import science2code.region", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
