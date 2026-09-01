"""Tests for the reference-region detector in science2code/region.py.

Stdlib unittest only, no third-party imports:

    python3 -m unittest discover -s tests -p "test_*.py"
    python3 tests/test_region.py

Every fixture in this file is a short inline string and every one of them is
INVENTED. No reference entry here is a real citation, no title is a real
title, and no author is a real person. Nothing here opens a corpus file: the
corpus the detector was measured on is gitignored and absent from a fresh
clone, so a test that read it would fail for every reader who does not hold
it.

The measured figures the module docstring quotes are therefore NOT retested
here. What is pinned here is the BEHAVIOUR those figures justify: which
headings are found, which are refused, that dilation is what makes the density
threshold reachable, that the region has an end and the end is the minimum of
two estimators, and that a failure to detect comes back as a label rather than
as a guess.
"""

import dataclasses
import os
import sys
import unittest

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from science2code import region as s2c_region  # noqa: E402
from science2code.region import (  # noqa: E402
    REGION_VERSION,
    Region,
    RegionStatus,
    classify_offset,
    detect_region,
)

# ---------------------------------------------------------------------------
# Fixture builders. All invented.
# ---------------------------------------------------------------------------

BODY = "\n".join([
    "The mechanism accepts a stream of characters and emits one label for",
    "each of them. We set out the mechanism, then report what it does on",
    "inputs that were built to break it, and where it declines to answer.",
    "There is no third path in which it guesses at an answer it cannot",
    "reach by looking at the characters it was handed.",
    "A label is not a score, and the difference is the whole point of the",
    "design, so nothing below ever hands back a number to be thresholded.",
])

APPENDIX = "\n".join([
    "Appendix A gives the full listing of the mechanism as it was run, and",
    "the paragraphs below walk through it one clause at a time so that a",
    "reader can follow what happens to a single character of input.",
    "The listing is prose rather than a table because a table would be",
    "clamped by the estimator and would tell us nothing about prose.",
    "We begin with the part that decides whether a line has been seen",
    "before, which is the only part that keeps any state at all.",
    "The state is a single counter and it is reset whenever the reader",
    "moves to a fresh page, so nothing survives a page break by accident.",
    "That choice costs a little accuracy at a page boundary and buys back",
    "a great deal of predictability everywhere else in the document.",
    "The next part decides what to do with a line that has been seen, and",
    "it does the simplest possible thing, which is to leave it alone.",
    "A reader who wants the longer argument for that choice will find it",
    "in the body of the paper rather than here in the appendix.",
    "What follows is a walk through the remaining clauses in order.",
    "The first clause looks at whether the line is empty of characters.",
    "The second clause looks at whether the line has any letters in it.",
    "The third clause looks at whether the letters form words at all.",
    "The fourth clause is the one that decides, and it decides by asking",
    "whether the answer is already known rather than by weighing anything.",
    "None of these clauses reads ahead, and none of them looks backward.",
    "That is what makes the whole walk a single pass over the input.",
    "A single pass is not required by anything, but it is easy to reason",
    "about and it means the cost is linear in the length of the input.",
    "The remaining paragraphs restate the argument for readers who came",
    "to the appendix first, which the editors tell us is most of them.",
    "We close by noting what the mechanism does not attempt to decide.",
    "It does not decide whether a passage supports a claim, and it never",
    "will, because that is a human judgement and not a string operation.",
    "Nor does it decide what a reader failed to look at in the first place.",
    "That question has no mechanical answer that we are aware of today.",
])


def wrapped_entries(count, start=1):
    """Reference entries in hanging-indent shape, three lines each.

    Line one carries the marker and the initials. Lines two and three are
    continuation lines of a long title and carry NO feature at all, which is
    the shape that makes dilation necessary: raw density over such a list is
    one in three.
    """
    out = []
    for i in range(start, start + count):
        out.append(
            "[%d] A. Author%d, B. Coauthor%d, On a made up topic that runs" % (i, i, i))
        out.append(
            "on and on across a wrapped line carrying no marker of its own")
        out.append(
            "and then a third wrapped line of the very same invented title")
    return out


def dense_entries(count, start=1):
    """One-line reference entries, four features each."""
    return [
        "[%d] A. Author%d, A made up title, in: Proceedings of the Invented"
        " Symposium, ACM, %d, pp. %d-%d." % (i, i, 2001 + (i % 20), i * 3, i * 3 + 9)
        for i in range(start, start + count)
    ]


def paper(heading="References", entries=None, tail="", body=BODY):
    """A whole invented document: body, heading, reference list, tail."""
    if entries is None:
        entries = wrapped_entries(9)
    parts = [body, "", heading]
    parts.extend(entries)
    if tail:
        parts.extend(["", tail])
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------


class TestApiShape(unittest.TestCase):

    def test_version_string(self):
        self.assertEqual(REGION_VERSION, "region/1.0.0")

    def test_region_is_a_frozen_dataclass(self):
        self.assertTrue(dataclasses.is_dataclass(Region))
        r = detect_region(paper())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            r.start_char = 0

    def test_region_fields_are_exactly_the_documented_six(self):
        names = [f.name for f in dataclasses.fields(Region)]
        self.assertEqual(names, [
            "start_char",
            "end_char",
            "density_end_char",
            "map_end_char",
            "heading_text",
            "status",
        ])

    def test_no_score_or_probability_ever_leaves_the_api(self):
        # The whole design refuses judgement calls. A caller handed a float
        # will invent a threshold for it, so no float may appear in a result.
        r = detect_region(paper())
        for f in dataclasses.fields(Region):
            value = getattr(r, f.name)
            self.assertNotIsInstance(value, float, f.name)
            self.assertNotIsInstance(value, bool, f.name)

    def test_status_vocabulary_is_closed_and_complete(self):
        self.assertEqual(
            sorted(m.name for m in RegionStatus),
            [
                "IN_REFERENCE_REGION",
                "IN_REFERENCE_REGION_UNCERTAIN",
                "OUTSIDE_REFERENCE_REGION",
                "REGION_END_UNCERTAIN",
                "REGION_MODEL_INAPPLICABLE",
                "REGION_UNKNOWN",
            ],
        )

    def test_detection_is_deterministic(self):
        text = paper()
        self.assertEqual(detect_region(text), detect_region(text))


class TestStartDetection(unittest.TestCase):

    def test_finds_a_plain_heading(self):
        text = paper()
        r = detect_region(text)
        self.assertEqual(r.status, RegionStatus.IN_REFERENCE_REGION)
        self.assertEqual(r.heading_text, "References")

    def test_offsets_index_into_the_string_that_was_passed_in(self):
        text = paper()
        r = detect_region(text)
        self.assertTrue(text[r.start_char:].startswith("References"))
        self.assertLessEqual(r.end_char, len(text))
        self.assertGreater(r.end_char, r.start_char)

    def test_all_caps_heading(self):
        r = detect_region(paper(heading="REFERENCES"))
        self.assertEqual(r.heading_text, "REFERENCES")

    def test_lower_case_heading_is_matched_because_it_is_the_common_form(self):
        # "References" is far more frequent in the wild than "REFERENCES";
        # a case sensitive matcher loses most of a corpus.
        r = detect_region(paper(heading="references"))
        self.assertEqual(r.status, RegionStatus.IN_REFERENCE_REGION)

    def test_kerned_heading_from_small_caps(self):
        # Small-caps kerning arrives from pdftotext as "R EFERENCES".
        r = detect_region(paper(heading="R EFERENCES"))
        self.assertEqual(r.heading_text, "R EFERENCES")

    def test_fully_letter_spaced_heading(self):
        r = detect_region(paper(heading="R E F E R E N C E S"))
        self.assertEqual(r.status, RegionStatus.IN_REFERENCE_REGION)

    def test_form_feed_prefixed_heading(self):
        # A form feed prefixes the heading whenever the list opens a page,
        # which is the common case rather than the exotic one.
        r = detect_region(paper(heading="\fReferences"))
        self.assertEqual(r.status, RegionStatus.IN_REFERENCE_REGION)

    def test_heading_after_a_column_gap_rather_than_at_line_start(self):
        # In reading order a heading that sat at the top of column two
        # arrives in the middle of a line, after the gap.
        text = paper(heading="end of the section.  References")
        r = detect_region(text)
        self.assertEqual(r.status, RegionStatus.IN_REFERENCE_REGION)
        self.assertTrue(text[r.start_char:].startswith("References"))

    def test_numbered_section_heading(self):
        for heading in ("5. REFERENCES", "5.1 REFERENCES", "V. REFERENCES",
                        "A) References", "12 References"):
            with self.subTest(heading=heading):
                r = detect_region(paper(heading=heading))
                self.assertEqual(r.status, RegionStatus.IN_REFERENCE_REGION, heading)

    def test_alternative_heading_words(self):
        for heading in ("Bibliography", "LITERATURE CITED", "Works Cited",
                        "Reference List"):
            with self.subTest(heading=heading):
                r = detect_region(paper(heading=heading))
                self.assertEqual(r.status, RegionStatus.IN_REFERENCE_REGION, heading)

    def test_earliest_accepted_candidate_wins(self):
        text = "\n".join([
            BODY,
            "",
            "References",
            "\n".join(dense_entries(30)),
            "",
            "Bibliography",
            "\n".join(dense_entries(30, start=100)),
        ])
        r = detect_region(text)
        self.assertEqual(r.heading_text, "References")
        self.assertEqual(r.start_char, text.index("References"))

    def test_a_word_boundary_is_required_after_the_heading(self):
        r = detect_region(paper(heading="Referencesplus"))
        self.assertEqual(r.status, RegionStatus.REGION_UNKNOWN)


class TestRejectionFilters(unittest.TestCase):
    """Both filters are needed. Each catches a different false heading."""

    def _prose_line_before_a_real_list(self, line):
        # The line under test is followed by a perfectly good reference list,
        # so if it is NOT rejected it will validate and win as the start.
        return "\n".join([BODY, "", line, "\n".join(wrapped_entries(9))]) + "\n"

    def test_line_reject_quoted_term(self):
        text = self._prose_line_before_a_real_list(
            'The field labelled "references" is left empty by the exporter.')
        self.assertEqual(detect_region(text).status, RegionStatus.REGION_UNKNOWN)

    def test_line_reject_reference_format(self):
        text = self._prose_line_before_a_real_list(
            "Bibliography reference format is set by the publisher template.")
        self.assertEqual(detect_region(text).status, RegionStatus.REGION_UNKNOWN)

    def test_line_reject_cross_references(self):
        text = self._prose_line_before_a_real_list(
            "Cross-references are resolved before the document is rendered.")
        self.assertEqual(detect_region(text).status, RegionStatus.REGION_UNKNOWN)

    def test_line_reject_dot_leaders_from_a_table_of_contents(self):
        # A contents line POINTS AT the reference section; it is not it.
        text = self._prose_line_before_a_real_list("References ............ 31")
        self.assertEqual(detect_region(text).status, RegionStatus.REGION_UNKNOWN)

    def test_line_reject_references_followed_by_a_connective(self):
        for line in ("References to the standard are collected below.",
                     "References of this kind are common in the corpus.",
                     "References in the appendix are numbered separately.",
                     "References for each chapter follow the chapter.",
                     "References were checked twice by both authors.",
                     "References and citations are handled the same way."):
            with self.subTest(line=line):
                text = self._prose_line_before_a_real_list(line)
                self.assertEqual(
                    detect_region(text).status, RegionStatus.REGION_UNKNOWN, line)

    def test_match_reject_when_a_stopword_follows_within_three_characters(self):
        for line in ("References, which the reader may skip, are listed later.",
                     "References: section 4 holds them all.",
                     "References  used by the tool are held in one folder.",
                     "References that matter are marked with a dagger."):
            with self.subTest(line=line):
                text = self._prose_line_before_a_real_list(line)
                self.assertEqual(
                    detect_region(text).status, RegionStatus.REGION_UNKNOWN, line)

    def test_a_genuine_heading_survives_both_filters(self):
        text = self._prose_line_before_a_real_list("References")
        self.assertEqual(detect_region(text).status, RegionStatus.IN_REFERENCE_REGION)


class TestDensityValidation(unittest.TestCase):

    def test_a_heading_followed_by_prose_is_not_a_start(self):
        # The word survives both rejection filters here, so only the density
        # check can refuse it. Nothing that follows is reference shaped.
        text = "\n".join([BODY, "", "References", APPENDIX]) + "\n"
        self.assertEqual(detect_region(text).status, RegionStatus.REGION_UNKNOWN)

    def test_abstention_is_a_label_and_carries_no_offsets(self):
        r = detect_region("\n".join([BODY, "", APPENDIX]))
        self.assertEqual(r.status, RegionStatus.REGION_UNKNOWN)
        self.assertIsNone(r.start_char)
        self.assertIsNone(r.end_char)
        self.assertIsNone(r.density_end_char)
        self.assertIsNone(r.map_end_char)
        self.assertIsNone(r.heading_text)

    def test_empty_and_blank_input(self):
        for text in ("", "\n\n\n", "   \n \n"):
            with self.subTest(text=repr(text)):
                self.assertEqual(
                    detect_region(text).status, RegionStatus.REGION_UNKNOWN)

    def test_a_heading_at_the_very_end_has_no_window_to_validate_against(self):
        self.assertEqual(
            detect_region(BODY + "\n\nReferences\n").status,
            RegionStatus.REGION_UNKNOWN,
        )


class TestDilationIsLoadBearing(unittest.TestCase):
    """Without dilation the threshold is unreachable on a real reference list.

    A hanging-indent entry puts every feature on its first line. The
    continuation lines carry a fragment of a title and score zero. Raw
    density over such a list is about one in three, far below the acceptance
    threshold, so an undilated detector abstains on ordinary papers.
    """

    def test_raw_density_of_a_wrapped_list_is_below_the_threshold(self):
        lines = wrapped_entries(9)
        raw = [s2c_region._is_reference_like(line) for line in lines]
        self.assertLess(sum(raw) / len(raw), s2c_region.DENSITY_ACCEPT)

    def test_dilated_density_of_the_same_list_clears_the_threshold(self):
        lines = wrapped_entries(9)
        raw = [s2c_region._is_reference_like(line) for line in lines]
        dilated = s2c_region._dilate(raw)
        self.assertGreaterEqual(
            sum(dilated) / len(dilated), s2c_region.DENSITY_ACCEPT)

    def test_and_therefore_the_wrapped_list_is_detected(self):
        self.assertEqual(
            detect_region(paper()).status, RegionStatus.IN_REFERENCE_REGION)

    def test_dilate_widens_by_exactly_one_in_each_direction(self):
        self.assertEqual(
            s2c_region._dilate([False, False, True, False, False]),
            [False, True, True, True, False],
        )
        self.assertEqual(s2c_region._dilate([True]), [True])
        self.assertEqual(s2c_region._dilate([False, False]), [False, False])
        self.assertEqual(s2c_region._dilate([]), [])


class TestLineFeatures(unittest.TestCase):

    def test_year_feature(self):
        self.assertTrue(s2c_region._is_reference_like("published in 1997 somewhere"))
        self.assertTrue(s2c_region._is_reference_like("a note from 2024 on this"))
        self.assertFalse(s2c_region._is_reference_like("published in 2077 somewhere"))
        self.assertFalse(s2c_region._is_reference_like("the number 199 appears"))

    def test_bibliographic_tokens(self):
        for line in ("pp. 12", "vol. 4", "no. 9", "et al. say so",
                     "doi somewhere", "10.1234/abcd", "In: a book",
                     "In Proc of it", "Proceedings of it", "Journal of it",
                     "arXiv thing", "LNCS thing", "https://example.invalid",
                     "IEEE thing", "ACM thing", "Springer thing", "Eds. of it",
                     "Ed. of it", "Press of it", "Conference on it",
                     "Workshop on it", "Symposium on it", "Trans. on it",
                     "pages 5"):
            with self.subTest(line=line):
                self.assertTrue(s2c_region._is_reference_like(line), line)

    def test_entry_markers(self):
        self.assertTrue(s2c_region._is_reference_like("[12] a thing"))
        self.assertTrue(s2c_region._is_reference_like("12. Author of a thing"))
        self.assertTrue(s2c_region._is_reference_like("a gap  [7] a thing"))

    def test_initials(self):
        self.assertTrue(s2c_region._is_reference_like("Smith, J. and someone"))
        self.assertTrue(s2c_region._is_reference_like("Smith, J., someone"))
        self.assertFalse(s2c_region._is_reference_like("Smith and someone else"))

    def test_ordinary_prose_carries_no_feature(self):
        for line in BODY.split("\n"):
            with self.subTest(line=line):
                self.assertFalse(s2c_region._is_reference_like(line), line)

    def test_feature_count_saturates_at_four(self):
        line = ("[3] A. Author, B. Coauthor, A title, in: Proceedings of a"
                " Conference, ACM, 2011, pp. 4-9, doi 10.1234/abcd.")
        self.assertEqual(s2c_region._feature_count(line), 4)
        self.assertEqual(s2c_region._feature_count("nothing at all here"), 0)

    def test_probability_table_has_one_entry_per_feature_count(self):
        self.assertEqual(len(s2c_region.MAP_PROBABILITIES), 5)


class TestTableClamp(unittest.TestCase):
    """A results table scores like a reference list under feature counting.

    Years, decimals and capital initials are exactly what a table of numbers
    is made of, which is why the published detector this estimator comes from
    reports a References heading F1 of 0.994 against a body F1 of 0.578.
    """

    def test_a_ruled_row_is_a_table_row(self):
        self.assertTrue(s2c_region._looks_like_table_row("| 2011 | 0.83 | 0.91 |"))

    def test_columns_separated_by_a_gap_are_a_table_row(self):
        self.assertTrue(
            s2c_region._looks_like_table_row("Model A     2011     0.83     0.91"))

    def test_a_mostly_numeric_row_is_a_table_row(self):
        self.assertTrue(s2c_region._looks_like_table_row("1 2011 0.83 0.91 12 44"))

    def test_a_reference_entry_is_not_a_table_row(self):
        for line in dense_entries(3) + wrapped_entries(3):
            with self.subTest(line=line):
                self.assertFalse(s2c_region._looks_like_table_row(line), line)

    def test_prose_is_not_a_table_row(self):
        for line in BODY.split("\n") + APPENDIX.split("\n"):
            with self.subTest(line=line):
                self.assertFalse(s2c_region._looks_like_table_row(line), line)

    def test_blank_is_not_a_table_row(self):
        self.assertFalse(s2c_region._looks_like_table_row("   "))


class TestRegionEnd(unittest.TestCase):

    def test_the_region_ends_before_the_appendix(self):
        # Nearly a quarter of real papers have content after the reference
        # list, so "everything after the heading" is unsound. The conservative
        # end must stop at or before the first line of that content.
        text = paper(entries=dense_entries(40), tail=APPENDIX)
        r = detect_region(text)
        self.assertIsNotNone(r.start_char)
        appendix_at = text.index("Appendix A gives")
        self.assertLessEqual(r.end_char, appendix_at)

    def test_appendix_prose_is_not_inside_the_region(self):
        text = paper(entries=dense_entries(40), tail=APPENDIX)
        r = detect_region(text)
        deep_in_appendix = text.index("It does not decide whether a passage")
        self.assertEqual(
            classify_offset(r, deep_in_appendix),
            RegionStatus.OUTSIDE_REFERENCE_REGION,
        )

    def test_end_char_is_the_minimum_of_the_two_estimators(self):
        for text in (paper(entries=dense_entries(40), tail=APPENDIX),
                     paper(entries=wrapped_entries(20)),
                     paper(entries=dense_entries(60))):
            with self.subTest():
                r = detect_region(text)
                self.assertEqual(
                    r.end_char, min(r.density_end_char, r.map_end_char))

    def test_both_estimators_are_reported_separately(self):
        r = detect_region(paper(entries=dense_entries(40), tail=APPENDIX))
        self.assertIsInstance(r.density_end_char, int)
        self.assertIsInstance(r.map_end_char, int)

    def test_a_list_that_runs_to_the_end_of_file_ends_at_the_end_of_file(self):
        text = paper(entries=dense_entries(40))
        r = detect_region(text)
        self.assertGreater(r.end_char, len(text) - 400)

    def test_an_interrupted_list_is_not_ended_at_the_interruption(self):
        # Elsevier lists are interrupted mid-list by CRediT statements,
        # competing-interest declarations and repeated page headers. A single
        # failing window must not end the region: the rest of the
        # bibliography would fall outside it.
        interruption = [
            "CRediT authorship contribution statement",
            "The first author wrote the draft and ran every experiment in it.",
            "The second author reviewed the draft and revised the argument.",
            "Declaration of competing interest",
            "The authors declare that they have no competing interest here.",
        ]
        entries = dense_entries(20) + interruption + dense_entries(20, start=100)
        text = paper(entries=entries)
        r = detect_region(text)
        resumed_at = text.index("[100]")
        self.assertEqual(
            classify_offset(r, resumed_at), RegionStatus.IN_REFERENCE_REGION)

    def test_a_sustained_drop_does_end_the_region(self):
        text = paper(entries=dense_entries(40), tail=APPENDIX)
        r = detect_region(text)
        self.assertLess(r.density_end_char, len(text))


class TestClassifyOffset(unittest.TestCase):

    def _region(self):
        return Region(
            start_char=1000,
            end_char=2000,
            density_end_char=3000,
            map_end_char=2000,
            heading_text="References",
            status=RegionStatus.IN_REFERENCE_REGION,
        )

    def test_before_the_start_is_outside(self):
        self.assertEqual(
            classify_offset(self._region(), 999),
            RegionStatus.OUTSIDE_REFERENCE_REGION)

    def test_at_the_start_is_inside(self):
        self.assertEqual(
            classify_offset(self._region(), 1000), RegionStatus.IN_REFERENCE_REGION)

    def test_inside_the_conservative_region(self):
        self.assertEqual(
            classify_offset(self._region(), 1999), RegionStatus.IN_REFERENCE_REGION)

    def test_between_the_two_ends_is_uncertain(self):
        for offset in (2000, 2500, 2999):
            with self.subTest(offset=offset):
                self.assertEqual(
                    classify_offset(self._region(), offset),
                    RegionStatus.IN_REFERENCE_REGION_UNCERTAIN)

    def test_past_both_ends_is_outside(self):
        self.assertEqual(
            classify_offset(self._region(), 3000),
            RegionStatus.OUTSIDE_REFERENCE_REGION)

    def test_unknown_region_answers_unknown_for_every_offset(self):
        r = Region(None, None, None, None, None, RegionStatus.REGION_UNKNOWN)
        for offset in (0, 500, 10 ** 9):
            with self.subTest(offset=offset):
                self.assertEqual(classify_offset(r, offset), RegionStatus.REGION_UNKNOWN)

    def test_inapplicable_model_answers_inapplicable_for_every_offset(self):
        r = Region(None, None, None, None, "References",
                   RegionStatus.REGION_MODEL_INAPPLICABLE)
        self.assertEqual(
            classify_offset(r, 4242), RegionStatus.REGION_MODEL_INAPPLICABLE)

    def test_end_uncertain_region_still_classifies_its_conservative_interior(self):
        r = dataclasses.replace(
            self._region(), status=RegionStatus.REGION_END_UNCERTAIN)
        self.assertEqual(classify_offset(r, 1500), RegionStatus.IN_REFERENCE_REGION)
        self.assertEqual(
            classify_offset(r, 2500), RegionStatus.IN_REFERENCE_REGION_UNCERTAIN)

    def test_classify_offset_returns_a_label_and_never_a_number(self):
        self.assertIsInstance(
            classify_offset(self._region(), 1500), RegionStatus)


class TestEndUncertain(unittest.TestCase):

    def test_widely_diverging_estimators_are_reported_as_such(self):
        # A list in the style that puts full first names and no entry marker
        # on every line. Each line carries a year and nothing else, so the
        # density estimator sees a reference list running to the end of file
        # while the MAP estimator, which needs two features before a line
        # pays for itself, stops almost immediately. The two disagree, and
        # saying so is the honest answer.
        sparse = [
            "Ada Author%d and Ben Coauthor%d. %d. A made up title of a work."
            % (i, i, 2001 + (i % 20))
            for i in range(1, 41)
        ]
        r = detect_region(paper(entries=sparse))
        self.assertEqual(r.status, RegionStatus.REGION_END_UNCERTAIN)
        self.assertGreater(
            abs(r.density_end_char - r.map_end_char),
            s2c_region.END_DIVERGENCE_FRACTION * (r.end_char - r.start_char),
        )

    def test_the_conservative_end_is_still_taken_when_the_ends_diverge(self):
        sparse = [
            "Ada Author%d and Ben Coauthor%d. %d. A made up title of a work."
            % (i, i, 2001 + (i % 20))
            for i in range(1, 41)
        ]
        r = detect_region(paper(entries=sparse))
        self.assertEqual(r.end_char, min(r.density_end_char, r.map_end_char))

    def test_agreeing_estimators_give_a_settled_region(self):
        r = detect_region(paper(entries=dense_entries(40)))
        self.assertEqual(r.status, RegionStatus.IN_REFERENCE_REGION)


class TestModelInapplicable(unittest.TestCase):
    """A monograph with a bibliography per chapter is not one region."""

    def test_many_validated_headings_spread_through_a_book(self):
        chapters = []
        for c in range(5):
            chapters.append(BODY)
            chapters.append("")
            chapters.append("References")
            chapters.extend(dense_entries(12, start=1 + c * 100))
            chapters.append("")
        text = "\n".join(chapters)
        r = detect_region(text)
        self.assertEqual(r.status, RegionStatus.REGION_MODEL_INAPPLICABLE)
        self.assertIsNone(r.start_char)
        self.assertIsNone(r.end_char)

    def test_one_paper_with_one_list_is_not_declared_inapplicable(self):
        r = detect_region(paper(entries=dense_entries(30)))
        self.assertEqual(r.status, RegionStatus.IN_REFERENCE_REGION)


class TestTheErrorThisModuleExistsToPrevent(unittest.TestCase):
    """A paper's bibliography contains the TITLES of other papers.

    So a title string occurs inside a paper that merely cites it, and a
    verbatim hit there reads as "this paper said this" unless somebody can
    say where in the document the hit landed. This is the recorded error.
    All names and titles below are invented.
    """

    TITLE = "A systematic mapping of invented methods for invented compliance"

    def _document(self):
        body = "\n".join([
            "Related work is thin on the ground in this area of the field.",
            "Redmond and colleagues surveyed the methods that were then in",
            "use and reported that most of them shared a single assumption.",
            "We take a different route and do not share that assumption.",
        ])
        entries = (
            dense_entries(9)
            + ["[10] O. Redmond, D. Vantree, %s, Requir. Eng. 24 (2019)." % self.TITLE]
            + dense_entries(20, start=11)
        )
        return "\n".join([body, "", "References"] + entries) + "\n"

    def test_the_cited_title_inside_the_bibliography_is_labelled_as_such(self):
        text = self._document()
        r = detect_region(text)
        at = text.index(self.TITLE)
        self.assertEqual(classify_offset(r, at), RegionStatus.IN_REFERENCE_REGION)

    def test_a_body_mention_of_the_same_author_is_outside_the_region(self):
        text = self._document()
        r = detect_region(text)
        at = text.index("Redmond and colleagues")
        self.assertEqual(classify_offset(r, at), RegionStatus.OUTSIDE_REFERENCE_REGION)

    def test_the_two_offsets_get_different_labels(self):
        text = self._document()
        r = detect_region(text)
        self.assertNotEqual(
            classify_offset(r, text.index(self.TITLE)),
            classify_offset(r, text.index("Redmond and colleagues")),
        )


class TestThresholdsAreNotToBeQuietlyLowered(unittest.TestCase):
    """Abstention is the designed behaviour, not a gap to be closed.

    The acceptance threshold was swept from 0.70 down to 0.05 on the papers
    the detector abstains on and no value yields a defensible boundary. What
    a lower threshold DOES buy is a related-work section, which is dense in
    years and author initials, accepted as the bibliography. These tests
    exist so that a maintainer who lowers the constant sees a red test rather
    than a quietly larger recall number.
    """

    def test_start_threshold_is_still_seventy_percent(self):
        self.assertEqual(s2c_region.DENSITY_ACCEPT, 0.70)

    def test_start_window_is_still_twenty_two_lines(self):
        self.assertEqual(s2c_region.START_WINDOW_LINES, 22)

    def test_end_window_floor_and_run_are_unchanged(self):
        self.assertEqual(s2c_region.END_WINDOW_LINES, 25)
        self.assertEqual(s2c_region.END_DENSITY_FLOOR, 0.34)
        self.assertEqual(s2c_region.END_RUN_WINDOWS, 3)

    def test_map_table_clamp_is_unchanged(self):
        self.assertEqual(s2c_region.MAP_TABLE_CLAMP, 0.25)
        self.assertEqual(
            s2c_region.MAP_PROBABILITIES, (0.10, 0.35, 0.62, 0.82, 0.93))

    def test_a_related_work_section_is_not_accepted_as_a_bibliography(self):
        # Dense in years, initials and venue names, and it is NOT the
        # bibliography. Nothing here is a reference heading, so nothing here
        # may be reported as the start of a region.
        related = "\n".join([
            "Redmond et al. (2011) reported a Journal result on this in IEEE",
            "venues, and Vantree, A. (2014) followed it up at a Conference.",
            "A. Author (2019) and B. Coauthor (2020) both used arXiv for it.",
        ] * 10)
        text = "\n".join([BODY, "", "Related work", related]) + "\n"
        self.assertEqual(detect_region(text).status, RegionStatus.REGION_UNKNOWN)


if __name__ == "__main__":
    unittest.main()
