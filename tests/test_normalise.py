"""Tests for science2code/normalise.py.

Stdlib unittest only, no third-party imports, so this runs on a fresh clone
with no virtualenv and nothing installed:

    python3 -m unittest discover -s tests -p "test_*.py"
    python3 tests/test_normalise.py

EVERY fixture in this file is a short inline string. Nothing here opens a
corpus file. The corpus-scale numbers that justify each stage live in the
module docstring, were measured on a private and non-redistributable set of
scientific PDFs, and are not re-runnable by a reader; what this file pins is
the BEHAVIOUR those numbers were measured on, which is re-runnable by anyone.

Non-ASCII fixtures are written as \\u escapes on purpose. The characters under
test are frequently invisible (zero width space, soft hyphen, word joiner) or
easy for an editor to silently rewrite (curly quotes, dashes), and an escape
cannot be mangled by a copy and paste.
"""

import ast
import os
import sys
import unicodedata
import unittest
from importlib import import_module

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from science2code.normalise import (  # noqa: E402
    ALL_STAGES,
    FINGERPRINT_UNAVAILABLE,
    INVISIBLE,
    MATCHFORM_VERSION,
    NORMALISER_FINGERPRINT,
    NORMALISER_VERSION,
    PUNCT_FOLD,
    S0_INVISIBLE,
    S1_NFKC,
    S2_PUNCTUATION,
    S3_DEHYPHENATE,
    S4_WHITESPACE,
    fingerprint_source,
    match_fold,
    match_form,
    match_form_with_map,
    normalise,
    normalise_text,
)

# Fixtures reused across several cases.
ZWSP = "\u200b"
SOFT_HYPHEN = "\u00ad"
BOM = "\ufeff"
MATH_ITALIC_X = "\U0001d465"
MATH_ITALIC_MU = "\U0001d707"
GREEK_MU = "\u03bc"


class TestS0InvisibleStrip(unittest.TestCase):
    """S0 removes the format characters that NFKC leaves in place."""

    def test_zero_width_space_is_stripped(self):
        self.assertEqual(normalise_text("h" + ZWSP + "tt" + ZWSP + "ps"), "https")

    def test_soft_hyphen_is_stripped(self):
        self.assertEqual(normalise_text("re" + SOFT_HYPHEN + "quire"), "require")

    def test_byte_order_mark_is_stripped(self):
        self.assertEqual(normalise_text(BOM + "abc"), "abc")

    def test_every_listed_invisible_is_stripped(self):
        for ch in sorted(INVISIBLE):
            with self.subTest(char=hex(ord(ch))):
                self.assertEqual(normalise_text("a" + ch + "b"), "ab")

    def test_zero_width_joiner_and_non_joiner_are_stripped(self):
        self.assertEqual(normalise_text("a\u200cb\u200dc"), "abc")

    def test_word_joiner_and_mongolian_vowel_separator_are_stripped(self):
        self.assertEqual(normalise_text("a\u2060b\u180ec"), "abc")

    def test_url_interleaved_with_zero_width_spaces_is_recovered(self):
        # The worst shape seen in the corpus: one zero width space between
        # every character of a URL. This is why S0 is not optional.
        raw = ZWSP.join("https://example.org/a")
        self.assertEqual(normalise_text(raw), "https://example.org/a")

    def test_nfkc_alone_does_not_remove_a_zero_width_space(self):
        # The reason S0 exists at all. If NFKC removed these, S0 would be dead
        # code; it does not, because U+200B has no compatibility decomposition.
        self.assertIn(ZWSP, unicodedata.normalize("NFKC", "a" + ZWSP + "b"))

    def test_disabling_s0_keeps_the_zero_width_space(self):
        without_s0 = [s for s in ALL_STAGES if s != S0_INVISIBLE]
        self.assertIn(ZWSP, normalise_text("a" + ZWSP + "b", stages=without_s0))

    def test_s0_runs_before_composition(self):
        # An invisible sitting between a base letter and its combining mark
        # must not stop the two composing. Stripping first is what guarantees
        # it: cluster first and the mark would be orphaned.
        raw = "a" + ZWSP + "\u0301"
        self.assertEqual(normalise_text(raw), "\u00e1")


class TestS1Nfkc(unittest.TestCase):
    """S1 is NFKC and not NFC, and it is applied cluster-wise."""

    def test_mathematical_italic_x_folds_to_ascii_x(self):
        self.assertEqual(normalise_text(MATH_ITALIC_X + " = 1"), "x = 1")

    def test_mathematical_italic_mu_folds_to_greek_mu_not_latin_u(self):
        # U+1D707 maps to U+03BC, not to "u". A fold that produced a Latin
        # letter here would corrupt every quoted formula that uses mu.
        result = normalise_text(MATH_ITALIC_MU)
        self.assertEqual(result, GREEK_MU)
        self.assertEqual(unicodedata.name(result), "GREEK SMALL LETTER MU")

    def test_plain_greek_letters_survive_as_greek(self):
        greek = "\u03b1\u03b2\u03bc\u03c3"
        self.assertEqual(normalise_text(greek), greek)

    def test_fi_ligature_decomposes(self):
        self.assertEqual(normalise_text("\ufb01ve"), "five")

    def test_all_latin_ligatures_decompose(self):
        self.assertEqual(normalise_text("\ufb00\ufb01\ufb02\ufb03\ufb04"),
                         "fffiflffiffl")

    def test_combining_acute_composes(self):
        self.assertEqual(normalise_text("Anto\u0301n"), "Ant\u00f3n")

    def test_no_break_space_becomes_a_space(self):
        self.assertEqual(normalise_text("a\u00a0b"), "a b")

    def test_superscript_two_is_lossy_and_that_is_documented(self):
        self.assertEqual(normalise_text("x\u00b2"), "x2")

    def test_masculine_ordinal_is_lossy_and_that_is_documented(self):
        self.assertEqual(normalise_text("4\u00ba"), "4o")

    def test_disabling_s1_leaves_the_math_alphanumeric_alone(self):
        without_s1 = [s for s in ALL_STAGES if s != S1_NFKC]
        self.assertEqual(normalise_text(MATH_ITALIC_X, stages=without_s1),
                         MATH_ITALIC_X)

    def test_disabling_s1_still_applies_canonical_composition(self):
        # Documented semantics: the cluster pass falls back to NFC, so an
        # ablation of S1 isolates the compatibility fold rather than removing
        # normalisation altogether.
        without_s1 = [s for s in ALL_STAGES if s != S1_NFKC]
        self.assertEqual(normalise_text("Anto\u0301n", stages=without_s1),
                         "Ant\u00f3n")


class TestClusterwiseNfkcMatchesWholeString(unittest.TestCase):
    """Cluster-wise NFKC must be indistinguishable from whole-string NFKC.

    This is the property that lets the index map exist at all. It held on all
    44 corpus documents; these fixtures pin the hard cases inline.
    """

    FIXTURES = (
        "plain ascii text",
        "Anto\u0301n Garci\u0301a",
        "e\u0301\u0327 stacked marks",
        MATH_ITALIC_X + MATH_ITALIC_MU + " formula",
        "\ufb01\ufb02 ligatures",
        "\u00b2\u00b3\u00ba\u00aa superscripts",
        "\u30ab\uff9e",              # kana plus halfwidth voiced sound mark
        "\uff76\uff9e\uff8a\uff9f",  # halfwidth kana plus voiced marks
        "\u1100\u1161\u11a8",        # conjoining Hangul jamo, L V T
        "\u1100\u1161",              # conjoining Hangul jamo, L V
        "\u0e01\u0e34\u0e49",        # Thai with marks
        "\u0915\u094d\u0937",        # Devanagari conjunct
        "\u0928\u093f",              # Devanagari with a spacing mark
        "\u0ba4\u0bcb",              # Tamil vowel sign that is a starter
        "\u0dc0\u0dcf",              # Sinhala vowel sign that is a starter
        "\u0301 leading combining mark",
        "\u0384 tonos that decomposes to space plus mark",
        "a\u0384b",
        "",
        " ",
    )

    def test_clusterwise_equals_whole_string_nfkc(self):
        for text in self.FIXTURES:
            with self.subTest(text=ascii(text)):
                self.assertEqual(normalise_text(text, stages=(S1_NFKC,)),
                                 unicodedata.normalize("NFKC", text))

    def test_clusterwise_equals_whole_string_nfc_when_s1_is_off(self):
        for text in self.FIXTURES:
            with self.subTest(text=ascii(text)):
                self.assertEqual(normalise_text(text, stages=()),
                                 unicodedata.normalize("NFC", text))

    def test_every_starter_second_composite_is_covered_by_the_cluster_rule(self):
        """Enumerate rather than trust.

        The cluster rule splits before a starter. Any canonical composite
        whose SECOND character is a starter would therefore be missed unless
        the rule catches that character some other way. Walk the whole Unicode
        decomposition table and prove each such character is handled.
        """
        missed = []
        for code in range(0x110000):
            ch = chr(code)
            decomp = unicodedata.decomposition(ch)
            if not decomp or decomp.startswith("<"):
                continue
            parts = decomp.split()
            if len(parts) != 2:
                continue
            second = chr(int(parts[1], 16))
            if unicodedata.combining(second) != 0:
                continue  # a real combining mark, blocked by the mark test
            pair = parts[0] and chr(int(parts[0], 16)) + second
            if normalise_text(pair, stages=(S1_NFKC,)) != unicodedata.normalize("NFKC", pair):
                missed.append((hex(code), hex(ord(second))))
        self.assertEqual(missed, [])


class TestS2PunctuationFold(unittest.TestCase):
    """S2 folds typographic punctuation to the ASCII an agent would type."""

    def test_curly_double_quotes(self):
        self.assertEqual(normalise_text("These \u201crules\u201d are"),
                         'These "rules" are')

    def test_curly_apostrophe(self):
        self.assertEqual(normalise_text("company\u2019s"), "company's")

    def test_en_dash(self):
        self.assertEqual(normalise_text("741\u2013742"), "741-742")

    def test_em_dash(self):
        self.assertEqual(normalise_text("a\u2014b"), "a-b")

    def test_minus_sign(self):
        self.assertEqual(normalise_text("\u22121"), "-1")

    def test_non_breaking_hyphen(self):
        self.assertEqual(normalise_text("e\u2011mail"), "e-mail")

    def test_every_dash_variant_folds_to_ascii_hyphen(self):
        for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u2043\ufe58\ufe63\uff0d":
            with self.subTest(char=hex(ord(ch))):
                self.assertEqual(normalise_text("a" + ch + "b"), "a-b")

    def test_ellipsis_expands_to_three_dots(self):
        self.assertEqual(normalise_text("a\u2026b"), "a...b")

    def test_fraction_slash_becomes_a_solidus(self):
        self.assertEqual(normalise_text("1\u20442"), "1/2")

    def test_greek_question_mark_becomes_a_semicolon(self):
        self.assertEqual(normalise_text("a\u037eb"), "a;b")

    def test_angle_quotes_become_ascii_double_quotes(self):
        self.assertEqual(normalise_text("\u00abcite\u00bb"), '"cite"')

    def test_exotic_spaces_become_one_ascii_space(self):
        for ch in "\u00a0\u2000\u2003\u2009\u200a\u202f\u205f\u3000":
            with self.subTest(char=hex(ord(ch))):
                self.assertEqual(normalise_text("a" + ch + "b"), "a b")

    def test_soft_hyphen_is_not_in_the_punctuation_table(self):
        # S0 owns U+00AD. Folding it to a visible hyphen here would make an
        # ablation of S0 measure two things at once.
        self.assertNotIn(SOFT_HYPHEN, PUNCT_FOLD)

    def test_disabling_s2_keeps_the_curly_quote(self):
        without_s2 = [s for s in ALL_STAGES if s != S2_PUNCTUATION]
        self.assertEqual(normalise_text("\u201ca\u201d", stages=without_s2),
                         "\u201ca\u201d")


class TestS3Dehyphenate(unittest.TestCase):
    """S3 rejoins a word broken across a single line break."""

    def test_soft_break_is_joined(self):
        self.assertEqual(normalise_text("require-\nments"), "requirements")

    def test_join_tolerates_spaces_on_both_sides_of_the_break(self):
        self.assertEqual(normalise_text("estab-  \n   lished"), "established")

    def test_join_tolerates_a_carriage_return(self):
        self.assertEqual(normalise_text("require-\r\nments"), "requirements")

    def test_a_real_compound_is_joined_too_and_that_is_the_known_loss(self):
        # S3 cannot tell a soft break from a compound. It joins both; the T2
        # match form is what recovers the compound.
        self.assertEqual(normalise_text("well-\nknown"), "wellknown")

    def test_hyphen_before_a_capital_is_kept(self):
        self.assertEqual(normalise_text("LLM-\nBased"), "LLM- Based")

    def test_a_blank_line_is_never_jumped(self):
        # The blank line means page furniture came between, so this is not a
        # continuation. "408" here is a page number.
        self.assertEqual(normalise_text("stud-\n\n408\n\nied"), "stud- 408 ied")

    def test_a_form_feed_is_never_jumped(self):
        self.assertEqual(normalise_text("stud-\n\x0cied"), "stud- ied")

    def test_a_digit_range_is_untouched(self):
        self.assertEqual(normalise_text("2010-\n2015"), "2010- 2015")

    def test_a_hyphen_after_a_digit_is_untouched(self):
        self.assertEqual(normalise_text("3-\nyear"), "3- year")

    def test_no_line_break_means_no_join(self):
        self.assertEqual(normalise_text("well-known"), "well-known")

    def test_a_leading_hyphen_is_kept(self):
        self.assertEqual(normalise_text("-\nrest"), "- rest")

    def test_two_joins_in_a_row(self):
        self.assertEqual(normalise_text("re-\nquire-\nments"), "requirements")

    def test_disabling_s3_keeps_the_hyphen(self):
        without_s3 = [s for s in ALL_STAGES if s != S3_DEHYPHENATE]
        self.assertEqual(normalise_text("require-\nments", stages=without_s3),
                         "require- ments")


class TestS3KeepsADashThatIsNotAHyphen(unittest.TestCase):
    """A dash at a line break is punctuation, not a syllable break.

    S2 folds every dash to the ASCII hyphen, after which S3 could no longer
    tell one from the other and deleted both. An em dash at the end of a line
    then fused the words either side of it: a document reading "achieve\u2014\\nin
    which case" rendered as "achievein which case", a word that is in no paper,
    and every quote spanning that point was unreachable at T1 and at T2,
    because the characters the quote holds were not in the rendering at all.
    Found on a real corpus: 29 such sites across 14 of 45 documents.

    The line break still goes, because the page shows no break there. The dash
    stays, because the page shows a dash.
    """

    def test_em_dash_at_a_line_break_keeps_the_dash(self):
        self.assertEqual(normalise_text("achieve\u2014\nin which"), "achieve-in which")

    def test_en_dash_at_a_line_break_keeps_the_dash(self):
        self.assertEqual(normalise_text("human\u2013\nrobot"), "human-robot")

    def test_minus_sign_at_a_line_break_keeps_the_dash(self):
        self.assertEqual(normalise_text("x\u2212\ny"), "x-y")

    def test_the_quote_a_reader_would_type_now_matches_the_document(self):
        # The reader sees one line, so this is what they type. Before the fix
        # the document rendered "achievein" and this could not match.
        document = normalise_text("difficult to achieve\u2014\nin which case")
        typed = normalise_text("difficult to achieve\u2014in which case")
        self.assertEqual(document, typed)

    def test_a_true_hyphen_at_a_line_break_is_still_deleted(self):
        self.assertEqual(normalise_text("require-\nments"), "requirements")

    def test_unicode_hyphen_at_a_line_break_is_still_deleted(self):
        # U+2010 HYPHEN is a hyphen, so it can be a line-break hyphen.
        self.assertEqual(normalise_text("require\u2010\nments"), "requirements")

    def test_a_dash_away_from_a_line_break_is_untouched(self):
        self.assertEqual(normalise_text("achieve\u2014in which"), "achieve-in which")

    def test_a_dash_before_a_capital_keeps_the_break_as_a_space(self):
        # The lowercase condition is unchanged: without it there is no reason
        # to think the next line continues this word at all.
        self.assertEqual(normalise_text("achieve\u2014\nIn which"), "achieve- In which")

    def test_the_index_map_still_lands_on_the_dash(self):
        source = "human\u2013\nrobot"
        text, index_map = normalise(source)
        self.assertEqual(len(text), len(index_map))
        self.assertEqual(source[index_map[text.index("-")]], "\u2013")


class TestS4WhitespaceCollapse(unittest.TestCase):
    """S4 collapses every whitespace run to one space and strips the ends."""

    def test_doubled_space_collapses(self):
        self.assertEqual(normalise_text("a  b"), "a b")

    def test_mixed_whitespace_run_collapses(self):
        self.assertEqual(normalise_text("a  \n\n  b\t\tc"), "a b c")

    def test_form_feed_collapses(self):
        self.assertEqual(normalise_text("a\n\x0c\nb"), "a b")

    def test_vertical_tab_collapses(self):
        self.assertEqual(normalise_text("a\x0b\x0bb"), "a b")

    def test_ends_are_stripped(self):
        self.assertEqual(normalise_text("  a b  \n"), "a b")

    def test_an_all_whitespace_string_becomes_empty(self):
        self.assertEqual(normalise_text("  \n\t\x0c "), "")

    def test_disabling_s4_keeps_the_run(self):
        without_s4 = [s for s in ALL_STAGES if s != S4_WHITESPACE]
        self.assertEqual(normalise_text("a  b", stages=without_s4), "a  b")


class TestCaseIsPreserved(unittest.TestCase):
    """normalise() never casefolds. That belongs to match_form() alone."""

    def test_mixed_case_survives(self):
        self.assertEqual(normalise_text("Long-Term Mean Value"),
                         "Long-Term Mean Value")

    def test_sharp_s_is_not_expanded_by_the_normaliser(self):
        self.assertEqual(normalise_text("Stra\u00dfe"), "Stra\u00dfe")

    def test_final_sigma_survives(self):
        self.assertEqual(normalise_text("\u03bf\u03c2"), "\u03bf\u03c2")


class TestIndexMapInvariants(unittest.TestCase):
    """The properties every caller of normalise() is entitled to rely on."""

    CASES = (
        "",
        "plain ascii",
        "These \u201crules\u201d are   often\nprecursors",
        "re" + SOFT_HYPHEN + "quire-\nments",
        "h" + ZWSP + "tt" + ZWSP + "ps://example.org",
        MATH_ITALIC_X + " = " + MATH_ITALIC_MU + "\u00b2",
        "  leading and trailing  \n",
        "a\u2026b\u2014c\u00a0d",
        "Anto\u0301n \u201cGarci\u0301a\u201d",
        "stud-\n\n408\n\nied",
        "\u30ab\uff9e\u1100\u1161",
        "\u200b\u200b\u200b",
        "\n\n\n",
    )

    def test_map_length_equals_output_length(self):
        for raw in self.CASES:
            with self.subTest(text=ascii(raw)):
                text, index_map = normalise(raw)
                self.assertEqual(len(index_map), len(text))

    def test_map_is_monotone_non_decreasing(self):
        for raw in self.CASES:
            with self.subTest(text=ascii(raw)):
                _, index_map = normalise(raw)
                self.assertTrue(
                    all(index_map[i] <= index_map[i + 1]
                        for i in range(len(index_map) - 1)))

    def test_every_map_value_is_a_valid_index_into_the_input(self):
        for raw in self.CASES:
            with self.subTest(text=ascii(raw)):
                _, index_map = normalise(raw)
                self.assertTrue(all(0 <= v < len(raw) for v in index_map))

    def test_the_map_holds_under_every_stage_subset(self):
        # 32 stage subsets times the case list. The ablation must not be able
        # to produce a map a caller cannot trust.
        for mask in range(1 << len(ALL_STAGES)):
            stages = tuple(s for i, s in enumerate(ALL_STAGES) if mask >> i & 1)
            for raw in self.CASES:
                with self.subTest(stages=stages, text=ascii(raw)):
                    text, index_map = normalise(raw, stages=stages)
                    self.assertEqual(len(index_map), len(text))
                    self.assertTrue(all(0 <= v < len(raw) for v in index_map))
                    self.assertTrue(
                        all(index_map[i] <= index_map[i + 1]
                            for i in range(len(index_map) - 1)))

    def test_the_map_points_at_the_source_character(self):
        raw = "These \u201crules\u201d are   often\nprecursors"
        text, index_map = normalise(raw)
        self.assertEqual(raw[index_map[text.index('"')]], "\u201c")

    def test_an_expansion_maps_every_output_character_to_one_source_offset(self):
        text, index_map = normalise("\u2026")
        self.assertEqual(text, "...")
        self.assertEqual(index_map, [0, 0, 0])

    def test_a_composition_maps_to_the_first_character_of_the_cluster(self):
        text, index_map = normalise("Anto\u0301n")
        self.assertEqual(text, "Ant\u00f3n")
        self.assertEqual(raw_span(index_map, text, "Anto\u0301n", "\u00f3"), "o")

    def test_an_untouched_run_keeps_a_one_to_one_map(self):
        raw = "abc def"
        text, index_map = normalise(raw)
        self.assertEqual(text, raw)
        self.assertEqual(index_map, list(range(len(raw))))

    def test_the_map_survives_a_dehyphenation(self):
        raw = "require-\nments"
        text, index_map = normalise(raw)
        self.assertEqual(text, "requirements")
        # The "m" of "ments" must still point at the "m" in the input.
        self.assertEqual(raw[index_map[text.index("ments")]], "m")

    def test_the_map_survives_a_zero_width_strip(self):
        raw = "h" + ZWSP + "ttps"
        text, index_map = normalise(raw)
        self.assertEqual(text, "https")
        for k, ch in enumerate(text):
            self.assertEqual(raw[index_map[k]], ch)


def raw_span(index_map, text, raw, needle):
    """The single raw character behind `needle`'s first output character."""
    return raw[index_map[text.index(needle)]]


class TestIdempotence(unittest.TestCase):
    """Normalising normalised text must be a no-op."""

    CASES = TestIndexMapInvariants.CASES + (
        "These \u201crules\u201d  are",
        "well-\nknown and re-\nquired",
        "\ufb01ve \u00b2 \u00ba \u2026",
    )

    def test_normalise_is_idempotent(self):
        for raw in self.CASES:
            with self.subTest(text=ascii(raw)):
                once = normalise(raw)[0]
                self.assertEqual(normalise(once)[0], once)

    def test_the_index_map_of_normalised_text_is_the_identity(self):
        for raw in self.CASES:
            with self.subTest(text=ascii(raw)):
                once, _ = normalise(raw)
                again, index_map = normalise(once)
                self.assertEqual(index_map, list(range(len(again))))

    def test_match_form_is_idempotent(self):
        for raw in self.CASES:
            with self.subTest(text=ascii(raw)):
                once = match_form(raw)
                self.assertEqual(match_form(once), once)


class TestMatchForm(unittest.TestCase):
    """The T2 relaxed form: intra-word hyphens deleted, then casefolded."""

    def test_it_recovers_a_compound_the_extractor_destroyed(self):
        # The measured case: pdftotext's own de-hyphenation dropped the hyphen
        # of "long-\nterm" and emitted "longterm". The quote says "long-term".
        haystack = "readings classified as longterm in table 6"
        needle = "long-term"
        self.assertNotIn(normalise_text(needle), normalise_text(haystack))
        self.assertIn(match_form(needle), match_form(haystack))

    def test_it_matches_in_the_other_direction_too(self):
        haystack = "readings classified as long-term in table 6"
        self.assertIn(match_form("longterm"), match_form(haystack))

    def test_it_casefolds(self):
        self.assertEqual(match_form("Long-Term Mean"), "longterm mean")

    def test_casefold_expands_the_sharp_s(self):
        self.assertEqual(match_form("Stra\u00dfe"), "strasse")

    def test_a_hyphen_between_a_letter_and_a_digit_is_kept(self):
        self.assertEqual(match_form("Table 6-1"), "table 6-1")

    def test_a_leading_or_trailing_hyphen_is_kept(self):
        self.assertEqual(match_form("- a -"), "- a -")

    def test_it_normalises_first(self):
        self.assertEqual(match_form("\u201cLong\u2011Term\u201d  Mean"),
                         '"longterm" mean')

    def test_it_returns_a_plain_string(self):
        self.assertIsInstance(match_form("anything"), str)

    def test_match_fold_ablation_can_disable_the_hyphen_fold(self):
        norm = normalise_text("long-term")
        self.assertEqual(match_fold(norm, hyphen_fold=False)[0], "long-term")

    def test_match_fold_ablation_can_disable_the_case_fold(self):
        norm = normalise_text("Long-Term")
        self.assertEqual(match_fold(norm, case_fold=False)[0], "LongTerm")

    def test_it_recovers_a_hyphen_the_caller_never_rejoined(self):
        # The measured case, from the other side. The corpus holds a
        # reading-order extraction, which already rejoined "action-\nable"
        # into "actionable". The caller ran their own extractor, or copied
        # from a viewer, and hands over "action- able" with the break still
        # in it. Same damage, same word, different extractor.
        haystack = "an actionable tool for implementing AI ethics"
        needle = "an action- able tool"
        self.assertNotIn(normalise_text(needle), normalise_text(haystack))
        self.assertIn(match_form(needle), match_form(haystack))

    def test_the_three_forms_of_the_same_damaged_word_all_agree(self):
        forms = ("an action- able tool", "an actionable tool", "an action-able tool")
        folded = {match_form(f) for f in forms}
        self.assertEqual(folded, {"an actionable tool"})

    def test_a_spaced_dash_between_two_words_is_kept(self):
        # Punctuation between whole words, not a broken word. The character
        # before the hyphen is a space, so the rule does not fire.
        self.assertEqual(match_form("the result - in other words"),
                         "the result - in other words")

    def test_a_hyphen_with_a_space_before_a_digit_is_kept(self):
        self.assertEqual(match_form("pages 2010- 2015"), "pages 2010- 2015")

    def test_the_fold_jumps_exactly_one_space_and_no_more(self):
        # Asked of match_fold directly, because match_form normalises first
        # and S4 has already collapsed every whitespace run to one space by
        # the time the fold sees it. One space is that collapsed line break.
        self.assertEqual(match_fold("action- able")[0], "actionable")
        self.assertEqual(match_fold("action-  able")[0], "action-  able")

    def test_a_word_split_without_a_hyphen_still_does_not_match(self):
        # The fold deletes a hyphen; it never joins two words that had none.
        self.assertNotEqual(match_form("action able"), match_form("actionable"))

    def test_match_fold_ablation_disables_the_spaced_hyphen_too(self):
        norm = normalise_text("action- able")
        self.assertEqual(match_fold(norm, hyphen_fold=False)[0], "action- able")

    def test_match_fold_agrees_with_match_form(self):
        for raw in TestIndexMapInvariants.CASES:
            with self.subTest(text=ascii(raw)):
                self.assertEqual(match_fold(normalise(raw)[0])[0],
                                 match_form(raw))


class TestMatchFormWithMap(unittest.TestCase):
    """The T2 form still has to be able to name a raw span."""

    def test_map_length_equals_output_length(self):
        for raw in TestIndexMapInvariants.CASES:
            with self.subTest(text=ascii(raw)):
                text, index_map = match_form_with_map(raw)
                self.assertEqual(len(index_map), len(text))

    def test_map_is_monotone_and_in_range(self):
        for raw in TestIndexMapInvariants.CASES:
            with self.subTest(text=ascii(raw)):
                _, index_map = match_form_with_map(raw)
                self.assertTrue(all(0 <= v < len(raw) for v in index_map))
                self.assertTrue(
                    all(index_map[i] <= index_map[i + 1]
                        for i in range(len(index_map) - 1)))

    def test_a_t2_hit_maps_back_to_the_raw_offset(self):
        raw = 'The \u201cLong-Term\u201d  system'
        text, index_map = match_form_with_map(raw)
        self.assertEqual(text, 'the "longterm" system')
        self.assertEqual(raw[index_map[text.index("longterm")]], "L")
        self.assertEqual(raw[index_map[text.index("system")]], "s")

    def test_the_text_agrees_with_match_form(self):
        for raw in TestIndexMapInvariants.CASES:
            with self.subTest(text=ascii(raw)):
                self.assertEqual(match_form_with_map(raw)[0], match_form(raw))


class TestStageSelection(unittest.TestCase):
    """`stages=` is the ablation handle. It has to be strict about its input."""

    def test_the_default_is_every_stage(self):
        raw = "\u201cwell-\nknown\u201d  " + ZWSP + MATH_ITALIC_X
        self.assertEqual(normalise_text(raw),
                         normalise_text(raw, stages=ALL_STAGES))

    def test_an_unknown_stage_is_refused(self):
        with self.assertRaises(ValueError):
            normalise("a", stages=(0, 1, 9))

    def test_a_negative_stage_is_refused(self):
        with self.assertRaises(ValueError):
            normalise("a", stages=(-1,))

    def test_a_non_iterable_stage_set_is_refused(self):
        with self.assertRaises(TypeError):
            normalise("a", stages=3)

    def test_stages_accepts_any_iterable(self):
        self.assertEqual(normalise_text("a  b", stages=[0, 1, 2, 3, 4]),
                         normalise_text("a  b", stages={4, 3, 2, 1, 0}))

    def test_an_empty_stage_set_is_canonical_composition_only(self):
        self.assertEqual(normalise_text("a  \u201cb\u201d", stages=()),
                         "a  \u201cb\u201d")


class TestPurity(unittest.TestCase):
    """No file I/O, no global state, no surprises."""

    def test_the_module_imports_nothing_outside_the_standard_library(self):
        # A fresh clone has no virtualenv. A third-party import here would
        # make the normaliser, and therefore every stored anchor, unreadable.
        module = import_module("science2code.normalise")
        with open(module.__file__, encoding="utf-8") as handle:
            source = handle.read()
        imported = []
        for line in source.splitlines():
            if line.startswith("import "):
                imported.append(line.split()[1].split(".")[0])
            elif line.startswith("from ") and " import " in line:
                imported.append(line.split()[1].split(".")[0])
        # Asserted against the interpreter's own list of standard library
        # module names rather than against a hand written one. A hand written
        # list fails whenever a stdlib import is legitimately swapped, which
        # says nothing about purity and trains a reader to edit the assertion
        # without reading it.
        self.assertTrue(imported, "no imports were parsed out of the module")
        outside = sorted(set(imported) - set(sys.stdlib_module_names))
        self.assertEqual([], outside)

    def test_repeated_calls_agree(self):
        raw = "These \u201crules\u201d  are well-\nknown"
        first = normalise(raw)
        for _ in range(3):
            self.assertEqual(normalise(raw), first)

    def test_the_fold_tables_are_not_mutated_by_a_call(self):
        before = dict(PUNCT_FOLD)
        normalise("\u201ca\u2014b\u2026\u201d")
        self.assertEqual(PUNCT_FOLD, before)

    def test_an_ablated_call_does_not_leak_into_the_next_default_call(self):
        raw = "\u201ca\u201d  b"
        expected = normalise(raw)
        normalise(raw, stages=())
        normalise(raw, stages=(S0_INVISIBLE,))
        self.assertEqual(normalise(raw), expected)

    def test_normalise_refuses_bytes(self):
        with self.assertRaises(TypeError):
            normalise(b"abc")

    def test_normalise_refuses_none(self):
        with self.assertRaises(TypeError):
            normalise(None)

    def test_match_fold_refuses_bytes(self):
        with self.assertRaises(TypeError):
            match_fold(b"abc")


class TestVersionContract(unittest.TestCase):
    """The version strings are stored in anchors, so their shape is a contract."""

    def test_normaliser_version_value(self):
        # norm/1.1.0, not norm/1.0.0: S3 used to delete an em dash, en dash or
        # minus sign that fell at a line break, fusing the two words either
        # side of it. Fixing that moved offsets, and the documented rule is to
        # bump the minor digit for a change to stage behaviour.
        self.assertEqual(NORMALISER_VERSION, "norm/1.1.0")

    def test_matchform_version_value(self):
        # match/1.1.0, not match/1.0.0: the T2 fold now also deletes a hyphen
        # that still has its line break beside it, "action- able", which is
        # what a caller who ran their own extractor hands over.
        self.assertEqual(MATCHFORM_VERSION, "match/1.1.0")

    def test_versions_are_strings_with_a_namespace_and_three_digits(self):
        for version, prefix in ((NORMALISER_VERSION, "norm"),
                                (MATCHFORM_VERSION, "match")):
            with self.subTest(version=version):
                self.assertIsInstance(version, str)
                namespace, _, number = version.partition("/")
                self.assertEqual(namespace, prefix)
                self.assertEqual(len(number.split(".")), 3)
                self.assertTrue(all(part.isdigit()
                                    for part in number.split(".")))

    def test_the_documented_import_surface_exists(self):
        # Other modules code against exactly this line.
        from science2code.normalise import (  # noqa: F401
            NORMALISER_VERSION as version,
        )
        from science2code.normalise import (
            match_form as mf,
        )
        from science2code.normalise import (
            normalise as n,
        )
        self.assertTrue(callable(n) and callable(mf) and isinstance(version, str))

    def test_normalise_returns_a_tuple_of_text_and_map(self):
        result = normalise("abc")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], list)
        self.assertTrue(all(isinstance(v, int) for v in result[1]))


class TestNormaliserFingerprint(unittest.TestCase):
    """The backstop for a forgotten version bump.

    NORMALISER_VERSION is a hand-maintained string literal. If someone changes
    the behaviour of the normaliser and forgets to edit that literal, every
    anchor stored afterwards records a version it does not have and no record
    ever reads as stale: the staleness mechanism the whole design rests on
    fails SILENTLY. The fingerprint is what makes that failure loud, so these
    are the tests that keep the fingerprint honest.
    """

    def test_the_fingerprint_exists_and_is_a_short_hex_digest(self):
        self.assertIsInstance(NORMALISER_FINGERPRINT, str)
        self.assertNotEqual(NORMALISER_FINGERPRINT, FINGERPRINT_UNAVAILABLE)
        self.assertEqual(len(NORMALISER_FINGERPRINT), 12)
        self.assertTrue(
            all(c in "0123456789abcdef" for c in NORMALISER_FINGERPRINT))

    def test_the_fingerprint_is_stable_across_calls(self):
        source = "x = 1\ndef f(a):\n    return a + 1\n"
        self.assertEqual(fingerprint_source(source), fingerprint_source(source))

    def test_the_fingerprint_matches_the_module_source_on_disk(self):
        module = import_module("science2code.normalise")
        with open(module.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertEqual(fingerprint_source(source), NORMALISER_FINGERPRINT)

    def test_a_comment_change_does_not_move_the_fingerprint(self):
        # Otherwise every typo fix would mark the whole corpus stale, and a
        # warning that fires constantly is a warning nobody reads.
        before = "x = 1  # one\ndef f(a):\n    return a + 1\n"
        after = "x = 1  # the first integer\ndef f(a):\n    return a + 1\n"
        self.assertEqual(fingerprint_source(before), fingerprint_source(after))

    def test_a_docstring_change_does_not_move_the_fingerprint(self):
        before = 'def f(a):\n    """One."""\n    return a + 1\n'
        after = 'def f(a):\n    """Quite a different explanation."""\n    return a + 1\n'
        self.assertEqual(fingerprint_source(before), fingerprint_source(after))

    def test_a_module_docstring_change_does_not_move_the_fingerprint(self):
        before = '"""Module."""\nx = 1\n'
        after = '"""A module, described at length."""\nx = 1\n'
        self.assertEqual(fingerprint_source(before), fingerprint_source(after))

    def test_blank_lines_and_reformatting_do_not_move_the_fingerprint(self):
        before = "def f(a):\n    return a + 1\n"
        after = "\n\ndef f(\n    a,\n):\n\n    return a + 1\n"
        self.assertEqual(fingerprint_source(before), fingerprint_source(after))

    def test_a_changed_literal_moves_the_fingerprint(self):
        before = "def f(a):\n    return a + 1\n"
        after = "def f(a):\n    return a + 2\n"
        self.assertNotEqual(fingerprint_source(before), fingerprint_source(after))

    def test_a_changed_fold_table_entry_moves_the_fingerprint(self):
        # The concrete hazard: someone edits one line of the S2 table and does
        # not touch NORMALISER_VERSION. Every stored offset is now wrong and
        # the version string still says everything is fine.
        module = import_module("science2code.normalise")
        with open(module.__file__, encoding="utf-8") as handle:
            source = handle.read()
        edited = source.replace(r'table["\u037e"] = ";"',
                                r'table["\u037e"] = ","')
        self.assertNotEqual(edited, source)
        self.assertNotEqual(fingerprint_source(edited), NORMALISER_FINGERPRINT)

    def test_a_changed_control_flow_moves_the_fingerprint(self):
        before = "def f(a):\n    if a:\n        return 1\n    return 2\n"
        after = "def f(a):\n    if not a:\n        return 1\n    return 2\n"
        self.assertNotEqual(fingerprint_source(before), fingerprint_source(after))

    def test_fingerprint_source_refuses_source_that_does_not_parse(self):
        with self.assertRaises(SyntaxError):
            fingerprint_source("def f(:\n")

    def test_a_docstring_only_function_survives_docstring_removal(self):
        # Stripping the docstring empties the body, which would be a syntax
        # error to reconstruct. The stripper substitutes `pass` for that case.
        self.assertIsInstance(fingerprint_source('def f():\n    """Doc."""\n'),
                              str)

    def test_the_fingerprint_is_not_the_version_string(self):
        # Two independent signals. If the fingerprint were derived from the
        # version it would inherit exactly the weakness it exists to cover.
        self.assertNotEqual(NORMALISER_FINGERPRINT, NORMALISER_VERSION)
        self.assertNotIn(NORMALISER_VERSION, NORMALISER_FINGERPRINT)

    def test_the_stripper_leaves_a_real_leading_expression_alone(self):
        # Only a leading STRING constant is a docstring. A leading call is
        # code and must count towards the digest.
        before = "print(1)\nx = 2\n"
        after = "print(9)\nx = 2\n"
        self.assertNotEqual(fingerprint_source(before), fingerprint_source(after))

    def test_the_digest_ignores_source_positions(self):
        # ast.dump is called without include_attributes, so line and column
        # numbers cannot leak into the digest. Proven rather than assumed.
        tree = ast.parse("x = 1\n")
        self.assertNotIn("lineno", ast.dump(tree))


class TestNoDashesInTheSource(unittest.TestCase):
    """This project refuses an en dash or em dash anywhere in a .py file."""

    def test_neither_source_file_contains_a_dash_the_hook_refuses(self):
        module = import_module("science2code.normalise")
        paths = [module.__file__, os.path.abspath(__file__)]
        for path in paths:
            with self.subTest(path=os.path.basename(path)):
                with open(path, encoding="utf-8") as handle:
                    source = handle.read()
                self.assertNotIn("\u2013", source, "en dash in %s" % path)
                self.assertNotIn("\u2014", source, "em dash in %s" % path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
