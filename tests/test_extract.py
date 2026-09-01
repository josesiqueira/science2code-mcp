"""Tests for PDF extraction.

Fixtures are inline. Each test that needs a PDF builds one here, with the
standard library only, so the suite depends on no data files and never reads
anyone's paper collection. Tests that need the poppler binary skip cleanly when
it is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from science2code import extract  # noqa: E402
from science2code.extract import (  # noqa: E402
    HEADER_SENTINEL,
    PAGE_BREAK,
    ExtractError,
    NoTextLayerError,
    blank_pages,
    build_document,
    build_header,
    describe_pages,
    extract_document,
    extract_pages,
    extractor_version,
    header_fields,
    measure,
    pdftotext_version,
    split_document,
)

HAVE_PDFTOTEXT = shutil.which("pdftotext") is not None
needs_pdftotext = unittest.skipUnless(
    HAVE_PDFTOTEXT, "pdftotext (poppler-utils) is not installed")


def make_pdf(path: Path, pages: list) -> Path:
    """Write a minimal one font PDF with the given lines per page.

    Hand assembled so that the test suite needs no PDF library. Objects are
    catalog, page tree, then a page and a content stream per page, then the
    font.
    """
    def escape(text: str) -> str:
        for old, new in (("\\", "\\\\"), ("(", "\\("), (")", "\\)")):
            text = text.replace(old, new)
        return text

    n_pages = len(pages)
    font_num = 3 + 2 * n_pages
    kids = " ".join("{} 0 R".format(3 + 2 * i) for i in range(n_pages))
    bodies = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(kids, n_pages),
    ]
    for index, lines in enumerate(pages):
        page_num = 3 + 2 * index
        content_num = page_num + 1
        bodies.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 {} 0 R >> >> /Contents {} 0 R >>".format(
                font_num, content_num)
        )
        drawn = "\n".join("({}) Tj T*".format(escape(line)) for line in lines)
        stream = "BT /F1 12 Tf 72 720 Td 14 TL\n{}\nET".format(drawn)
        bodies.append("<< /Length {} >>\nstream\n{}\nendstream".format(
            len(stream), stream))
    bodies.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += "{} 0 obj\n{}\nendobj\n".format(number, body).encode("latin-1")
    xref_at = len(out)
    out += "xref\n0 {}\n".format(len(bodies) + 1).encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += "{:010d} 00000 n \n".format(offset).encode("latin-1")
    out += "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n".format(
        len(bodies) + 1, xref_at).encode("latin-1")
    path.write_bytes(bytes(out))
    return path


def body_lines(marker: str, count: int = 40) -> list:
    """Enough text that a page sits comfortably above the text layer floor."""
    return ["{} line {:02d}: the quick brown fox jumps over the lazy dog.".format(
        marker, n) for n in range(count)]


class TemporaryCorpus(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class VersionTests(unittest.TestCase):
    @needs_pdftotext
    def test_version_is_parsed_from_the_binary(self) -> None:
        version = pdftotext_version()
        self.assertRegex(version, r"^\d+\.\d")

    @needs_pdftotext
    def test_extractor_version_names_poppler_and_reading_order(self) -> None:
        self.assertEqual(
            extractor_version(), "poppler/{} reading-order".format(pdftotext_version()))

    @needs_pdftotext
    def test_module_constant_matches_the_function(self) -> None:
        self.assertEqual(extract.EXTRACTOR_VERSION, extractor_version())

    def test_unknown_module_attribute_still_raises(self) -> None:
        # A module attribute lookup by name, so the PEP 562 __getattr__ hook
        # runs. Written as a lambda because a bare `extract.NO_SUCH_CONSTANT`
        # reads as a statement with no effect.
        self.assertRaises(AttributeError, lambda: extract.NO_SUCH_CONSTANT)


class CommandTests(unittest.TestCase):
    def test_layout_is_never_passed(self) -> None:
        argv = extract._command(Path("/tmp/x.pdf"))
        self.assertNotIn("-layout", argv)
        self.assertIn("-enc", argv)
        self.assertEqual(argv[-1], "-")

    def test_forbidden_flags_are_rejected_if_ever_added(self) -> None:
        original = extract._FORBIDDEN_FLAGS
        extract._FORBIDDEN_FLAGS = ("-q",)  # -q is in the real command line
        try:
            with self.assertRaises(ExtractError):
                extract._command(Path("/tmp/x.pdf"))
        finally:
            extract._FORBIDDEN_FLAGS = original


class ExtractPagesTests(TemporaryCorpus):
    @needs_pdftotext
    def test_one_string_per_page(self) -> None:
        pdf = make_pdf(self.root / "three.pdf",
                       [body_lines("alpha"), body_lines("beta"), body_lines("gamma")])
        pages = extract_pages(pdf)
        self.assertEqual(len(pages), 3)
        self.assertIn("alpha line 00", pages[0])
        self.assertIn("beta line 00", pages[1])
        self.assertIn("gamma line 00", pages[2])
        for page in pages:
            self.assertNotIn(PAGE_BREAK, page)

    @needs_pdftotext
    def test_pages_are_exactly_what_pdftotext_produced(self) -> None:
        pdf = make_pdf(self.root / "two.pdf", [body_lines("one"), body_lines("two")])
        raw = subprocess.run(
            ["pdftotext", "-q", "-enc", "UTF-8", "-eol", "unix", str(pdf), "-"],
            capture_output=True, check=True).stdout.decode("utf-8")
        self.assertEqual(PAGE_BREAK.join(extract_pages(pdf)) + PAGE_BREAK, raw)

    @needs_pdftotext
    def test_missing_file_is_a_clear_error(self) -> None:
        with self.assertRaises(ExtractError):
            extract_pages(self.root / "absent.pdf")


class DocumentTests(TemporaryCorpus):
    @needs_pdftotext
    def test_split_gives_header_then_one_part_per_page(self) -> None:
        pdf = make_pdf(self.root / "p.pdf",
                       [body_lines("alpha"), body_lines("beta"), body_lines("gamma")])
        text = extract_document(pdf, {"title": "A Paper"})
        parts = text.split(PAGE_BREAK)
        self.assertEqual(len(parts), 4)
        self.assertIn(HEADER_SENTINEL, parts[0])
        self.assertIn("alpha line 00", parts[1])
        self.assertIn("beta line 00", parts[2])
        self.assertIn("gamma line 00", parts[3])

    @needs_pdftotext
    def test_no_page_loses_a_single_character(self) -> None:
        pdf = make_pdf(self.root / "p.pdf", [body_lines("one"), body_lines("two")])
        pages = extract_pages(pdf)
        parts = build_document(pdf, pages).split(PAGE_BREAK)
        for index, page in enumerate(pages, start=1):
            self.assertEqual(len(parts[index]), len(page))
            self.assertEqual(parts[index], page)

    @needs_pdftotext
    def test_a_page_that_would_be_damaged_is_refused(self) -> None:
        pdf = make_pdf(self.root / "p.pdf", [body_lines("one")])
        spliced = ["text with an embedded" + PAGE_BREAK + "form feed"]
        with self.assertRaises(ExtractError):
            build_document(pdf, spliced)

    @needs_pdftotext
    def test_round_trip_through_split_document(self) -> None:
        pdf = make_pdf(self.root / "p.pdf", [body_lines("one"), body_lines("two")])
        pages = extract_pages(pdf)
        header, back = split_document(build_document(pdf, pages))
        self.assertIn(HEADER_SENTINEL, header)
        self.assertEqual(back, pages)

    def test_split_document_tolerates_a_plain_pdftotext_dump(self) -> None:
        dump = "page one\x0cpage two\x0c"
        header, pages = split_document(dump)
        self.assertEqual(header, "")
        self.assertEqual(pages, ["page one", "page two"])


class HeaderTests(TemporaryCorpus):
    @needs_pdftotext
    def test_supplied_metadata_reaches_the_header(self) -> None:
        pdf = make_pdf(self.root / "p.pdf", [body_lines("one")])
        text = extract_document(pdf, {
            "title": "An Analysis of Something",
            "authors": ["A. Author", "B. Writer"],
            "year": 1994,
            "venue": "ICRE 1994",
            "doi": "10.1109/X.1994.1",
            "paper_id": "author-1994",
        })
        fields = header_fields(text)
        self.assertEqual(fields["Title"], "An Analysis of Something")
        self.assertEqual(fields["Authors"], "A. Author; B. Writer")
        self.assertEqual(fields["Date"], "1994")
        self.assertEqual(fields["Venue"], "ICRE 1994")
        self.assertEqual(fields["DOI"], "10.1109/X.1994.1")
        self.assertEqual(fields["Paper ID"], "author-1994")
        self.assertEqual(fields["Source PDF"], "p.pdf")
        self.assertEqual(fields["Pages"], "1")
        self.assertRegex(fields["Source SHA256"], r"^[0-9a-f]{64}$")

    @needs_pdftotext
    def test_unknown_metadata_is_omitted_never_guessed(self) -> None:
        pdf = make_pdf(self.root / "p.pdf", [body_lines("one")])
        text = extract_document(pdf, None)
        fields = header_fields(text)
        for absent in ("Title", "Authors", "Date", "Venue", "DOI", "Paper ID"):
            self.assertNotIn(absent, fields)
        self.assertIn("Source SHA256", fields)
        header = text.split(PAGE_BREAK)[0]
        self.assertNotIn("unknown", header.lower())
        self.assertNotIn("untitled", header.lower())

    @needs_pdftotext
    def test_header_carries_the_derivation_note(self) -> None:
        pdf = make_pdf(self.root / "p.pdf", [body_lines("one")])
        header = extract_document(pdf).split(PAGE_BREAK)[0]
        note = [line for line in header.splitlines() if line.startswith("Note:")]
        self.assertEqual(len(note), 1)
        self.assertIn("derived", note[0])
        self.assertIn("provenance", note[0])
        self.assertIn("never generated", note[0])

    @needs_pdftotext
    def test_the_header_carries_no_markup(self) -> None:
        pdf = make_pdf(self.root / "p.pdf", [body_lines("one")])
        header = extract_document(pdf, {"title": "A Paper"}).split(PAGE_BREAK)[0]
        for character in "#*`<>|[]{}":
            self.assertNotIn(character, header)
        for line in header.splitlines():
            if line == HEADER_SENTINEL:
                continue
            self.assertRegex(line, r"^[A-Za-z0-9 ]+: ")

    def test_header_can_be_built_without_running_pdftotext_on_the_body(self) -> None:
        pdf = self.root / "fake.pdf"
        pdf.write_bytes(b"%PDF-1.4 not really a pdf")
        if not HAVE_PDFTOTEXT:
            self.skipTest("pdftotext is not installed")
        header = build_header(pdf, ["a", "b"], {"title": "T"})
        self.assertIn("Pages: 2", header)
        self.assertTrue(header.rstrip("\n").endswith(HEADER_SENTINEL))


class TextLayerFloorTests(TemporaryCorpus):
    def test_measure_handles_an_empty_document(self) -> None:
        self.assertEqual(measure([]), (0, 0, 0))

    def test_measure_averages_over_pages(self) -> None:
        self.assertEqual(measure(["a" * 300, "b" * 100]), (400, 2, 200))

    @needs_pdftotext
    def test_a_scan_with_no_text_layer_is_refused(self) -> None:
        pdf = make_pdf(self.root / "scan.pdf", [["hi"], ["there"]])
        with self.assertRaises(NoTextLayerError) as caught:
            extract_document(pdf)
        error = caught.exception
        self.assertLess(error.chars_per_page, extract.MIN_CHARS_PER_PAGE)
        self.assertEqual(error.floor, extract.MIN_CHARS_PER_PAGE)
        self.assertIn("no text layer", str(error))

    @needs_pdftotext
    def test_no_junk_file_is_produced_for_a_scan(self) -> None:
        pdf = make_pdf(self.root / "scan.pdf", [["hi"]])
        with self.assertRaises(NoTextLayerError):
            extract_document(pdf)
        self.assertFalse((self.root / "scan.txt").exists())

    @needs_pdftotext
    def test_the_floor_is_adjustable_for_a_caller_who_means_it(self) -> None:
        pdf = make_pdf(self.root / "scan.pdf", [["hi"]])
        text = extract_document(pdf, floor=1)
        self.assertIn("hi", text)


class BlankPageTests(TemporaryCorpus):
    """The floor divides the whole document by its page count, so it is an
    average, and an average says nothing about any particular page."""

    def test_a_page_with_nothing_on_it_is_named(self) -> None:
        self.assertEqual(blank_pages(["full", "", "also full"]), [2])

    def test_whitespace_only_counts_as_blank(self) -> None:
        self.assertEqual(blank_pages(["a", " \n\t ", "b", ""]), [2, 4])

    def test_a_readable_document_names_no_pages(self) -> None:
        self.assertEqual(blank_pages(["a", "b", "c"]), [])

    def test_numbers_are_one_indexed_like_every_locator(self) -> None:
        self.assertEqual(blank_pages([""]), [1])

    def test_a_document_can_clear_the_floor_with_pages_that_carry_nothing(self) -> None:
        # One dense page against nine images: 312 characters per page, which
        # is comfortably over the floor, and nine pages of nothing.
        pages = ["x" * 3120] + [""] * 9
        self.assertGreaterEqual(measure(pages)[2], extract.MIN_CHARS_PER_PAGE)
        self.assertEqual(blank_pages(pages), [2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_describe_pages_stops_rather_than_printing_a_hundred(self) -> None:
        self.assertEqual(describe_pages([1, 2, 3]), "1, 2, 3")
        rendered = describe_pages(list(range(1, 31)))
        self.assertIn("and 18 more", rendered)
        self.assertLess(len(rendered), 80)

    @needs_pdftotext
    def test_a_part_scanned_pdf_keeps_its_empty_pages(self) -> None:
        # Dropping an empty page would move every page number after it, so the
        # pages stay and the emptiness is reported instead.
        pdf = make_pdf(self.root / "mixed.pdf",
                       [body_lines("alpha", 120), [], body_lines("gamma", 120)])
        pages = extract_pages(pdf)
        self.assertEqual(len(pages), 3)
        self.assertEqual(blank_pages(pages), [2])
        text = extract_document(pdf)
        self.assertEqual(len(split_document(text)[1]), 3)


class PageIntactnessTests(TemporaryCorpus):
    """Why the per page character assertion cannot fire on this path.

    ``build_document`` joins the pages with a form feed and then splits the
    result back on the same character, so the assertion is comparing derived
    values unless a page already contains a form feed. It does earn its place
    for a caller that supplies its own pages, and the guarantee that makes it
    unreachable from extraction is worth stating outright: no page that
    ``extract_pages`` returns can contain a page break, because that is the
    character it split on.
    """

    @needs_pdftotext
    def test_no_extracted_page_can_contain_a_page_break(self) -> None:
        pdf = make_pdf(self.root / "p.pdf",
                       [body_lines("one"), body_lines("two"), body_lines("three")])
        for page in extract_pages(pdf):
            self.assertNotIn(PAGE_BREAK, page)

    @needs_pdftotext
    def test_a_form_feed_in_the_metadata_cannot_shift_the_page_numbers(self) -> None:
        # A hand written manifest supplies the header fields. A form feed in a
        # title would add a section to the split and make page n read as page
        # n plus one for the whole document.
        pdf = make_pdf(self.root / "p.pdf", [body_lines("one"), body_lines("two")])
        pages = ["PAGE ONE", "PAGE TWO"]
        text = build_document(pdf, pages, {"title": "Trust\x0cand Reliance"})
        self.assertEqual(text.split(PAGE_BREAK)[1:], pages)
        self.assertEqual(header_fields(text)["Title"], "Trust and Reliance")

    @needs_pdftotext
    def test_a_newline_in_the_metadata_cannot_forge_a_header_field(self) -> None:
        pdf = make_pdf(self.root / "p.pdf", [body_lines("one")])
        text = build_document(pdf, ["PAGE ONE"],
                              {"title": "Real Title\nAuthors: Fabricated Person"})
        self.assertNotIn("Authors", header_fields(text))


if __name__ == "__main__":
    unittest.main()
