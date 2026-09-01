"""PDF to plain text, in reading order.

The output of this module is plain UTF-8 text with no markup. No headings, no
fences, no tags, no JSON. A converted paper is meant to read exactly like the
paper reads, so that a passage quoted out of it is the passage a human sees on
the page.

The one structural element is the ASCII form feed (U+000C) that ``pdftotext``
already emits between pages. It is invisible in every viewer, it is not markup,
and it is load bearing: every locator this project hands back carries a page
number, so the page boundary has to survive into the text file. For a document
produced by :func:`extract_document`::

    parts = text.split("\x0c")
    parts[0]  # the header block
    parts[n]  # page n, one indexed

Extraction runs in reading order. The ``-layout`` flag is never passed, and
:func:`_command` refuses to build a command line containing it. Measured over a
44 paper census and 879 quote attempts, page correct locatability was 30.5
percent with ``-layout`` against 80.7 percent in reading order; on two column
papers, which is most of a scholarly corpus, ``-layout`` scored 1.6 percent
because it puts left column and right column text on the same physical line and
so splices every body sentence.

Metadata is never parsed out of the PDF. It arrives from the caller, which is
the corpus manifest, and an unknown field is omitted rather than guessed.
Guessing a title from a PDF is how a wrong citation starts.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

#: The page separator. This is what pdftotext emits, and what a locator needs.
PAGE_BREAK = "\x0c"

#: The last line of the header block. Everything before it is metadata about
#: the extraction; everything after the first form feed is extracted text.
HEADER_SENTINEL = "Extracted text follows. Pages are separated by a form feed."

#: A document below this many characters per page has no usable text layer.
#: 200 is the measured safe value: real papers sit far above it, while image
#: only scans in the census measured 105 and 1 characters per page.
MIN_CHARS_PER_PAGE = 200

_PDFTOTEXT = "pdftotext"
_FORBIDDEN_FLAGS = ("-layout", "-raw", "-fixed", "-tsv", "-htmlmeta", "-bbox",
                    "-bbox-layout")

_HEADER_ORDER = (
    "Title",
    "Authors",
    "Date",
    "Venue",
    "DOI",
    "Paper ID",
    "Source PDF",
    "Source SHA256",
    "Pages",
    "Extractor",
)

_NOTE = (
    "Note: This file is derived. The PDF named above is the provenance. "
    "The text was extracted from it and never generated."
)


class ExtractError(RuntimeError):
    """Extraction could not be completed."""


class PdftotextUnavailable(ExtractError):
    """The pdftotext binary is not installed or is not runnable."""


class NoTextLayerError(ExtractError):
    """The PDF carries no usable text layer, so no text file is written.

    This is a boundary, not a bug. A junk text file that looks like a paper is
    strictly worse than an absent one, because the absent one says "quote this
    by hand" and the junk one silently answers with nonsense.
    """

    def __init__(self, pdf: Path, chars: int, n_pages: int,
                 chars_per_page: int, floor: int) -> None:
        self.pdf = Path(pdf)
        self.chars = chars
        self.n_pages = n_pages
        self.chars_per_page = chars_per_page
        self.floor = floor
        super().__init__(
            "no text layer: {} characters per page over {} page(s), "
            "floor is {} ({})".format(chars_per_page, n_pages, floor, self.pdf)
        )


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdftotext_version() -> str:
    """The installed poppler pdftotext version, for example ``24.02.0``.

    Raises :class:`PdftotextUnavailable` if the binary cannot be run or its
    banner cannot be parsed. The version is never invented.
    """
    try:
        proc = subprocess.run(
            [_PDFTOTEXT, "-v"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError) as exc:
        raise PdftotextUnavailable(
            "cannot run '{}': {}. Install poppler-utils.".format(_PDFTOTEXT, exc)
        ) from exc
    banner = (proc.stderr or "") + (proc.stdout or "")
    match = re.search(r"pdftotext\s+version\s+(\S+)", banner)
    if not match:
        first = banner.strip().splitlines()[0] if banner.strip() else "(no output)"
        raise PdftotextUnavailable(
            "could not parse a version from 'pdftotext -v': {}".format(first)
        )
    return match.group(1)


def extractor_version() -> str:
    """The extractor identity written into every header, for example
    ``poppler/24.02.0 reading-order``."""
    return "poppler/{} reading-order".format(pdftotext_version())


_EXTRACTOR_VERSION_CACHE: list[str] = []


def __getattr__(name: str) -> str:
    # EXTRACTOR_VERSION is resolved on first use rather than at import, so that
    # importing this module never shells out and never fails on a machine
    # without poppler. PEP 562 makes it behave like a plain module constant.
    if name == "EXTRACTOR_VERSION":
        if not _EXTRACTOR_VERSION_CACHE:
            _EXTRACTOR_VERSION_CACHE.append(extractor_version())
        return _EXTRACTOR_VERSION_CACHE[0]
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def _command(pdf: Path) -> list[str]:
    """Build the pdftotext command line. Reading order, always."""
    argv = [_PDFTOTEXT, "-q", "-enc", "UTF-8", "-eol", "unix", str(pdf), "-"]
    for flag in _FORBIDDEN_FLAGS:
        if flag in argv:
            raise ExtractError(
                "refusing to run pdftotext with {}: this project extracts in "
                "reading order".format(flag)
            )
    return argv


def extract_pages(pdf: Path) -> list[str]:
    """Extract one string per page, in reading order.

    The strings are exactly what pdftotext produced, byte for byte after UTF-8
    decoding. Nothing is stripped, folded, joined or repaired here.
    """
    pdf = Path(pdf)
    if not pdf.is_file():
        raise ExtractError("not a file: {}".format(pdf))
    try:
        proc = subprocess.run(_command(pdf), capture_output=True, check=False)
    except OSError as exc:
        raise PdftotextUnavailable(
            "cannot run '{}': {}. Install poppler-utils.".format(_PDFTOTEXT, exc)
        ) from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise ExtractError(
            "pdftotext exited {} on {}{}".format(
                proc.returncode, pdf, ": " + detail if detail else ""
            )
        )
    try:
        raw = proc.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractError(
            "pdftotext produced bytes that are not UTF-8 for {}: {}".format(pdf, exc)
        ) from exc
    pages = raw.split(PAGE_BREAK)
    # pdftotext writes a form feed after every page, including the last, so the
    # split leaves one empty trailing element. Drop that and only that.
    if pages and pages[-1] == "":
        pages.pop()
    return pages


def measure(pages: list[str]) -> tuple[int, int, int]:
    """Return ``(characters, page count, characters per page)``.

    A document with no pages measures zero characters per page, which is below
    every floor, which is the right answer.
    """
    chars = sum(len(page) for page in pages)
    n_pages = len(pages)
    return chars, n_pages, (chars // n_pages if n_pages else 0)


def has_text_layer(pages: list[str], floor: int = MIN_CHARS_PER_PAGE) -> bool:
    """True when the extracted text is dense enough to be a real text layer."""
    return measure(pages)[2] >= floor


def blank_pages(pages: list[str]) -> list[int]:
    """One indexed numbers of the pages that carry no text at all.

    :func:`measure` divides the whole document's characters by its page count,
    so the floor is an average and a document can clear it while individual
    pages carry nothing. That is what a part scanned paper looks like: a born
    digital front matter and photographed plates, or one dense page against
    nine images.

    Those pages are kept rather than dropped, because dropping one would move
    every page number after it and a locator would then point at the wrong
    page. They are reported instead, so that "nothing was found on page 7" can
    be told apart from "page 7 was never extracted". Silence about them is the
    one outcome this project cannot have.
    """
    return [number for number, page in enumerate(pages, start=1) if not page.strip()]


def describe_pages(numbers: list[int], limit: int = 12) -> str:
    """Render page numbers for a human, without printing a hundred of them."""
    shown = [str(number) for number in numbers[:limit]]
    if len(numbers) > limit:
        shown.append("and {} more".format(len(numbers) - limit))
    return ", ".join(shown)


def _header_values(pdf: Path, pages: list[str], meta: dict | None) -> dict:
    meta = dict(meta or {})
    values: dict[str, str] = {}

    def put(key: str, value: object) -> None:
        # Unknown metadata is omitted, never guessed and never left as a
        # placeholder that a later reader could mistake for a fact.
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            parts = [str(item).strip() for item in value if str(item).strip()]
            if not parts:
                return
            text = "; ".join(parts)
        else:
            text = str(value).strip()
        if not text:
            return
        values[key] = " ".join(text.split())

    put("Title", meta.get("title"))
    put("Authors", meta.get("authors"))
    put("Date", meta.get("date") if meta.get("date") is not None else meta.get("year"))
    put("Venue", meta.get("venue"))
    put("DOI", meta.get("doi"))
    put("Paper ID", meta.get("paper_id"))
    # The remaining fields are measured here, not supplied, so they cannot be
    # overridden by a caller with a stale idea of the file.
    put("Source PDF", pdf.name)
    put("Source SHA256", sha256_file(pdf))
    put("Pages", len(pages))
    put("Extractor", extractor_version())
    return values


def build_header(pdf: Path, pages: list[str], meta: dict | None = None) -> str:
    """The header block: ``Key: value`` lines, the note, then the sentinel."""
    values = _header_values(Path(pdf), pages, meta)
    lines = ["{}: {}".format(key, values[key]) for key in _HEADER_ORDER if key in values]
    lines.append(_NOTE)
    lines.append(HEADER_SENTINEL)
    return "\n".join(lines) + "\n"


def _assert_pages_intact(text: str, pages: list[str], pdf: Path) -> None:
    """Every page written must be character for character what pdftotext gave.

    Text is never silently dropped. If this ever fires, the file is not written
    at all, because a short page is a wrong quote waiting to happen.
    """
    parts = text.split(PAGE_BREAK)
    if len(parts) != len(pages) + 1:
        raise ExtractError(
            "page structure damaged for {}: {} sections for {} pages".format(
                pdf, len(parts), len(pages)
            )
        )
    for index, page in enumerate(pages, start=1):
        written = parts[index]
        if len(written) != len(page):
            raise ExtractError(
                "page {} of {} would lose text: {} characters written against "
                "{} extracted".format(index, pdf, len(written), len(page))
            )
        if written != page:
            raise ExtractError(
                "page {} of {} would be altered".format(index, pdf)
            )


def build_document(pdf: Path, pages: list[str], meta: dict | None = None) -> str:
    """Assemble the header and the pages into the final plain text document."""
    pdf = Path(pdf)
    text = build_header(pdf, pages, meta) + PAGE_BREAK + PAGE_BREAK.join(pages)
    _assert_pages_intact(text, pages, pdf)
    return text


def extract_document(pdf: Path, meta: dict | None = None, *,
                     floor: int = MIN_CHARS_PER_PAGE) -> str:
    """Extract a PDF into the plain text document format.

    Raises :class:`NoTextLayerError` when the PDF is an image only scan. The
    caller is expected to record that state and write nothing.
    """
    pdf = Path(pdf)
    pages = extract_pages(pdf)
    chars, n_pages, per_page = measure(pages)
    if per_page < floor:
        raise NoTextLayerError(pdf, chars, n_pages, per_page, floor)
    return build_document(pdf, pages, meta)


def split_document(text: str) -> tuple[str, list[str]]:
    """Inverse of :func:`build_document`: ``(header, pages)``.

    Tolerates a plain pdftotext dump with no header block, which is what a user
    who converted a paper by hand will have.
    """
    parts = text.split(PAGE_BREAK)
    if parts and HEADER_SENTINEL in parts[0]:
        return parts[0], parts[1:]
    # No header. A raw dump ends with a form feed, so drop one empty tail.
    if len(parts) > 1 and parts[-1] == "":
        parts = parts[:-1]
    return "", parts


def header_fields(text: str) -> dict:
    """Parse the header block back into a ``{Key: value}`` dictionary."""
    header, _ = split_document(text)
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if line == HEADER_SENTINEL or line.startswith("Note:"):
            continue
        key, sep, value = line.partition(":")
        if sep and key in _HEADER_ORDER:
            fields[key] = value.strip()
    return fields
