"""The science2code MCP server: two tools, both read only.

WHY THIS IS AN MCP SERVER AND NOT A CLI
---------------------------------------
On 2026-08-28 an agent made three literature errors in one session: it
attributed a paper to the wrong author, it reported a stated goal as an
achieved result, and it reversed a judgement about a paper once the paper was
actually read. All three came from reasoning over one-line summaries and search
snippets instead of reading. A written rule already forbade exactly that, and
was ignored for a whole session.

MCP tools are surfaced to the model automatically, so the model sees the tool
exists without having to remember a rule. Discoverability is the reliability
mechanism here, which makes the tool descriptions below the primary reliability
surface of this project. They are written as engineering, not as documentation.

STATELESSNESS
-------------
The current MCP revision removes protocol-level sessions, so state travels in
arguments. This module holds no cross-call state except a read cache keyed by
the content hash of a document's characters, which is a pure function of bytes
on disk and so cannot go stale or leak between callers.

THIS SERVER NEVER WRITES A FILE. There is no open() for writing, no mkdir, no
delete, anywhere in this package's runtime path, and a test asserts it.

The MCP import is confined to `build_server()` so that `envelope.py` stays
stdlib only and both tools stay callable, and testable, with no MCP client.
"""

from __future__ import annotations

import bisect
import collections.abc
import dataclasses
import hashlib
import os
import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from science2code import __version__, markers
from science2code import envelope as env
from science2code.anchor import (
    T_LOCATE_DEFAULT,
    T_LOCATE_EQUIVALENT_RANGE,
    VERIFIER_VERSION,
    Anchor,
    Tier,
    locate,
    prepare,
)
from science2code.corpus import Corpus
from science2code.envelope import Outcome
from science2code.extract import HEADER_SENTINEL, PAGE_BREAK
from science2code.normalise import NORMALISER_VERSION, normalise

__all__ = [
    "verify_quote",
    "find_passage",
    "TOOL_SPECS",
    "VERIFY_QUOTE_DESCRIPTION",
    "FIND_PASSAGE_DESCRIPTION",
    "READ_ONLY_ANNOTATIONS",
    "build_server",
    "main",
]

#: Fixed for this release. rapidfuzz is optional and its scale is not
#: interchangeable with difflib's, so the scorer is pinned rather than exposed
#: as an argument a caller could vary without noticing the threshold moved.
SCORER = "difflib"

#: A paper identifier safe to name in a response without vetting: the shape
#: a real identifier has, and nothing that could carry prose or a path.
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")

#: A string shorter than this, after normalisation, occurs in so many places
#: that reporting one of them would name a location the caller did not mean.
#: Reported as NOT_LOCATABLE with the reason stated, never as a match.
MIN_LOCATABLE_CHARS = 12
#: A verbatim quote is a sentence, a paragraph, at most a long abstract, never
#: a book. Five thousand characters is past any real quote yet still bounds the
#: work: T3 relocation scores windows with an O(needle squared) comparison, but
#: anchor._best_window caps the NUMBER of windows in inverse proportion to
#: needle-squared, so scoring cost is flat above a needle of about 1,150 and
#: the ceiling above that point does not change the CPU worst case. It is here
#: to refuse an absurd argument and bound memory; the scale-independent CPU
#: bound is the window cap. A legitimate long quote that is actually present is
#: found by the exact tiers before T3 is ever reached. A refusal here is a
#: typed outcome, not a crash.
MAX_LOCATABLE_CHARS = 5000

#: Environment variable naming the corpus root. Resolved per call, so nothing
#: is captured at import time.
CORPUS_ROOT_ENV = "SCIENCE2CODE_CORPUS"
CORPUS_ROOT_DEFAULT = "corpus"

MAX_HITS_LIMIT = 50

# Tier names rather than Tier members, so a rename in the ladder fails loudly
# at the boundary instead of silently falling through to a default.
_TIER_TO_OUTCOME: dict[str, Outcome] = {
    "T1_EXACT": Outcome.VERBATIM_EXACT,
    "T2_RELAXED": Outcome.VERBATIM_RELAXED_EXTRACTOR_DAMAGE,
    "T3_LOCATED": Outcome.PASSAGE_RELOCATED_QUOTE_DIFFERS,
    "T4_NOT_LOCATABLE": Outcome.NOT_LOCATABLE,
}

_TIER_RANK: dict[str, int] = {
    "T1_EXACT": 0,
    "T2_RELAXED": 1,
    "T3_LOCATED": 2,
    "T4_NOT_LOCATABLE": 3,
}


# ---------------------------------------------------------------------------
# tool descriptions: the reliability surface
# ---------------------------------------------------------------------------

def _paragraph(*parts: str) -> str:
    """Wrap a paragraph to 79 columns.

    The tool description is what the model actually reads, so the shared
    ceiling sentence is interpolated and then rewrapped rather than dropped
    into a hand-wrapped block, where it would leave one very long line.
    """
    return textwrap.fill("".join(parts), width=79)


_VERIFY_CEILING = _paragraph(
    "CEILING. This tool decides one question: does this string occur in this ",
    "document, and where. ",
    env.CEILING_SENTENCE,
    " A located passage is a place to start reading, not a finished citation, ",
    "and a passage containing a citation marker is an attributed claim rather ",
    "than the document's own. Every response carries interpretation_notice, a ",
    "provenance block naming the normaliser, verifier, scorer and server ",
    "version, and a not_available list naming every absence with its reason, ",
    "so nothing is silently omitted.",
)

_FIND_CEILING = _paragraph(
    "CEILING. ",
    env.CEILING_SENTENCE,
    " Locating a phrase says where it is and nothing about what the ",
    "surrounding argument was doing with it, so reading around the location is ",
    "the caller's job and cannot be delegated to this tool. Every response ",
    "carries interpretation_notice, a provenance block and a not_available ",
    "list.",
)


VERIFY_QUOTE_DESCRIPTION = """\
Check a string against the scientific papers held locally and report which of a
closed set of mechanical outcomes it meets. This is the refusal engine. It
either returns the document's own characters with a locator, or it refuses in a
way that leaves the caller nothing to copy.

WHEN TO CALL IT. Call it before writing any sentence that attributes words, a
finding, a method or a position to a paper. Call it instead of recalling what a
paper says, instead of a one-line summary in a reference list, instead of a
search-result snippet, and instead of an abstract. Those habits are the
recorded cause of the errors this server exists to prevent: a paper attributed
to the wrong author, a stated goal reported as an achieved result, and a
judgement about a paper reversed once the paper was read in full. If the only
source for a sentence is a summary or a snippet, that sentence is not grounded
yet, and this tool is how you find out before it reaches a file.

ARGUMENTS. `text` is the string to be attributed. `paper_id` names the document
to check it against; omit it to check every document held, which is also how
you find out that the same string occurs in more than one paper. `t_locate` is
the fuzzy threshold for the relocation tier and defaults to 0.65. On the
perturbed quotes it was measured on, every value from 0.55 to 0.72 gave the
same outcomes, but that is not an equivalence for arbitrary strings: a title
of a paper the corpus does not hold scored 0.610 against an unrelated window,
which is NOT_LOCATABLE at the default and a located passage at 0.55. Lowering
it names locations for strings that are not there, so leave it alone unless
you have a reason you can state.

OUTCOMES, one of a closed set of eight:
  VERBATIM_EXACT: the normalised string occurs literally in the document. No
    threshold is involved, so a paraphrase cannot reach this outcome.
  VERBATIM_RELAXED_EXTRACTOR_DAMAGE: it occurs literally once intra-word
    hyphens are removed and case is folded. That is the signature of damage the
    PDF extractor did, not damage the quote did. Still character identity.
  PASSAGE_RELOCATED_QUOTE_DIFFERS: a passage above threshold was located, and
    your string is not what the document says. The response carries
    document_text, your_text and char_diff. It deliberately carries no field
    you can lift as a quote, because the document did not say it your way.
  NOT_LOCATABLE: nothing above threshold. char_interval is explicitly null, and
    best_score and scorer are reported so you can say how close you got without
    that reading as partial success. A string that is under 12 characters once
    invisible characters are removed and runs of whitespace are collapsed comes
    back this way, because it cannot identify one location rather than many.
  SOURCE_UNKNOWN: no document with that identifier is held.
  SOURCE_NOT_HELD: the document is known but no characters are held for it, for
    example a scan with no text layer. Quote it by hand or not at all.
  CORPUS_UNAVAILABLE: the local corpus could not be read.

CITATION MARKERS. Every response that names a location carries
citation_markers. Bracketed reference numbers and strict author-year citations
are counted separately, and the count inside the located characters is kept
apart from the count in the 64 characters either side, because those are four
different facts and merging them hides which one fired. Read the surrounding
counts especially: a quote that stops one character before its marker looks
clean, and that is the case that has caught a caller out. A marker at or beside
a passage means the passage attributes its claim to another work, so the words
are that document's and the claim may not be. This server does not resolve a
marker to the work it points at.

THE SENTENCE. citation_markers.sentence carries the document's own sentence
around your span, under document_text, with its own char_interval. It is there
because the sentence is the unit a marker attaches to, and because returning
the sentence is the one primitive here with a published precision of 1.00
behind it (Sarol et al., Bioinformatics 2024, where the trivial sentence
baseline scored P 1.00 / R 0.90 / F1 0.94 and beat a fine-tuned PubMedBERT).
Read it.

MARKER STYLE, and the reading that is a trap. citation_markers.marker_style
names the citation style seen in this document's body. The label that matters
is MARKER_STYLE_UNDETECTABLE: a reference region was found, so the document
does cite other work, and yet its body holds no bracketed number and no strict
author-year marker anywhere. 2 of 47 papers measured cite with superscripts,
which pdftotext glues onto the word before them as Principles4 or
hallucination26,27, and those cannot be told apart from GPT4 or Section3
without the layout the extractor discarded. When you see that label, a marker
count of zero carries no information whatever, and reading it as "no marker, so
this is the author's own claim" is a false negative wearing the shape of a
finding. Open the paper at the offset instead. MARKER_STYLE_NOT_ASSESSED means
no region was found either, so even that much was not established.

REFERENCE REGION. Every response that names a location also carries
reference_region, holding two independent signals that are deliberately not
merged into one number, because a caller told "reference like: 0.83" cannot
tell which test fired and the two need different actions. reference_region
.status says where the offset sits relative to this document's detected
bibliography: IN_REFERENCE_REGION, OUTSIDE_REFERENCE_REGION,
IN_REFERENCE_REGION_UNCERTAIN, REGION_END_UNCERTAIN, REGION_MODEL_INAPPLICABLE
or REGION_UNKNOWN. A bibliography holds OTHER papers' titles, so a string
located inside one was written by whoever this document cites, and attributing
it to this document's author is the wrong-author error. reference_region
.local_block is a second test, run over the document's raw characters:
REFERENCE_LIKE_BLOCK_WARNING means at least 3 of {{year, bibliographic token,
initials, entry marker}} were found in the block around the hit. It is a
warning and never a verdict, because 2.4% of genuine body offsets trip it, on
numbered lists and author-email footers. A hit inside a reference region is
still returned and never dropped: labelling a location is a fact, dropping one
would be a judgement.

WHAT NO FIELD HERE WILL EVER SAY. No response names whose claim a sentence
carries. A marker beside a passage means the claim may belong to another work;
which work, and whether the claim really is that work's, is not decided here
and is not decidable at an accuracy worth printing. Trained human annotators
agree on citation accuracy at Cohen's kappa 0.18 to 0.31, and the one dedicated
attribution study reaches alpha .654 against a human ceiling of .806. A field
answering it would be a number nobody can stand behind, printed in a shape you
would act on. Read the sentence in the paper.

{ceiling}""".format(ceiling=_VERIFY_CEILING)


FIND_PASSAGE_DESCRIPTION = """\
Find where a phrase occurs in the scientific papers held locally and get back
the location: the document identifier, a character interval, the page where the
corpus exposes one, and the document's own characters at that location. It is a
locator, not an answer engine. There is no embedding, no semantic search, no
ranking model and no model call anywhere in this server, so it finds a phrase
you already have; it will not find a document that is merely about your topic,
and it does not rank what it returns.

WHEN TO CALL IT. Call it when you know roughly what was written but not where,
and you need the location before you can cite it. Call it to check whether a
phrase you are about to attribute to one paper in fact occurs in several, which
is how a passage ends up attributed to the wrong author. Call it instead of a
search-result snippet and instead of your memory of a paper. If the phrase you
hold was reconstructed rather than copied, use verify_quote instead: this tool
reports character-identity occurrences only and returns nothing for a
paraphrase, which is deliberate rather than a gap.

ARGUMENTS. `query` is the phrase to locate. `paper_ids` restricts the search to
named documents; omit it to search everything held. `max_hits` caps how many
documents are reported and defaults to 10.

OUTCOMES, drawn from the same closed set:
  OK: the search ran. `hits` holds one entry per document where the phrase
    occurs, each carrying its own outcome of VERBATIM_EXACT or
    VERBATIM_RELAXED_EXTRACTOR_DAMAGE, a char_interval and document_text. An
    empty hits list with hit_count 0 is a result and not an error: the phrase
    does not occur in what is held.
  NOT_LOCATABLE: the query is under 12 characters once invisible characters
    are removed and runs of whitespace are collapsed, so it cannot identify one
    location rather than many, and no search was run.
  SOURCE_UNKNOWN: none of the identifiers you named is held.
  CORPUS_UNAVAILABLE: the local corpus could not be read.
Documents you named that are not held, and documents held with no characters,
are listed individually in not_available rather than quietly dropped.

WHAT EACH HIT CARRIES BEYOND ITS LOCATION. Every hit carries citation_markers
and reference_region, and both change how the hit should be read.

citation_markers counts bracketed reference numbers and strict author-year
citations separately, inside the located characters and in the 64 characters
either side, because a passage carrying one attributes its claim to another
work and a quote that stops one character before its marker looks clean.
citation_markers.sentence returns the document's own sentence around the hit,
under document_text, which is the unit a marker attaches to.
citation_markers.marker_style names the citation style found in that
document's body, and MARKER_STYLE_UNDETECTABLE is the label to watch: it means
a reference region was found, so the document does cite other work, while its
body holds no bracketed and no strict author-year marker at all, because it
cites with superscripts that pdftotext glued onto the preceding word as
Principles4 or hallucination26,27. For such a document a marker count of zero
carries no information, and reading it as an uncited claim of the author's own
is a false negative wearing the shape of a finding.

reference_region says where the hit sits relative to that document's
bibliography, as one of IN_REFERENCE_REGION, OUTSIDE_REFERENCE_REGION,
IN_REFERENCE_REGION_UNCERTAIN, REGION_END_UNCERTAIN, REGION_MODEL_INAPPLICABLE
or REGION_UNKNOWN. This matters here more than anywhere else in the server: a
bibliography holds OTHER papers' titles, so searching a title finds it in every
paper that cites it, and every one of those hits is a location where somebody
else's words appear. reference_region.local_block is a second and independent
test over the raw characters around the hit, reported as
REFERENCE_LIKE_BLOCK_WARNING or NO_REFERENCE_LIKE_BLOCK; it is a warning and
never a verdict, since 2.4% of genuine body offsets trip it. A hit inside a
reference region is still listed and never dropped, because labelling it is a
fact and dropping it would be a judgement.

No field on a hit says whose claim a sentence carries, and none ever will:
human annotators agree on that question at Cohen's kappa 0.18 to 0.31, which is
below what is worth printing. Read the sentence in the paper.

{ceiling}""".format(ceiling=_FIND_CEILING)


#: Both tools read. Neither writes, neither deletes, and neither reaches the
#: network. Stated to the client so a host can reason about them without
#: reading this file.
READ_ONLY_ANNOTATIONS: dict[str, Any] = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


# ---------------------------------------------------------------------------
# indirections, so the tools are testable without a corpus on disk
# ---------------------------------------------------------------------------


def _load_corpus(root: Path) -> Any:
    """Load the corpus. Overridden in tests with an inline fixture."""
    return Corpus.load(root)


def _prepare(raw: str) -> Any:
    """Normalise a document once. Overridden in tests."""
    return prepare(raw)


def _locate(haystack: Any, quote: str, t_locate: float) -> Anchor:
    """Walk the ladder. Overridden in tests to pin a tier deterministically."""
    return locate(haystack, quote, t_locate=t_locate, scorer=SCORER)


# The one piece of cross-call state, and it is keyed by the content hash of the
# characters it was built from, so it is a pure function of bytes on disk. Two
# callers with the same document share an entry; a changed document gets a new
# key rather than a stale answer.
_PREPARED_CACHE: dict[str, Any] = {}
_PREPARED_CACHE_MAX = 64


def _prepared(raw: str) -> Any:
    key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    hit = _PREPARED_CACHE.get(key)
    if hit is None:
        hit = _prepare(raw)
        if len(_PREPARED_CACHE) >= _PREPARED_CACHE_MAX:
            _PREPARED_CACHE.clear()
        _PREPARED_CACHE[key] = hit
    return hit


def _provenance() -> dict[str, str]:
    return env.provenance(NORMALISER_VERSION, VERIFIER_VERSION, SCORER)


def _corpus_root() -> Path:
    return Path(os.environ.get(CORPUS_ROOT_ENV, CORPUS_ROOT_DEFAULT)).expanduser()


# ---------------------------------------------------------------------------
# corpus access, all of it defensive
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _OpenCorpus:
    corpus: Any
    absences: tuple[dict[str, str], ...]
    stale_ids: frozenset[str] = frozenset()


def _open_corpus() -> tuple[_OpenCorpus | None, str, str | None]:
    # Resolving the root is inside the guard, not before it. It reads an
    # environment variable and calls expanduser(), and expanduser() raises:
    # SCIENCE2CODE_CORPUS=~nosuchuser/papers raises RuntimeError("Could not
    # determine home directory."). Outside the guard that escaped the tool
    # altogether and reached the client as a protocol-level error carrying a
    # bare string, so a plain misconfiguration produced an answer that was not
    # in the closed vocabulary at all. It is CORPUS_UNAVAILABLE, which is the
    # outcome that exists for exactly this.
    root: Path | None = None
    try:
        root = _corpus_root()
        corpus = _load_corpus(root)
    except Exception as exc:  # a missing or malformed corpus is a boundary
        if root is None:
            return (
                None,
                "the corpus root could not be resolved from the environment",
                "%s: %s" % (CORPUS_ROOT_ENV, exc),
            )
        return None, "the corpus at the configured root could not be read", "%s: %s" % (root, exc)
    if corpus is None:
        return None, "no corpus was returned for the configured root", str(root)
    absences: list[dict[str, str]] = []
    try:
        # The corpus reports staleness as a list of problems, one per hash that
        # no longer matches the file on disk. The count travels, the messages do
        # not: they embed filenames, which are neither this server's words nor
        # the document's, and an envelope carries no unvetted prose.
        problems = corpus.is_stale()
        count = len(problems) if isinstance(problems, (list, tuple)) else int(bool(problems))
        stale_ids = _stale_id_set(corpus)
        if count:
            detail = None
            if stale_ids:
                # Identifiers only, in `detail`, which is exempt from the prose
                # rules because it is not this server's own words. Naming them
                # lets a caller tell whether the document it just matched is one
                # of the stale ones, which the bare count cannot.
                detail = "affected paper_ids: %s" % ", ".join(sorted(stale_ids))
            absences.append(
                env.absence(
                    "corpus_freshness",
                    "the manifest and the files on disk disagree in %d %s, so a "
                    "document may have changed since it was extracted. Re-index "
                    "before relying on an offset."
                    % (count, "place" if count == 1 else "places"),
                    detail=detail,
                )
            )
    except Exception as exc:
        stale_ids = frozenset()
        absences.append(
            env.absence(
                "corpus_freshness",
                "staleness could not be determined",
                detail=str(exc),
            )
        )
    return _OpenCorpus(corpus, tuple(absences), frozenset(stale_ids)), "", None


def _stale_id_set(corpus: Any) -> frozenset[str]:
    """The stale paper ids as a set of vetted identifiers, never raising."""
    try:
        ids = corpus.stale_paper_ids()
    except Exception:
        return frozenset()
    return frozenset(str(pid) for pid in ids if _SAFE_ID.match(str(pid)))


def _paper_ids(corpus: Any) -> list[str]:
    """Every held identifier, in a deterministic order."""
    try:
        papers = list(corpus.papers())
    except Exception:
        return []
    ids = []
    for paper in papers:
        pid = getattr(paper, "paper_id", None)
        if pid is None:
            pid = getattr(paper, "id", paper)
        if pid is not None:
            ids.append(str(pid))
    return sorted(set(ids))


def _is_known(corpus: Any, paper_id: str) -> bool:
    try:
        if corpus.get(paper_id) is not None:
            return True
    except Exception:
        pass
    return paper_id in _paper_ids(corpus)


def _document(corpus: Any, paper_id: str) -> tuple[str | None, str, str | None]:
    """Return (raw characters, reason it is absent, foreign detail).

    The reason is this server's own prose and never names the document or
    quotes an operating-system message, because a reason is checked against
    rules 2 and 4 and a filename is not this server's words. Anything foreign
    travels in the third slot and reaches the envelope under `detail`, which
    is exempt. See `science2code.envelope.FOREIGN_TEXT_FIELDS`.
    """
    try:
        raw = corpus.text(paper_id)
    except Exception as exc:
        return None, "the characters for this document could not be read", str(exc)
    if not isinstance(raw, str) or not raw.strip():
        return None, (
            "no characters are held for this document. A document with no text "
            "layer is a boundary, not a failure: quote it by hand or not at all."
        ), None
    return raw, "", None


def _page_of(corpus: Any, paper_id: str, span: str) -> tuple[int | None, str, str | None]:
    """Best effort page number for a located span, or the reason there is none.

    Page locality is worth having and is never worth guessing, so anything the
    corpus does not expose in a shape this can read comes back as an absence
    with its reason rather than as a number.
    """
    if not span:
        return None, "the located span is empty, so no page can be named", None
    try:
        pages = corpus.pages(paper_id)
    except Exception as exc:
        return None, "page offsets are not available for this document", str(exc)
    if not isinstance(pages, collections.abc.Sequence) or isinstance(pages, str) or not pages:
        return None, "the corpus does not expose page text for this document", None
    for i, page in enumerate(pages, start=1):
        body = page if isinstance(page, str) else getattr(page, "text", None)
        if isinstance(body, str) and span in body:
            return i, "", None
    return None, (
        "the located span was not found whole on any single page of this "
        "document, which happens when a passage runs across a page break"
    ), None


def _header_end(raw: str) -> int:
    """Where the extraction header stops and the paper's own words start.

    This matters because the header carries a title and an author list written
    by this toolchain. A string located in it is not something the paper said,
    and reporting one as though it were would be the wrong-author error this
    server exists to prevent, committed by the server itself.

    The boundary test is the header sentinel, which is the same test
    `science2code.extract.split_document` applies, so the server and the
    document format cannot disagree about where the header ends. An earlier
    version counted form feeds instead, on the theory that a document with a
    header holds one more of them than a document without. That theory is
    false: pdftotext writes a form feed after the last page too, so a plain
    dump with no header counted equal and the server suppressed the whole of
    its first page as though it were metadata. A false refusal is a cheaper
    failure than a false quote, but it is still a failure, and the sentinel
    decides the question outright rather than inferring it.
    """
    head, sep, _rest = raw.partition(PAGE_BREAK)
    if not sep or HEADER_SENTINEL not in head:
        return 0
    return len(head) + len(PAGE_BREAK)


def _body_floor_norm(prepared: Any) -> int:
    """The normalised offset where the body begins, past any toolchain header.

    Maps the raw header boundary from `_header_end` into normalised
    coordinates, so the citation-marker block cannot expand a sentence or a
    marker context back into the header. Returns 0 for a plain document with no
    header. The index map is non-decreasing, so the first normalised offset
    whose source character is at or past the header end is a bisect away.
    """
    raw = getattr(prepared, "raw", None)
    index_map = getattr(prepared, "norm_index_map", None)
    if not isinstance(raw, str) or index_map is None:
        return 0
    header_end = _header_end(raw)
    if header_end <= 0:
        return 0
    return bisect.bisect_left(index_map, header_end)


def _raw_span(prepared: Any, offset: int, length: int) -> str | None:
    """Map a span in the normalised document back to the document's own bytes.

    Returns None when the anchor handed back an interval this cannot resolve,
    which is treated as a refusal rather than reported without its characters.
    An outcome that asserts character identity but cannot show the characters
    would be the exact blur this server exists to prevent.
    """
    if offset is None or length is None or length <= 0 or offset < 0:
        return None
    index_map = getattr(prepared, "norm_index_map", None)
    raw = getattr(prepared, "raw", None)
    if index_map is None or raw is None or offset >= len(index_map):
        return None
    end_norm = offset + length
    start_raw = index_map[offset]
    if end_norm < len(index_map):
        end_raw = index_map[end_norm]
    else:
        end_raw = len(raw)
    if end_raw <= start_raw:
        return None
    return raw[start_raw:end_raw]


def _raw_offset(prepared: Any, offset: int | None) -> int | None:
    """Map a normalised offset back to an offset in the document's own bytes.

    The reference-region model and the local-block test both read the raw
    characters, and they have to: the normaliser collapses every run of
    whitespace to one space, so the blank lines that delimit a reference entry
    do not survive into the rendering the offsets index. Handing either of them
    a normalised offset would point them at a different string.
    """
    if offset is None or offset < 0:
        return None
    index_map = getattr(prepared, "norm_index_map", None)
    raw = getattr(prepared, "raw", None)
    if index_map is None or not isinstance(raw, str):
        return None
    if offset >= len(index_map):
        return len(raw)
    return int(index_map[offset])


_REGION_ABSENT_REASON = (
    "the reference-region model is not installed in this build, so every "
    "location is reported as %s and no location can be told apart from one "
    "inside a bibliography. The local-block test in the same object still ran "
    "and answers independently."
    % markers.REGION_UNKNOWN
)

_UNDETECTABLE_REASON = (
    "this document cites in a style this extractor cannot see, so the marker "
    "counts beside this location carry no information and a count of zero must "
    "not be read as one. A reference region was found, so the document does "
    "cite something, and yet its body holds no bracketed number and no strict "
    "author-year marker anywhere: 2 of 47 papers measured cite with "
    "superscripts, which pdftotext glues to the word before them as "
    "Principles4 or hallucination26,27, and those are not separable from GPT4 "
    "or Section3 without the layout the extractor discarded. Open the document "
    "at this offset and look at the sentence."
)


def _attribution_blocks(
    prepared: Any, offset: int | None, length: int | None, field_suffix: str = ""
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    """The two blocks that ride on every located passage, and their absences.

    THE TWO ERRORS THIS ANSWERS, which are different errors and are reported
    in different objects because merging them would hide which one fired.

    IN BODY. A paper writes "eating oranges is good [12]". The whole sentence
    is in that paper, character for character, so it reaches a
    character-identity outcome and every field of the response is true, while
    the claim belongs to reference 12. Measured over this corpus, 13.9% of body
    sentences carry a marker inside the located span and 23.6% carry one within
    64 characters. The worst case is invisible: a quote that stops one
    character before the marker looks clean. So the counts inside the span and
    beside it are reported separately, and the sentence the marker attaches to
    is returned whole.

    IN THE BIBLIOGRAPHY. A paper's reference list holds OTHER papers' titles.
    Search a title, find it in the citing paper, and a caller concludes the
    citing author wrote it. Two independent tests report on that: region
    membership from the region model, and the local-block test. Neither is
    allowed to suppress the hit. Labelling a location is a fact; dropping it
    would be a judgement, and this server does not make judgements.

    Neither block ever says whose claim a sentence carries. That question sits
    above the decidability ceiling: human annotators agree on it at Cohen's
    kappa 0.18 to 0.31. `envelope.validate()` enforces the omission as rule 5.
    """
    norm = getattr(prepared, "norm", None)
    raw = getattr(prepared, "raw", None)
    if not isinstance(norm, str):
        norm = ""
    if not isinstance(raw, str):
        raw = ""

    raw_offset = _raw_offset(prepared, offset)
    status, region_start, region_version = markers.region_status(raw, raw_offset)

    # The style label is a fact about the DOCUMENT, read off the body, which is
    # everything before the reference region. Read off the raw characters,
    # because that is the string the region model bounded.
    body = raw[:region_start] if isinstance(region_start, int) else raw
    style = markers.marker_style(body, region_start is not None)

    citation_block = markers.citation_marker_block(
        norm, offset, length, style=style, body_floor=_body_floor_norm(prepared)
    )
    region_block = markers.reference_region_block(
        status,
        region_version=region_version,
        local_block=markers.local_block_signals(raw, raw_offset),
    )

    absences: list[dict[str, str]] = []
    if markers.region_available() == "REGION_MODEL_ABSENT":
        absences.append(
            env.absence("reference_region%s" % field_suffix, _REGION_ABSENT_REASON)
        )
    if style == markers.MARKER_STYLE_UNDETECTABLE:
        absences.append(
            env.absence("citation_markers%s" % field_suffix, _UNDETECTABLE_REASON)
        )
    return citation_block, region_block, absences


@dataclasses.dataclass(frozen=True)
class _Located:
    paper_id: str
    anchor: Anchor
    prepared: Any


def _search(
    corpus: Any, paper_ids: Sequence[str], quote: str, t_locate: float
) -> tuple[list[_Located], list[dict[str, str]], int]:
    """Locate a quote in each document. Returns (locations, absences, readable).

    `readable` counts documents whose characters were held and searched,
    whether or not anything was found in them. It separates "nothing matched"
    from "there was nothing to match against", which are different answers and
    need different actions from the human.
    """
    found: list[_Located] = []
    absences: list[dict[str, str]] = []
    readable = 0
    for pid in paper_ids:
        raw, reason, detail = _document(corpus, pid)
        if raw is None:
            absences.append(env.absence("document:%s" % pid, reason, detail=detail))
            continue
        readable += 1
        prepared = _prepared(raw)
        try:
            anchor = _locate(prepared, quote, t_locate)
        except Exception as exc:
            absences.append(
                env.absence(
                    "document:%s" % pid, "the lookup could not run", detail=str(exc)
                )
            )
            continue
        if _lands_in_header(raw, prepared, anchor):
            absences.append(
                env.absence(
                    "document:%s" % pid,
                    "the only location found in this document falls inside the "
                    "extraction header, which holds metadata written by this "
                    "toolchain rather than the document's own words, so it is "
                    "not reported as a location in the document",
                )
            )
            continue
        found.append(_Located(pid, anchor, prepared))
    return found, absences, readable


def _lands_in_header(raw: str, prepared: Any, anchor: Anchor) -> bool:
    offset = getattr(anchor, "offset_norm", None)
    if offset is None:
        return False
    index_map = getattr(prepared, "norm_index_map", None)
    if index_map is None or offset >= len(index_map):
        return False
    return index_map[offset] < _header_end(raw)


def _tier_name(anchor: Anchor) -> str:
    tier = getattr(anchor, "tier", None)
    return getattr(tier, "name", str(tier))


def _outcome_for(anchor: Anchor) -> Outcome:
    name = _tier_name(anchor)
    outcome = _TIER_TO_OUTCOME.get(name)
    if outcome is None:
        # A tier this server does not map must never fall through to a default,
        # because every available default is one of the wrong merges.
        raise env.EnvelopeViolation(
            "tier %r is not mapped to an outcome. Known tiers: %s"
            % (name, sorted(t.name for t in Tier))
        )
    return outcome


def _rank(item: _Located) -> tuple[int, float, str]:
    score = getattr(item.anchor, "score", None)
    return (_TIER_RANK.get(_tier_name(item.anchor), 99), -(score or 0.0), item.paper_id)


def _clamp_threshold(t_locate: Any) -> tuple[float, str]:
    try:
        value = float(t_locate)
    except (TypeError, ValueError):
        return T_LOCATE_DEFAULT, (
            "t_locate was not a number, so the default %.2f was used" % T_LOCATE_DEFAULT
        )
    if not 0.0 <= value <= 1.0:
        return T_LOCATE_DEFAULT, (
            "t_locate %r is outside 0.0 to 1.0, so the default %.2f was used"
            % (t_locate, T_LOCATE_DEFAULT)
        )
    low, high = T_LOCATE_EQUIVALENT_RANGE
    if not low <= value <= high:
        # Accepted, because the relocation tier never asserts character
        # identity and so cannot turn a wrong string into a located quote
        # whatever this is set to. Said out loud, because outside the measured
        # band the tier's behaviour has no evidence behind it: at 0.0 every
        # string relocates somewhere, and the window it relocates to means
        # nothing.
        return value, (
            "t_locate %.2f is outside the range %.2f to %.2f over which the "
            "relocation tier was measured, so the passage this call relocates "
            "to has no measured basis. The two character-identity outcomes are "
            "unaffected: they use no threshold at all."
            % (value, low, high)
        )
    return value, ""


def _too_short(quote: str) -> bool:
    """Is this string too short to name one location rather than many?

    Measured on the NORMALISED string, not on the raw one. The normaliser
    deletes zero-width characters and soft hyphens outright and collapses runs
    of whitespace, so a raw length says nothing about how much text is actually
    being matched. Measuring the raw length let a caller defeat this floor
    completely: the word "not" followed by twenty ZERO WIDTH SPACE characters
    is 23 raw characters and 3 normalised ones, and it came back as a
    character-identity outcome with a character interval naming ONE occurrence
    of the word "not" in one paper. That is the wrong-attribution failure this
    server exists to prevent, produced by the server itself, and it was
    reachable from any caller.
    """
    return len(normalise(quote)[0].strip()) < MIN_LOCATABLE_CHARS


_SHORT_REASON = (
    "the string is shorter than %d characters once invisible characters are "
    "removed and runs of whitespace are collapsed, so it occurs in too many "
    "places to name one location. Give more of the sentence."
    % MIN_LOCATABLE_CHARS
)


_STALE_DOC_REASON = (
    "the document this location comes from has changed on disk since it was "
    "indexed, so the characters returned here are from the older extraction "
    "and may no longer match the PDF. Re-index before relying on this."
)


def _too_long(quote: str) -> bool:
    """Is this string longer than any real quote, and so a resource hazard?

    Measured on the RAW length, because the cost this bounds is paid before
    normalisation: a multi-megabyte argument is seconds of fuzzy scoring on a
    single stateless call. A real verbatim quote never approaches the ceiling.
    """
    return len(quote) > MAX_LOCATABLE_CHARS


_LONG_REASON = (
    "the string is longer than %d characters, which is past any quote a paper "
    "would carry and into the length where relocating one argument costs the "
    "server real time. Give the sentence or paragraph you want checked, not "
    "the whole document." % MAX_LOCATABLE_CHARS
)


# ---------------------------------------------------------------------------
# tool one: verify_quote
# ---------------------------------------------------------------------------


def verify_quote(
    text: str,
    paper_id: str | None = None,
    t_locate: float = T_LOCATE_DEFAULT,
) -> dict[str, Any]:
    """Run the four-tier ladder for one string. See VERIFY_QUOTE_DESCRIPTION."""
    prov = _provenance()
    your_text = text if isinstance(text, str) else str(text)
    threshold, threshold_note = _clamp_threshold(t_locate)
    absences: list[dict[str, str]] = []
    if threshold_note:
        absences.append(env.absence("t_locate", threshold_note))

    opened, error, error_detail = _open_corpus()
    if opened is None:
        if error_detail is not None:
            absences.append(env.absence("corpus", error, detail=error_detail))
        return env.corpus_unavailable(
            reason=error, provenance_block=prov, your_text=your_text, not_available=absences
        )
    absences.extend(opened.absences)
    corpus = opened.corpus

    if _too_long(your_text):
        absences.append(env.absence("char_interval", _LONG_REASON))
        return env.not_locatable(
            your_text=your_text[:MAX_LOCATABLE_CHARS],
            best_score=None,
            scorer=SCORER,
            provenance_block=prov,
            paper_id=paper_id,
            not_available=absences,
        )

    if _too_short(your_text):
        absences.append(env.absence("char_interval", _SHORT_REASON))
        return env.not_locatable(
            your_text=your_text,
            best_score=None,
            scorer=SCORER,
            provenance_block=prov,
            paper_id=paper_id,
            not_available=absences,
        )

    if paper_id is not None:
        if not _is_known(corpus, str(paper_id)):
            return env.source_unknown(
                paper_id=paper_id,
                reason=(
                    "no document with the identifier named in paper_id is held "
                    "locally"
                ),
                provenance_block=prov,
                your_text=your_text,
                not_available=absences,
            )
        targets = [str(paper_id)]
    else:
        targets = _paper_ids(corpus)
        if not targets:
            return env.source_not_held(
                paper_id=None,
                reason="the corpus holds no documents to search",
                provenance_block=prov,
                your_text=your_text,
                not_available=absences,
            )

    found, search_absences, readable = _search(corpus, targets, your_text, threshold)
    absences.extend(search_absences)

    if not found and readable == 0:
        return env.source_not_held(
            paper_id=paper_id,
            reason=(
                "no characters are held for any document that was searched, so "
                "the ladder could not be run"
            ),
            provenance_block=prov,
            your_text=your_text,
            not_available=absences,
        )

    if not found:
        # Documents were read and none yielded a location this server will
        # report. That is not locatable, not a missing source.
        absences.append(
            env.absence(
                "char_interval",
                "%d documents were searched and none yielded a location that "
                "could be reported" % readable,
            )
        )
        return env.not_locatable(
            your_text=your_text,
            best_score=None,
            scorer=SCORER,
            provenance_block=prov,
            paper_id=paper_id,
            not_available=absences,
        )

    found.sort(key=_rank)
    best = found[0]
    outcome = _outcome_for(best.anchor)
    anchor = best.anchor
    score = getattr(anchor, "score", None)

    if outcome is Outcome.NOT_LOCATABLE:
        absences.append(
            env.absence(
                "char_interval",
                "no window in any searched document reached the threshold of "
                "%.2f, so no location is reported" % threshold,
            )
        )
        return env.not_locatable(
            your_text=your_text,
            best_score=score,
            scorer=getattr(anchor, "scorer", SCORER),
            provenance_block=prov,
            paper_id=paper_id,
            not_available=absences,
        )

    offset = getattr(anchor, "offset_norm", None)
    length = getattr(anchor, "length_norm", None)
    span = _raw_span(best.prepared, offset, length)
    if span is None:
        absences.append(
            env.absence(
                "char_interval",
                "the lookup reached a locating tier in one document but "
                "returned an interval this server could not resolve back to "
                "the document's own characters, so it is reported as not "
                "located rather than without them",
                detail="tier %s" % _tier_name(anchor),
            )
        )
        return env.not_locatable(
            your_text=your_text,
            best_score=score,
            scorer=getattr(anchor, "scorer", SCORER),
            provenance_block=prov,
            paper_id=best.paper_id,
            not_available=absences,
        )

    if best.paper_id in opened.stale_ids:
        # The document this match came from is itself one of the stale ones.
        # The corpus-wide freshness note above cannot say that; this one names
        # the served document, so a caller keying on the outcome is not left
        # matching against text that no longer corresponds to the on-disk PDF.
        absences.append(
            env.absence(
                "document_freshness",
                _STALE_DOC_REASON,
                detail=("paper_id: %s" % best.paper_id
                        if _SAFE_ID.match(str(best.paper_id)) else None),
            )
        )

    page, page_reason, page_detail = _page_of(corpus, best.paper_id, span)
    if page is None:
        absences.append(env.absence("page", page_reason, detail=page_detail))

    marker_block, region_block, block_absences = _attribution_blocks(
        best.prepared, offset, length
    )
    absences.extend(block_absences)

    if outcome is Outcome.PASSAGE_RELOCATED_QUOTE_DIFFERS:
        diff = getattr(anchor, "diff", None)
        if diff is None:
            absences.append(
                env.absence("char_diff", "the lookup returned no character diff for this window")
            )
        response = env.relocated(
            paper_id=best.paper_id,
            document_text=span,
            your_text=your_text,
            char_diff=diff,
            start_pos=offset,
            end_pos=offset + length,
            best_score=score,
            scorer=getattr(anchor, "scorer", SCORER),
            provenance_block=prov,
            page=page,
            citation_marker_block=marker_block,
            reference_region_block=region_block,
            not_available=absences,
        )
        return env.validate(_with_other_documents(response, found, paper_id))

    response = env.verbatim(
        outcome,
        paper_id=best.paper_id,
        document_text=span,
        start_pos=offset,
        end_pos=offset + length,
        score=score,
        scorer=getattr(anchor, "scorer", SCORER),
        provenance_block=prov,
        page=page,
        citation_marker_block=marker_block,
        reference_region_block=region_block,
        not_available=absences,
    )
    return env.validate(_with_other_documents(response, found, paper_id))


def _with_other_documents(
    response: dict[str, Any], found: Sequence[_Located], paper_id: str | None
) -> dict[str, Any]:
    """Name every other document the string reached the same tier in.

    One of the three recorded errors was a passage attributed to the wrong
    author. A string that occurs in several papers is exactly that risk, so it
    is reported rather than resolved by picking one.
    """
    if paper_id is not None or len(found) < 2:
        return response
    top = _tier_name(found[0].anchor)
    others = sorted(f.paper_id for f in found[1:] if _tier_name(f.anchor) == top)
    if others:
        response["also_occurs_in"] = others
        response["also_occurs_in_note"] = (
            "the same string reached the same tier in these documents too, so "
            "the identifier above is one location and not the only one"
        )
    return response


# ---------------------------------------------------------------------------
# tool two: find_passage
# ---------------------------------------------------------------------------


def find_passage(
    query: str,
    paper_ids: list[str] | None = None,
    max_hits: int = 10,
) -> dict[str, Any]:
    """Locate a phrase across the corpus. See FIND_PASSAGE_DESCRIPTION."""
    prov = _provenance()
    text = query if isinstance(query, str) else str(query)
    absences: list[dict[str, str]] = []

    try:
        cap = int(max_hits)
    except (TypeError, ValueError):
        cap = 10
        absences.append(env.absence("max_hits", "max_hits was not a whole number, so 10 was used"))
    if cap < 1 or cap > MAX_HITS_LIMIT:
        absences.append(
            env.absence(
                "max_hits",
                "max_hits %r was clamped into 1 to %d" % (max_hits, MAX_HITS_LIMIT),
            )
        )
        cap = min(max(cap, 1), MAX_HITS_LIMIT)

    opened, error, error_detail = _open_corpus()
    if opened is None:
        if error_detail is not None:
            absences.append(env.absence("corpus", error, detail=error_detail))
        return env.corpus_unavailable(
            reason=error, provenance_block=prov, your_text=text, not_available=absences
        )
    absences.extend(opened.absences)
    corpus = opened.corpus

    if _too_long(text):
        absences.append(env.absence("hits", _LONG_REASON))
        return env.not_locatable(
            your_text=text[:MAX_LOCATABLE_CHARS],
            best_score=None,
            scorer=SCORER,
            provenance_block=prov,
            paper_id=None,
            not_available=absences,
        )

    if _too_short(text):
        absences.append(env.absence("hits", _SHORT_REASON))
        return env.not_locatable(
            your_text=text,
            best_score=None,
            scorer=SCORER,
            provenance_block=prov,
            paper_id=None,
            not_available=absences,
        )

    held = _paper_ids(corpus)
    if paper_ids:
        requested = [str(p) for p in paper_ids]
        targets = [p for p in requested if p in held or _is_known(corpus, p)]
        for missing in sorted(set(requested) - set(targets)):
            absences.append(
                env.absence(
                    "document:%s" % missing,
                    "no document with this identifier is held locally",
                )
            )
        if not targets:
            return env.source_unknown(
                paper_id=requested,
                reason="none of the identifiers named is held locally",
                provenance_block=prov,
                your_text=text,
                not_available=absences,
            )
    else:
        targets = held
        if not targets:
            return env.source_unknown(
                paper_id=None,
                reason="the corpus holds no documents to search",
                provenance_block=prov,
                your_text=text,
                not_available=absences,
            )

    found, search_absences, _readable = _search(corpus, sorted(targets), text, T_LOCATE_DEFAULT)
    absences.extend(search_absences)

    hits: list[dict[str, Any]] = []
    for item in found:
        outcome = _outcome_for(item.anchor)
        # Character identity only. Returning fuzzy windows here would make this
        # a ranking engine over the corpus, which is exactly what it must not
        # be: a relocated passage is a judgement call and belongs in
        # verify_quote, where the caller sees the diff.
        if outcome not in env.VERBATIM_OUTCOMES:
            continue
        offset = getattr(item.anchor, "offset_norm", None)
        length = getattr(item.anchor, "length_norm", None)
        span = _raw_span(item.prepared, offset, length)
        if span is None:
            absences.append(
                env.absence(
                    "document:%s" % item.paper_id,
                    "an interval was returned that could not be resolved back "
                    "to the document's own characters, so it is not listed",
                )
            )
            continue
        page, page_reason, page_detail = _page_of(corpus, item.paper_id, span)
        marker_block, region_block, block_absences = _attribution_blocks(
            item.prepared, offset, length, field_suffix=":%s" % item.paper_id
        )
        absences.extend(block_absences)
        hit: dict[str, Any] = {
            "outcome": outcome.value,
            "paper_id": item.paper_id,
            "document_text": span,
            "char_interval": env.char_interval(offset, offset + length),
            "score": getattr(item.anchor, "score", None),
            "scorer": getattr(item.anchor, "scorer", SCORER),
            "page": page,
            "citation_markers": marker_block,
            "reference_region": region_block,
        }
        if page is None:
            absences.append(
                env.absence("page:%s" % item.paper_id, page_reason, detail=page_detail)
            )
        hits.append(hit)

    # Ordered by identifier, not by score. There is no relevance model here and
    # an ordering that looked like one would invite it to be read as one.
    hits.sort(key=lambda h: h["paper_id"])
    if len(hits) > cap:
        absences.append(
            env.absence(
                "hits",
                "%d further documents hold this phrase and were not listed "
                "because max_hits is %d" % (len(hits) - cap, cap),
            )
        )
        hits = hits[:cap]

    if not hits:
        absences.append(
            env.absence(
                "hits",
                "the phrase does not occur, as characters, in any document that "
                "was searched. That is a result and not a failure. Use "
                "verify_quote if the phrase was reconstructed rather than copied.",
            )
        )

    return env.ok(
        query=text,
        hits=hits,
        searched=sorted(targets),
        provenance_block=prov,
        not_available=absences,
    )


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., dict[str, Any]]
    annotations: Mapping[str, Any]


#: Registration order is fixed here rather than derived from a dict or a scan,
#: so two runs of the server present the same tool list in the same order.
TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("verify_quote", VERIFY_QUOTE_DESCRIPTION, verify_quote, READ_ONLY_ANNOTATIONS),
    ToolSpec("find_passage", FIND_PASSAGE_DESCRIPTION, find_passage, READ_ONLY_ANNOTATIONS),
)


def build_server() -> Any:
    """Build the MCP server. The MCP import lives here and nowhere else."""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "fastmcp is not installed, so the MCP server cannot start. The two "
            "tools remain importable and callable as plain functions from "
            "science2code.server."
        ) from exc

    # version: without it the MCP handshake reports fastmcp's own version as
    # this server's, so a client that asked "which science2code is this?" was
    # told "3.4.7". Every response already carries the real one in
    # provenance.server; the handshake now agrees with it.
    #
    # mask_error_details: the two tools are written so that every path returns
    # an envelope, but an unhandled exception would otherwise have its own
    # message forwarded to the client verbatim, and an OSError's message
    # carries an absolute local path. Measured on the wire: raising
    # OSError("/home/you/secret") inside a tool put that path in the client's
    # content block. Masked, the client is told the call failed and the detail
    # stays in the server's stderr, where the operator can see it and a
    # transcript cannot. Argument validation errors are unaffected: fastmcp
    # raises those as tool errors, which are never masked, so a caller that
    # omits a required argument is still told which one.
    mcp = FastMCP(
        name="science2code",
        version=__version__,
        mask_error_details=True,
    )
    for spec in TOOL_SPECS:
        mcp.tool(
            name=spec.name,
            description=spec.description,
            annotations=dict(spec.annotations),
        )(spec.fn)
    return mcp


def main() -> None:  # pragma: no cover - process entry point
    """Entry point for `python -m science2code.server`."""
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
