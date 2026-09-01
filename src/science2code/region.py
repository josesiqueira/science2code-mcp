"""Find the reference REGION of a paper: where the bibliography starts and
where it ends.

Why this module exists
----------------------
A paper's bibliography contains the TITLES of other papers. So the title of
Hasan 2016 physically occurs inside Kosenkov's PDF, in the reference list, and
asking "where does this title appear?" comes back VERBATIM_EXACT in Kosenkov.
A caller reads that as "Kosenkov wrote this". That is a recorded error, made in
this project's own history, and it is the error this module exists to prevent.
The fix is not to suppress the match. It is to be able to say WHERE the match
sits, so a hit inside the bibliography is labelled as one.

Why a REGION, with a start AND an end, rather than a boundary
-------------------------------------------------------------
The naive alternative has been implemented and measured. `pdfssa4met` flips a
flag on the first block containing "Reference" and calls everything from there
to end of file a reference. In the comparison that reports it, that approach
scores References F1 **0.07**, the worst result in the table. A start-only
boundary is not a weak version of this; it is a different and much worse thing.

The reason is measurable in any real corpus: **23.8 percent of the corpus used
to design this module has content AFTER the reference list**, in one case 67
percent of the whole document (appendices, supplementary material, author
biographies, a second paper bound into the same PDF). "Everything after the
heading" is therefore unsound for nearly a quarter of papers, and unsound in
the worst possible direction, because appendix prose is exactly the kind of
text a caller would quote in good faith.

So: a start, validated by density, and an end, estimated two ways with the
minimum taken.

What it will not do
-------------------
The public API returns a LABEL or nothing. It never returns a score or a
probability. Every number in here is internal. The tool refuses judgement
calls by design, and a caller handed a 0.63 will invent a threshold for it.

Method
------
Input is `pdftotext` reading-order text (no `-layout`; see the project design
rationale for why). Work is over lines, indexed by NON-BLANK line so that the
blank lines pdftotext scatters through a two-column reflow do not dilute any
density.

  start   Candidate headings are matched, filtered, and then each is validated
          by the density of reference-like lines in the 22 non-blank lines that
          follow it. The earliest candidate reaching 0.70 wins.

  end     Two estimators, minimum taken. A sliding-window density run, and a
          MAP endpoint after Zou, Le and Thoma (2010) stage 2.

Abstention
----------
Detecting nothing is a correct outcome, not a gap to be closed. See the note
above `DENSITY_ACCEPT` before touching any threshold in this file.
"""

import dataclasses
import enum
import math
import re

REGION_VERSION = "region/1.0.0"

# ---------------------------------------------------------------------------
# Tunables. All measured; see the note on each. None of these is exposed.
# ---------------------------------------------------------------------------

#: Non-blank lines examined after a candidate heading to validate it.
START_WINDOW_LINES = 22

#: Dilated reference-like density a candidate heading must reach to be
#: accepted as the start of the region.
#:
#: DO NOT LOWER THIS TO INCREASE COVERAGE. It has already been swept, from
#: 0.70 down to 0.05, on the two text-bearing papers this detector abstains on.
#: Neither yields a defensible boundary at ANY threshold: the density curve has
#: no knee, because in both papers the reading order is scrambled and the
#: reference entries are interleaved with body text line by line. Lowering the
#: threshold does not recover those two papers. It only starts accepting
#: related-work sections, which are dense in years and author initials, as the
#: bibliography. 4 abstentions in 47 papers is the designed behaviour and the
#: measured cost of 100 percent start precision.
DENSITY_ACCEPT = 0.70

#: Sliding window, in non-blank lines, for the density-run end estimator.
END_WINDOW_LINES = 25

#: Dilated density below which a window is no longer reference list.
END_DENSITY_FLOOR = 0.34

#: Consecutive failing windows required before the density run calls the end.
#:
#: This sustained-run requirement is not defensive coding, it is load bearing.
#: Elsevier reference lists are interrupted mid-list by CRediT author statement
#: blocks, competing-interest declarations and repeated page headers. In
#: ADD-29 the list is interrupted between references [8] and [9]. A single
#: failing window would end the region there and put the rest of the
#: bibliography outside it.
END_RUN_WINDOWS = 3

#: P(line is in the reference list | it shows N of the four features), for
#: N = 0, 1, 2, 3, 4. From Zou, Le and Thoma (2010), stage 2: a fixed feature
#: count to probability table, no training and no model anywhere in this file.
MAP_PROBABILITIES = (0.10, 0.35, 0.62, 0.82, 0.93)

#: Ceiling applied to any line that looks like a table row.
#:
#: This clamp is the single highest-value line in the end estimator. LA-PDFText
#: reports a References HEADING F1 of 0.994 against a References BODY F1 of
#: 0.578, and its authors attribute that collapse entirely to tables sharing
#: the reference font: a table of numeric results scores like a dense reference
#: list under every font and layout feature they use. Feature counting has the
#: same blind spot, because a results table is full of years, decimals and
#: capital initials. Without this clamp the MAP end walks straight through the
#: appendix tables of any paper that has them.
MAP_TABLE_CLAMP = 0.25

#: How far the two end estimators may diverge, as a fraction of the
#: conservative region length, before the region is reported as
#: REGION_END_UNCERTAIN rather than as a settled region.
END_DIVERGENCE_FRACTION = 0.5

#: Accepted headings needed, and the fraction of the document they must be
#: spread across, before the one-region-at-the-end model is declared
#: inapplicable. A monograph with a bibliography per chapter trips this.
MULTI_REGION_MIN_HEADINGS = 3
MULTI_REGION_SPREAD_FRACTION = 0.5


# ---------------------------------------------------------------------------
# Line features
# ---------------------------------------------------------------------------
#
# Four independent feature families. A line is reference-like if ANY fires;
# the MAP estimator uses how MANY fire. They are deliberately cheap and
# deliberately case sensitive, exactly as written: `doi` is lowercase in the
# wild, `In:` and `Proceedings` are capitalised, and folding case would make
# `Press` fire on "press the button" throughout a body section.

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-4]\d)\b")

_BIB_TOKEN_RE = re.compile(
    r"pp\.\s*\d"
    r"|vol\."
    r"|no\.\s*\d"
    r"|et al\."
    r"|doi"
    r"|10\.\d{4,}/"
    r"|In:"
    r"|In Proc"
    r"|Proceedings"
    r"|Journal"
    r"|arXiv"
    r"|LNCS"
    r"|https?://"
    r"|IEEE"
    r"|ACM"
    r"|Springer"
    r"|Eds?\."
    r"|Press"
    r"|Conference"
    r"|Workshop"
    r"|Symposium"
    r"|Trans\."
    r"|pages\s+\d"
)

# "[12]" or "12. Surname", at line start or after a column gap.
_ENTRY_MARKER_RE = re.compile(r"(?:^|\s\s)\s*(\[\d{1,3}\]|\d{1,3}\.\s+[A-Z])")

# "J. " or "J.," as in "Smith, J., and Jones, A."
_INITIALS_RE = re.compile(r"\b[A-Z]\.[ ,]")

_FEATURE_RES = (_YEAR_RE, _BIB_TOKEN_RE, _ENTRY_MARKER_RE, _INITIALS_RE)


def _feature_count(line: str) -> int:
    """How many of the four feature families fire on this line, 0 to 4."""
    return sum(1 for rx in _FEATURE_RES if rx.search(line))


def _is_reference_like(line: str) -> bool:
    """True if ANY feature family fires."""
    for rx in _FEATURE_RES:
        if rx.search(line):
            return True
    return False


_TABLE_CELL_SPLIT_RE = re.compile(r"\s{2,}|\t|\|")


def _looks_like_table_row(line: str) -> bool:
    """A cheap table-row test, for the MAP clamp only.

    Three signals, any of which is enough. Three or more short cells separated
    by a column gap; a line whose letters are outnumbered by digits and
    separators; an ASCII-ruled row. None of these fires on a reference entry in
    reading-order text, where the whole entry is one run of prose with single
    spaces.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.count("|") >= 2:
        return True
    cells = [c for c in _TABLE_CELL_SPLIT_RE.split(stripped) if c]
    if len(cells) >= 3 and all(len(c) <= 24 for c in cells):
        return True
    digits = sum(1 for c in stripped if c.isdigit())
    alpha = sum(1 for c in stripped if c.isalpha())
    if digits and digits >= alpha:
        return True
    return False


# ---------------------------------------------------------------------------
# Heading candidates
# ---------------------------------------------------------------------------

_HEADING_WORDS = (
    "REFERENCES",
    "BIBLIOGRAPHY",
    "LITERATURE CITED",
    "WORKS CITED",
    "REFERENCE LIST",
)


def _kerned(phrase: str) -> str:
    """Allow one optional space or tab between EVERY letter of a heading word.

    Small-caps and letter-spaced headings come out of pdftotext with kerning
    turned into spaces: "R EFERENCES", "R E F E R E N C E S". Four papers in
    the design corpus need this and are lost without it.
    """
    words = phrase.split(" ")
    return r"[ \t]{1,8}".join(
        r"[ \t]?".join(re.escape(ch) for ch in word) for word in words
    )


# Three details, each of which costs real recall if dropped:
#
#   ^\f?          a form feed prefixes the heading whenever the reference list
#                 opens a new page, which is the common case. Without the
#                 optional \f this loses ADD-18, REF-18 and REF-22.
#
#   (?<=\s\s)     a heading is not always at line start. In reading-order text
#                 a heading that sat at the top of column two arrives mid-line,
#                 after the column gap.
#
#   (?i: ... )    case insensitive on the heading word itself, because
#                 "References" is far more common than "REFERENCES". The
#                 section-number prefix stays case SENSITIVE so that the
#                 single-letter appendix form [A-Z] and the roman numeral form
#                 do not start matching ordinary lowercase prose.
_HEADING_RE = re.compile(
    r"(?:^\f?|(?<=\s\s))"
    r"(?:(?:[0-9]{1,2}(?:\.[0-9]{1,2})?|[IVXLC]{1,6}|[A-Z])[.)]?[ \t]{1,8})?"
    r"(?i:" + "|".join(_kerned(w) for w in _HEADING_WORDS) + r")"
    r"(?![A-Za-z])",
    re.MULTILINE,
)

# Reject the whole LINE. A line that talks ABOUT references is not a heading,
# and a dot-leader run is a table of contents entry pointing at the real one.
_LINE_REJECT_RE = re.compile(
    r"[\"“”‘’]references[\"“”‘’]"
    r"|reference format"
    r"|cross-references"
    r"|\.{5,}"
    r"|references (?:to|of|are|in|for|were|and)",
    re.IGNORECASE,
)

# Reject the MATCH. "References to the standard", "References are numbered",
# "Reference list format". A stopword within three characters of the match end
# means the word was used in a sentence, not as a heading.
_STOPWORD_AFTER_RE = re.compile(
    r"[^A-Za-z0-9]{0,3}"
    r"(?:to|of|in|for|are|is|and|or|that|which|from|used|such|list"
    r"|section|format|cited|were|have|can)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class RegionStatus(enum.Enum):
    """The closed vocabulary this module answers in. A label, never a score."""

    #: The offset is inside the region, between the start and the conservative
    #: end. As a Region status: a region was established and its two end
    #: estimators agree.
    IN_REFERENCE_REGION = "IN_REFERENCE_REGION"

    #: The offset is before the start, or past the furthest end either
    #: estimator would defend.
    OUTSIDE_REFERENCE_REGION = "OUTSIDE_REFERENCE_REGION"

    #: The offset is past the conservative end but not past the generous one,
    #: so the two estimators disagree about this particular offset.
    IN_REFERENCE_REGION_UNCERTAIN = "IN_REFERENCE_REGION_UNCERTAIN"

    #: No heading passed density validation. Nothing is known about any offset
    #: in this document. This is a correct and expected outcome.
    REGION_UNKNOWN = "REGION_UNKNOWN"

    #: A region was established, but the two end estimators diverge widely.
    REGION_END_UNCERTAIN = "REGION_END_UNCERTAIN"

    #: The one-region-at-the-end model does not fit this document, for example
    #: a monograph with a bibliography at the end of every chapter.
    REGION_MODEL_INAPPLICABLE = "REGION_MODEL_INAPPLICABLE"


@dataclasses.dataclass(frozen=True)
class Region:
    """Where the bibliography is, in character offsets into the text given.

    `end_char` is the CONSERVATIVE end, `min(density_end_char, map_end_char)`.
    Both estimators are carried separately so a caller can see that they were
    two independent estimates, but the region proper is the minimum: taking the
    minimum is what produced zero overshoots into appendix prose across the
    whole design corpus.
    """

    start_char: int | None
    end_char: int | None
    density_end_char: int | None
    map_end_char: int | None
    heading_text: str | None
    status: RegionStatus


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _line_starts(lines: list[str]) -> list[int]:
    starts = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1  # the "\n" that split consumed
    return starts


def _dilate(flags: list[bool]) -> list[bool]:
    """Widen every True by one position in each direction.

    DILATION IS LOAD BEARING, NOT DEFENSIVE. A reference list in reading-order
    text is a hanging indent: the first line of an entry carries the year, the
    marker and the initials, and the continuation lines carry only the tail of
    a title or a publisher name and score ZERO features. Undilated, the density
    at a TRUE heading measures 0.06 to 0.60 depending on how many continuation
    lines each entry runs to, which is below the acceptance threshold for a
    large share of the corpus: without this the detector fails on ADD-19,
    ADD-17 and REF-29 among others. Dilation credits a continuation line to the
    entry it belongs to, which is what the density is meant to be measuring in
    the first place.
    """
    n = len(flags)
    out = [False] * n
    for i, flag in enumerate(flags):
        if flag:
            out[i] = True
            if i > 0:
                out[i - 1] = True
            if i + 1 < n:
                out[i + 1] = True
    return out


def _prefix_sums(values: list[float]) -> list[float]:
    total = 0.0
    out = []
    for v in values:
        total += v
        out.append(total)
    return out


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _candidate_headings(text: str, lines: list[str], starts: list[int]) -> list[tuple]:
    """Every heading match that survives both rejection filters.

    Returns (match_start, line_index, heading_text).
    """
    out = []
    for m in _HEADING_RE.finditer(text):
        # Which line the match starts on.
        line_index = _line_index_of(starts, m.start())
        line = lines[line_index]
        if _LINE_REJECT_RE.search(line):
            continue
        tail = text[m.end():m.end() + 24]
        if _STOPWORD_AFTER_RE.match(tail):
            continue
        out.append((m.start(), line_index, m.group(0).strip()))
    return out


def _line_index_of(starts: list[int], offset: int) -> int:
    """Index of the line containing `offset`. Binary search over line starts."""
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _density_end(dilated: list[bool], first: int) -> int | None:
    """Non-blank index at which the sustained density run ends, or None.

    `first` is the non-blank index of the first line after the heading. Slides
    a window of END_WINDOW_LINES and returns the position of the first window
    that falls below the floor and STAYS below it for END_RUN_WINDOWS windows
    in a row.
    """
    n = len(dilated)
    below = []
    positions = []
    for w in range(first, n):
        window = dilated[w:w + END_WINDOW_LINES]
        if not window:
            break
        positions.append(w)
        below.append(sum(window) / len(window) < END_DENSITY_FLOOR)
    for i, w in enumerate(positions):
        if all(below[i:i + END_RUN_WINDOWS]) and len(below) - i >= END_RUN_WINDOWS:
            return w
    return None


def _map_end(lines: list[str], nonblank: list[int], first: int) -> int | None:
    """Non-blank index of the MAP endpoint, after Zou et al. 2010 stage 2.

    Each line gets a probability from its feature count, clamped if it looks
    like a table row, dilated by one, turned into a logit. The endpoint is the
    one maximising the running sum of those logits. That is a prefix-max scan
    in O(n), the same shape as Kadane's algorithm over the logits, with the
    left end pinned to the heading. No training and no model.
    """
    tail = nonblank[first:]
    if not tail:
        return None
    probs = []
    for idx in tail:
        line = lines[idx]
        p = MAP_PROBABILITIES[min(_feature_count(line), 4)]
        if _looks_like_table_row(line):
            p = min(p, MAP_TABLE_CLAMP)
        probs.append(p)
    # Dilation, the probability analogue of the boolean one above and for
    # exactly the same reason: a continuation line inherits its entry's
    # evidence rather than voting against it.
    dilated = [
        max(probs[max(0, i - 1):min(len(probs), i + 2)]) for i in range(len(probs))
    ]
    sums = _prefix_sums([_logit(p) for p in dilated])
    best = 0
    for i in range(1, len(sums)):
        if sums[i] > sums[best]:
            best = i
    return first + best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_region(text: str) -> Region:
    """Locate the reference region of `text`, or abstain.

    `text` is `pdftotext` reading-order output. All offsets in the result index
    into exactly the string passed in.
    """
    empty = Region(None, None, None, None, None, RegionStatus.REGION_UNKNOWN)
    if not text:
        return empty

    lines = text.split("\n")
    starts = _line_starts(lines)
    nonblank = [i for i, line in enumerate(lines) if line.strip()]
    if not nonblank:
        return empty

    flags = [_is_reference_like(lines[i]) for i in nonblank]
    dilated = _dilate(flags)

    # Map a real line index to its position in the non-blank sequence.
    nb_position = {line_index: pos for pos, line_index in enumerate(nonblank)}

    accepted = []
    for match_start, line_index, heading_text in _candidate_headings(text, lines, starts):
        pos = nb_position.get(line_index)
        if pos is None:
            continue
        window = dilated[pos + 1:pos + 1 + START_WINDOW_LINES]
        if not window:
            continue
        if sum(window) / len(window) >= DENSITY_ACCEPT:
            accepted.append((match_start, pos, heading_text))

    if not accepted:
        return empty

    # A monograph with a bibliography per chapter has several validated
    # headings spread through the body. One region with one end does not
    # describe that document, and pretending otherwise would put most of the
    # book inside "the reference region".
    if len(accepted) >= MULTI_REGION_MIN_HEADINGS:
        spread = accepted[-1][0] - accepted[0][0]
        if spread >= MULTI_REGION_SPREAD_FRACTION * len(text):
            return Region(
                None, None, None, None, accepted[0][2],
                RegionStatus.REGION_MODEL_INAPPLICABLE,
            )

    start_char, heading_pos, heading_text = accepted[0]
    first = heading_pos + 1

    d_index = _density_end(dilated, first)
    m_index = _map_end(lines, nonblank, first)

    density_end_char = _end_char_for(lines, starts, nonblank, d_index, len(text))
    map_end_char = _end_char_for(lines, starts, nonblank, m_index, len(text), inclusive=True)

    ends = [e for e in (density_end_char, map_end_char) if e is not None]
    end_char = min(ends) if ends else len(text)
    if end_char < start_char:
        end_char = start_char

    status = RegionStatus.IN_REFERENCE_REGION
    if density_end_char is not None and map_end_char is not None:
        span = max(1, end_char - start_char)
        if abs(density_end_char - map_end_char) > END_DIVERGENCE_FRACTION * span:
            status = RegionStatus.REGION_END_UNCERTAIN

    return Region(
        start_char=start_char,
        end_char=end_char,
        density_end_char=density_end_char,
        map_end_char=map_end_char,
        heading_text=heading_text,
        status=status,
    )


def _end_char_for(lines, starts, nonblank, nb_index, fallback, inclusive=False):
    """Character offset for a non-blank index returned by an end estimator."""
    if nb_index is None:
        return fallback
    if nb_index >= len(nonblank):
        return fallback
    line_index = nonblank[nb_index]
    if inclusive:
        return starts[line_index] + len(lines[line_index])
    return starts[line_index]


def classify_offset(region: Region, offset: int) -> RegionStatus:
    """Say where `offset` sits relative to `region`. A label, never a score."""
    if region.status in (
        RegionStatus.REGION_UNKNOWN,
        RegionStatus.REGION_MODEL_INAPPLICABLE,
    ):
        return region.status
    if region.start_char is None:
        return RegionStatus.REGION_UNKNOWN
    if offset < region.start_char:
        return RegionStatus.OUTSIDE_REFERENCE_REGION

    conservative = region.end_char
    if conservative is None:
        return RegionStatus.REGION_UNKNOWN
    if offset < conservative:
        return RegionStatus.IN_REFERENCE_REGION

    generous = max(
        e for e in (region.density_end_char, region.map_end_char, conservative)
        if e is not None
    )
    if offset < generous:
        return RegionStatus.IN_REFERENCE_REGION_UNCERTAIN
    return RegionStatus.OUTSIDE_REFERENCE_REGION


__all__ = [
    "REGION_VERSION",
    "Region",
    "RegionStatus",
    "classify_offset",
    "detect_region",
]
