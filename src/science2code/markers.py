"""Citation-marker detection, reference-region reporting, and the citance span.

This module is stdlib only. It imports `science2code.envelope` for the two
constants that describe the context window and nothing else, and it imports
`science2code.region` lazily and defensively, so that this module loads and
answers on a bare Python whether or not the region model is installed.


WHAT THIS MODULE WILL NEVER DO
------------------------------
It will never say whose claim a sentence carries.

That is not modesty, it is the measured decidability ceiling. Trained human
annotators judging citation accuracy agree at Cohen's kappa 0.18 to 0.31,
which is slight-to-fair agreement: two experts reading the same sentence and
the same cited paper reach different answers most of the time. The one
dedicated attribution-transfer study reaches Krippendorff's alpha .654 against
a human ceiling of .806, so even the best reported system sits below the point
where humans stop disagreeing with each other. A field in a response saying
"this claim belongs to reference 12" would therefore be a number this project
cannot stand behind, presented in a shape a caller would act on.

So every function here returns a COUNT, a LABEL from a closed set, or a SPAN
of the document's own characters. Never a probability, never a boolean, and
never an attribution. The one primitive with a published precision of 1.00
behind it is the citance itself: Sarol et al., Bioinformatics 2024;40(7):
btae420, where the trivial "return the sentence containing the marker"
baseline scored P 1.00 / R 0.90 / F1 0.94 and beat a fine-tuned PubMedBERT.
Returning the sentence and letting the human read it is the strongest move
available, and it is available precisely because it decides nothing.


THE THREE MARKER STYLES, AND THE ONE THAT CANNOT BE SEEN
--------------------------------------------------------
Counted over a 47-paper corpus: 32 papers cite with numeric brackets, 10 with
author-year, 2 with superscripts.

NUMERIC BRACKETS are trivial and near-perfect. On 72 hand-checked markers
across 12 papers the bracketed pattern scored 72/72, precision 1.00, with no
false alarm at all. Square brackets in this literature are essentially never
anything else, so the pattern is kept strict rather than widened.

AUTHOR-YEAR is where the work is. A naive "parenthesis, something, year,
parenthesis" scores 77.0% precision. The strict form below scores 95.6%
precision at 1.000 recall on 113 hand-labelled matches. Of the 26 naive false
alarms, 20 were JOURNAL RUNNING HEADS, the line pdftotext leaves on every page
of a modern journal PDF:

    Scientific Data |   (2026) 13:936 | https://doi.org/10....

Four more were reference-list entries in papers where no region was detected,
and only 2 of 113 (1.8%) are genuinely undecidable: a named document that
carries a year, such as "the National Research Act (1974)", is the same
character sequence as a citation and no amount of context separates them.

SUPERSCRIPTS ARE THE SILENT FAILURE, and they are the reason this module
exists in the shape it does. `pdftotext` has no layout information, so it
glues a superscript citation to the word before it: `Principles4`,
`hallucination26,27`, `FAIRness5,9`. One paper in the corpus has 92 such
occurrences and ZERO bracketed markers anywhere. Those strings are not
separable from `GPT4`, `COVID19` or `Section3` without the layout the
extractor threw away, so this module DOES NOT TRY TO DETECT THEM.

It detects their ABSENCE instead. A paper with a detected reference region and
zero bracketed and zero strict author-year markers in its body is reported as
MARKER_STYLE_UNDETECTABLE. Before that label existed, such a paper answered
"no citation marker near this span", which is character-for-character
indistinguishable from "this is the author's own uncited claim". That is a
false negative shaped exactly like a positive finding, which makes it the most
dangerous state this system can be in, and it is now named.


TWO INDEPENDENT REFERENCE-LIST SIGNALS, DELIBERATELY NOT MERGED
---------------------------------------------------------------
A paper's bibliography holds OTHER papers' titles. Search a title, find it in
the citing paper, and a caller concludes the citing author wrote it. Two
separate tests report on that, and they are reported separately because they
fail differently:

  1. REGION MEMBERSHIP, from `science2code.region`: does the located offset
     fall inside the detected reference region. Precise, and silent whenever
     the region model did not apply.

  2. THE LOCAL-BLOCK TEST, here: does the blank-line-delimited block around the
     hit, capped at LOCAL_BLOCK_CHARS either side, carry at least
     LOCAL_BLOCK_MIN_SIGNALS of {year, bibliographic token, initials, entry
     marker}. Measured: it catches 31 of 36 known reference-list hits on its
     own, and the UNION of the two catches 36 of 36. Its false-positive rate
     over 1,640 genuine body offsets is 2.4%, firing on numbered lists and on
     author-email footnote blocks.

A 2.4% false-positive rate is small enough to be worth reporting and far too
large to act on unseen, so the local-block result is surfaced as a WARNING and
never as a verdict. The two are never combined into one score: a caller who is
told "reference-like: 0.83" cannot tell which test fired, and the two need
different actions from the human.

A HIT IN A REFERENCE REGION IS STILL RETURNED. Labelling it is a fact;
dropping it would be a judgement, and this package does not make judgements.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from science2code.envelope import CITATION_CONTEXT_CHARS

__all__ = [
    "MARKERS_VERSION",
    "MARKER_STYLE_NUMERIC_BRACKETED",
    "MARKER_STYLE_AUTHOR_YEAR",
    "MARKER_STYLE_MIXED",
    "MARKER_STYLE_UNDETECTABLE",
    "MARKER_STYLE_NOT_ASSESSED",
    "MARKER_STYLES",
    "MARKER_STYLE_RULE",
    "REGION_STATUS_NAMES",
    "REGION_UNKNOWN",
    "REGION_RULE",
    "LOCAL_BLOCK_WARNING",
    "LOCAL_BLOCK_CLEAR",
    "LOCAL_BLOCK_NOT_ASSESSED",
    "LOCAL_BLOCK_LABELS",
    "LOCAL_BLOCK_CHARS",
    "LOCAL_BLOCK_MIN_SIGNALS",
    "LOCAL_BLOCK_SIGNAL_NAMES",
    "LOCAL_BLOCK_RULE",
    "SENTENCE_RULE",
    "ATTRIBUTION_CEILING",
    "RUNNING_HEAD_WINDOW",
    "SENTENCE_MAX_CHARS",
    "bracketed_spans",
    "author_year_spans",
    "marker_spans",
    "count_bracketed",
    "count_author_year",
    "count_markers",
    "sentence_span",
    "marker_style",
    "local_block_signals",
    "region_status",
    "region_available",
    "citation_marker_block",
    "reference_region_block",
]

MARKERS_VERSION = "markers/1.0.0"


# ---------------------------------------------------------------------------
# closed vocabularies
# ---------------------------------------------------------------------------

#: The paper cites with bracketed reference numbers, and nothing else was seen.
MARKER_STYLE_NUMERIC_BRACKETED = "MARKER_STYLE_NUMERIC_BRACKETED"

#: The paper cites with author-year parentheticals, and no bracketed number
#: was seen.
MARKER_STYLE_AUTHOR_YEAR = "MARKER_STYLE_AUTHOR_YEAR"

#: Both were seen in the body. Common in papers that number their references
#: and also name a law or a standard by year in running prose.
MARKER_STYLE_MIXED = "MARKER_STYLE_MIXED"

#: THE IMPORTANT ONE. A reference region was detected, so the paper does cite
#: something, and yet its body carries zero bracketed and zero strict
#: author-year markers. The citation style is one this extractor cannot see,
#: which in this corpus means superscripts glued to the preceding word. For
#: such a paper a marker count of zero says nothing whatever about whether a
#: passage is attributed, and must not be read as though it did.
MARKER_STYLE_UNDETECTABLE = "MARKER_STYLE_UNDETECTABLE"

#: No reference region was detected and no marker was seen, so the two cases
#: above cannot be separated. Distinct from UNDETECTABLE, which is a finding.
MARKER_STYLE_NOT_ASSESSED = "MARKER_STYLE_NOT_ASSESSED"

MARKER_STYLES: frozenset[str] = frozenset(
    {
        MARKER_STYLE_NUMERIC_BRACKETED,
        MARKER_STYLE_AUTHOR_YEAR,
        MARKER_STYLE_MIXED,
        MARKER_STYLE_UNDETECTABLE,
        MARKER_STYLE_NOT_ASSESSED,
    }
)

MARKER_STYLE_RULE = (
    "marker_style names which citation style was seen in this document's body, "
    "outside any detected reference region. %s means a reference region was "
    "detected and the body carries no bracketed number and no strict "
    "author-year marker at all, so this document cites in a style this "
    "extractor cannot see: 2 of 47 papers measured cite with superscripts, "
    "which pdftotext glues to the preceding word as Principles4 or "
    "hallucination26,27, and those are not separable from GPT4 or Section3 "
    "without layout information. When you see that label, a marker count of "
    "zero carries no information about this passage and must not be read as "
    "one. %s means no reference region was detected either, so the two cases "
    "cannot be told apart."
    % (MARKER_STYLE_UNDETECTABLE, MARKER_STYLE_NOT_ASSESSED)
)

#: The status vocabulary `science2code.region` returns, mirrored here so that
#: an unrecognised member degrades to REGION_UNKNOWN instead of reaching a
#: caller as a label this server never described.
REGION_UNKNOWN = "REGION_UNKNOWN"

REGION_STATUS_NAMES: frozenset[str] = frozenset(
    {
        "IN_REFERENCE_REGION",
        "OUTSIDE_REFERENCE_REGION",
        "IN_REFERENCE_REGION_UNCERTAIN",
        REGION_UNKNOWN,
        "REGION_END_UNCERTAIN",
        "REGION_MODEL_INAPPLICABLE",
    }
)

#: The region model is not installed in this build at all. Reported so that a
#: missing signal is never read as a signal that fired negative.
REGION_NOT_INSTALLED = "REGION_UNKNOWN"

REGION_RULE = (
    "status says where the located offset falls relative to this document's "
    "detected reference region. A bibliography holds OTHER papers' titles, so "
    "a string located inside one was written by whoever this document cites, "
    "not by whoever wrote this document, and reporting it as this document's "
    "words would be the wrong-author error. A hit inside the region is still "
    "returned and never dropped: labelling it is a fact and dropping it would "
    "be a judgement. %s means the region model reached no answer for this "
    "document, which is not the same as the offset being outside the region."
    % REGION_UNKNOWN
)


#: At least LOCAL_BLOCK_MIN_SIGNALS of the four signals were found in the block
#: around the hit. A warning, never a verdict: 2.4% of genuine body offsets
#: trip it, mostly numbered lists and author-email footer blocks.
LOCAL_BLOCK_WARNING = "REFERENCE_LIKE_BLOCK_WARNING"

#: Fewer than LOCAL_BLOCK_MIN_SIGNALS were found.
LOCAL_BLOCK_CLEAR = "NO_REFERENCE_LIKE_BLOCK"

#: The test could not run, because the document's own line structure was not
#: available at this offset. The normalised rendering collapses every run of
#: whitespace to one space, so this test reads the raw characters or it does
#: not run at all.
LOCAL_BLOCK_NOT_ASSESSED = "LOCAL_BLOCK_NOT_ASSESSED"

LOCAL_BLOCK_LABELS: frozenset[str] = frozenset(
    {LOCAL_BLOCK_WARNING, LOCAL_BLOCK_CLEAR, LOCAL_BLOCK_NOT_ASSESSED}
)

#: How far either side of the hit the block is allowed to run when no blank
#: line bounds it.
LOCAL_BLOCK_CHARS = 700

#: How many of the four signals must be present. Measured at 3: it catches 31
#: of 36 known reference-list hits alone and trips on 2.4% of 1,640 genuine
#: body offsets.
LOCAL_BLOCK_MIN_SIGNALS = 3

LOCAL_BLOCK_SIGNAL_NAMES: tuple[str, ...] = (
    "year",
    "bibliographic_token",
    "initials",
    "entry_marker",
)

LOCAL_BLOCK_RULE = (
    "A second and independent reference-list test, run over the document's own "
    "raw characters rather than the normalised rendering. The blank-line "
    "delimited block around the hit, capped at %d characters either side, is "
    "checked for four signals: a year, a bibliographic token such as pp. or "
    "doi, author initials, and a numbered entry marker at the start of a line. "
    "%d of the 4 raise %s. Measured: alone it catches 31 of 36 known "
    "reference-list hits, and together with region membership 36 of 36, at a "
    "cost of 2.4%% of 1,640 genuine body offsets, which are numbered lists and "
    "author-email footer blocks. That rate is small enough to report and far "
    "too large to act on unseen, so this is a warning and not a verdict: open "
    "the document at the offset and look."
    % (
        LOCAL_BLOCK_CHARS,
        LOCAL_BLOCK_MIN_SIGNALS,
        LOCAL_BLOCK_WARNING,
    )
)

SENTENCE_RULE = (
    "sentence.document_text holds the document's own sentence containing the "
    "located span, and sentence.char_interval says where that sentence sits. "
    "It is named document_text for the same reason the located characters are: "
    "the name says whose words these are. It is given because the sentence is "
    "the unit the marker attaches to: a "
    "quote that stops one character before its marker looks clean, and the "
    "sentence around it does not. Returning the sentence is the strongest move "
    "available here, and it is available precisely because it decides nothing. "
    "The trivial sentence baseline has a published precision of 1.00 for this "
    "task (Sarol et al., Bioinformatics 2024;40(7):btae420, P 1.00 / R 0.90 / "
    "F1 0.94), beating a fine-tuned PubMedBERT. Read it and judge."
)

ATTRIBUTION_CEILING = (
    "These are counts, labels and spans. None of them says whose claim a "
    "sentence carries, and this server will not add a field that does. Trained "
    "human annotators judging citation accuracy agree at Cohen's kappa 0.18 to "
    "0.31, and the one dedicated attribution study reaches alpha .654 against "
    "a human ceiling of .806, so the question sits above what any mechanical "
    "test here can decide. Read the sentence in the document."
)


# ---------------------------------------------------------------------------
# numeric bracketed markers: precision 72/72 on the hand-checked sample
# ---------------------------------------------------------------------------

# [12], [3, 4], [7-9], [1; 2]. The dash class carries the ASCII hyphen and the
# two long dashes by escape, because the raw document has not been through the
# punctuation fold when this runs over raw characters, and has been when it
# runs over the normalised rendering. Both must match.
_BRACKETED = re.compile(
    r"\[\s*\d{1,3}(?:\s*[,;\u2013\u2014-]\s*\d{1,3})*\s*\]"
)


# ---------------------------------------------------------------------------
# author-year markers: 95.6% precision at 1.000 recall on 113 labelled matches
# ---------------------------------------------------------------------------

_YEAR = r"(?:1[5-9]|20)\d{2}[a-z]?"
_YEAR_RE = re.compile(r"\b" + _YEAR + r"\b")

# A capitalised NAME token: an initial capital followed by a lower-case letter,
# which is the shape of a surname. Deliberately not any capitalised token: an
# all-capitals acronym is not a name, and admitting one is what turns
# "(equivalent to 9.2 million or USD 120,000 in 2023)", "COM(2019)168" and the
# grant identifier "(JAES/2024/EVIL-AI)" into citations. Measured on this
# corpus, requiring the surname shape removes four false alarms in a
# 37-match body sample and costs no marker at all, because the standards
# citations it declines, "(ISO/IEC TR 24028:2020 [10])", carry a bracketed
# number that the bracketed detector counts anyway.
_NAME_TOKEN = re.compile(r"\b[A-Z][a-z\u00df-\u024f'\u2019]")

_ET_AL = re.compile(r"\bet\s+al\b", re.IGNORECASE)
_CONJUNCTION = re.compile(r"&|\band\b")

# Everything inside one level of parentheses. The 60-character ceiling is
# inherited from the pattern this module replaces and it earns its place: a
# citation parenthetical is short, and the long ones in this corpus are table
# cells, monetary asides and two-column text that pdftotext interleaved, all of
# which merely happen to end in a date. Measured on a 37-match body sample,
# the ceiling removes five false alarms and one genuine marker, and that one
# carries a bracketed number beside it that is counted regardless.
_PARENTHETICAL = re.compile(r"\(([^()]{0,60}?)\)")

# The same surname shape as _NAME_TOKEN, for the same reason: without it
# "COM(2019)168", the number of a European Commission document, reads as
# "Surname (year)".
_SURNAME = (
    r"[A-Z][a-z\u00df-\u024f'\u2019][A-Za-z\u00c0-\u024f'\u2019-]*"
)

# The narrative form: Smith (2019), Smith et al. (2019), Smith and Jones
# (2019), Smith & Jones (2019). Here the parenthetical is a bare year and the
# name sits outside it, which is why it needs its own pattern.
_NARRATIVE = re.compile(
    _SURNAME
    + r"(?:\s+(?:et\s+al\.?|and\s+" + _SURNAME + r"|&\s*" + _SURNAME + r"))?"
    + r",?\s*\((?:" + _YEAR + r")\)"
)

#: How far either side of a candidate the running-head test looks.
RUNNING_HEAD_WINDOW = 45

# The filter that does most of the work. Of 26 naive false alarms, 20 were
# journal running heads: "Scientific Data |   (2026) 13:936 |
# https://doi.org/10...". Each alternative below is one shape seen in that
# line. `Journal` stays case sensitive, because a body sentence may well say
# "published in a journal" and that is not a running head; `doi` and the URL
# forms do not, because they are never prose either way.
_RUNNING_HEAD = re.compile(
    r"\|\s*\("            # a pipe immediately before the parenthetical
    r"|\)\s*\d{1,3}\s*:\s*\d"   # ") 13:936", volume and article number
    r"|\bJournal\b"
    r"|\bpp\."
    r"|(?i:\bdoi\b)"
    r"|(?i:https?://)"
    r"|(?i:www\.)"
)


def _looks_like_running_head(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - RUNNING_HEAD_WINDOW):end + RUNNING_HEAD_WINDOW]
    return _RUNNING_HEAD.search(window) is not None


def bracketed_spans(text: str) -> list[tuple[int, int]]:
    """Every bracketed reference number in `text`, as (start, end) offsets.

    Kept strict on purpose. On 72 hand-checked markers across 12 papers this
    pattern matched 72 and raised no false alarm, so widening it can only cost
    precision.
    """
    if not isinstance(text, str) or not text:
        return []
    return [m.span() for m in _BRACKETED.finditer(text)]


def author_year_spans(text: str) -> list[tuple[int, int]]:
    """Every STRICT author-year citation in `text`, as (start, end) offsets.

    Strict means both of these:

      * the parenthetical carries `et al.`, `&`, `and`, or a capitalised name
        token standing before the year, OR the narrative form `Surname (et al.
        | and X)? (year)` matches;
      * AND the surrounding RUNNING_HEAD_WINDOW characters do not look like a
        journal running head.

    The naive version of the first half alone scores 77.0% precision. Both
    halves together score 95.6% at 1.000 recall on 113 hand-labelled matches,
    and the second half is what buys the difference: 20 of the 26 naive false
    alarms were running heads.
    """
    if not isinstance(text, str) or not text:
        return []
    spans: dict[tuple[int, int], None] = {}

    for m in _PARENTHETICAL.finditer(text):
        inner = m.group(1)
        year = _YEAR_RE.search(inner)
        if year is None:
            continue
        before_year = inner[:year.start()]
        named = (
            _ET_AL.search(inner) is not None
            or _CONJUNCTION.search(inner) is not None
            or _NAME_TOKEN.search(before_year) is not None
        )
        if not named:
            continue
        if _looks_like_running_head(text, m.start(), m.end()):
            continue
        spans[m.span()] = None

    for m in _NARRATIVE.finditer(text):
        if _looks_like_running_head(text, m.start(), m.end()):
            continue
        spans[m.span()] = None

    return sorted(spans)


def marker_spans(text: str) -> list[tuple[int, int]]:
    """Both kinds together, sorted, with overlaps collapsed to one marker."""
    spans = sorted(bracketed_spans(text) + author_year_spans(text))
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            continue
        merged.append((start, end))
    return merged


def count_bracketed(text: str) -> int:
    """How many bracketed reference numbers occur in `text`."""
    return len(bracketed_spans(text))


def count_author_year(text: str) -> int:
    """How many strict author-year citations occur in `text`."""
    return len(author_year_spans(text))


def count_markers(text: str) -> int:
    """How many citation markers of either kind occur in `text`."""
    return len(marker_spans(text))


# ---------------------------------------------------------------------------
# the citance
# ---------------------------------------------------------------------------

#: A sentence longer than this is almost always a table row, a reference list
#: run together, or a heading block that carries no terminator at all. Cut
#: rather than returned whole, and the cut is reported by the interval.
SENTENCE_MAX_CHARS = 1000

# Tokens that end in a period and do not end a sentence. Matched at the end of
# the candidate, case sensitively where case carries the signal.
_ABBREVIATIONS = (
    "et al.", "e.g.", "i.e.", "cf.", "vs.", "etc.", "approx.", "resp.",
    "Fig.", "Figs.", "Eq.", "Eqs.", "Ref.", "Refs.", "Sec.", "Tab.",
    "No.", "Nos.", "Vol.", "pp.", "ed.", "eds.", "al.", "Dr.", "Prof.",
    "Mr.", "Ms.", "Mrs.", "St.", "Inc.", "Ltd.", "Univ.", "Dept.",
)

_SENTENCE_END = re.compile(r"[.?!][\"'\u2019\u201d)\]]*\s")


def _is_boundary(text: str, end_of_terminator: int) -> bool:
    """Is the terminator ending at `end_of_terminator` a real sentence end?"""
    head = text[:end_of_terminator]
    for abbrev in _ABBREVIATIONS:
        if head.endswith(abbrev):
            return False
    # A period between two digits is a decimal point or a section number.
    if len(head) >= 2 and head[-1] == "." and head[-2].isdigit():
        tail = text[end_of_terminator:end_of_terminator + 2].lstrip()
        if tail[:1].isdigit():
            return False
    # A single capital before the period is an initial: "Smith, J. A."
    if (
        len(head) >= 2
        and head[-1] == "."
        and head[-2].isupper()
        and (len(head) < 3 or not head[-3].isalpha())
    ):
        return False
    return True


def sentence_span(
    text: str, start: int, end: int, floor_offset: int = 0
) -> tuple[int, int] | None:
    """The span of the sentence containing [start, end) in `text`.

    Returns None when the offsets do not index `text`. Never returns a
    judgement, and never trims the span to the located characters: the point is
    to hand back MORE than the caller located, because the marker that owns the
    claim is routinely just outside what was located.

    `floor_offset` is a hard left wall the expansion may not cross. It exists
    so a document that carries a toolchain header cannot have that header's
    text swept into a body sentence: after normalisation the page break between
    header and body is a single space, so a leftward walk would otherwise run
    straight into the header and return it as the paper's own words.
    """
    if not isinstance(text, str) or not text:
        return None
    if start is None or end is None:
        return None
    if start < 0 or start > len(text) or end < start:
        return None
    end = min(end, len(text))
    floor_offset = max(0, min(floor_offset, start))

    floor = max(floor_offset, start - SENTENCE_MAX_CHARS)
    left = floor
    for m in _SENTENCE_END.finditer(text, floor, max(floor, start)):
        if _is_boundary(text, m.start() + 1):
            left = m.end()
    while left < len(text) and text[left].isspace():
        left += 1
    if left > start:
        left = start

    ceiling = min(len(text), end + SENTENCE_MAX_CHARS)
    right = ceiling
    # Start the terminator search one character back, so a sentence end that
    # sits on the LAST character of the located span (the common "...[12]."
    # case, where `end` is just past the full stop) is seen. Starting at `end`
    # skips it and swallows the whole next sentence and its markers. Never
    # search before the sentence start.
    right_from = max(left, end - 1)
    for m in _SENTENCE_END.finditer(text, right_from, ceiling):
        if _is_boundary(text, m.start() + 1):
            right = m.start() + 1
            break
    else:
        # No terminator after the span. A final sentence with no trailing
        # whitespace is the common case, so try the very end of the text.
        if ceiling == len(text) and text[max(0, len(text) - 1):] in (".", "?", "!"):
            right = len(text)
    if right < end:
        right = end
    return left, right


# ---------------------------------------------------------------------------
# document-level style, including the label for the style that cannot be seen
# ---------------------------------------------------------------------------

_STYLE_CACHE: dict[tuple[str, bool], str] = {}
_STYLE_CACHE_MAX = 64


def marker_style(body_text: str, region_detected: bool) -> str:
    """Which citation style this document's BODY uses, as a closed label.

    `region_detected` says whether `science2code.region` found and bounded a
    reference region for this document. It is the difference between the two
    labels that matter: with a region and no markers the answer is
    MARKER_STYLE_UNDETECTABLE, which is a finding about the extractor, and
    without one the answer is MARKER_STYLE_NOT_ASSESSED, which is a statement
    that nothing was established.

    Never returns a count and never returns a boolean. The caller reads the
    label; the counts for the span it asked about are reported separately.
    """
    if not isinstance(body_text, str):
        body_text = ""
    key = (
        hashlib.sha256(body_text.encode("utf-8", "replace")).hexdigest(),
        bool(region_detected),
    )
    hit = _STYLE_CACHE.get(key)
    if hit is not None:
        return hit
    brackets = count_bracketed(body_text)
    years = count_author_year(body_text)
    if brackets and years:
        style = MARKER_STYLE_MIXED
    elif brackets:
        style = MARKER_STYLE_NUMERIC_BRACKETED
    elif years:
        style = MARKER_STYLE_AUTHOR_YEAR
    elif region_detected:
        style = MARKER_STYLE_UNDETECTABLE
    else:
        style = MARKER_STYLE_NOT_ASSESSED
    if len(_STYLE_CACHE) >= _STYLE_CACHE_MAX:
        _STYLE_CACHE.clear()
    _STYLE_CACHE[key] = style
    return style


# ---------------------------------------------------------------------------
# the local-block test: the second, independent reference-list signal
# ---------------------------------------------------------------------------

_BLANK_LINE = re.compile(r"\n[ \t]*\n")

_LOCAL_YEAR = re.compile(r"\b(?:1[5-9]|20)\d{2}[a-z]?\b")

_LOCAL_BIBLIOGRAPHIC = re.compile(
    r"\bpp\.|\bvol\.|\bno\.|\bnos\.|\beds?\.|\bIn:"
    r"|\bProceedings\b|\bJournal\b|\bConference\b|\bPress\b|\barXiv\b"
    r"|\bISBN\b|\bRetrieved\b|\bAvailable\s+at\b|\bAccessed\b"
    r"|(?i:\bdoi\b)|(?i:https?://)"
    r"|\b\d{1,4}\s*\(\s*\d{1,3}\s*\)\s*[,:]"     # 40(7),
    r"|\b\d{1,4}\s*:\s*\d{1,5}\s*[-\u2013]\s*\d"  # 13:936-948
)

# "Smith, J. A." or "J. A. Smith". Two initial-shaped tokens are required,
# because one on its own is as likely to be "Fig. A" or a sentence-initial
# abbreviation as it is to be an author.
_LOCAL_INITIAL = re.compile(r"\b[A-Z]\.(?=[\s,;)\]]|$)")

_LOCAL_ENTRY_MARKER = re.compile(
    r"(?m)^[ \t]*(?:\[\s*\d{1,3}\s*\]|\(\s*\d{1,3}\s*\)|\d{1,3}\.)[ \t]"
)


def local_block_signals(raw_text: str, raw_offset: int) -> dict[str, Any]:
    """Run the local-block test at a RAW offset in a document's RAW characters.

    Raw, not normalised, and the distinction is load-bearing: the normaliser
    collapses every run of whitespace to one space, so there are no blank lines
    left in the normalised rendering and a blank-line-delimited block cannot be
    found there at all.

    Returns the label, the count of signals found and their names. Never a
    score, and never a decision about the hit.
    """
    if not isinstance(raw_text, str) or not raw_text or raw_offset is None:
        return {
            "signal": LOCAL_BLOCK_NOT_ASSESSED,
            "signals_found": 0,
            "signals_named": [],
            "signals_looked_for": list(LOCAL_BLOCK_SIGNAL_NAMES),
            "block_chars": LOCAL_BLOCK_CHARS,
            "signals_required": LOCAL_BLOCK_MIN_SIGNALS,
            "rule": LOCAL_BLOCK_RULE,
        }
    offset = max(0, min(int(raw_offset), len(raw_text)))

    floor = max(0, offset - LOCAL_BLOCK_CHARS)
    ceiling = min(len(raw_text), offset + LOCAL_BLOCK_CHARS)
    start = floor
    for m in _BLANK_LINE.finditer(raw_text, floor, offset):
        start = m.end()
    end = ceiling
    m = _BLANK_LINE.search(raw_text, offset, ceiling)
    if m is not None:
        end = m.start()
    block = raw_text[start:end]

    named: list[str] = []
    if _LOCAL_YEAR.search(block):
        named.append("year")
    if _LOCAL_BIBLIOGRAPHIC.search(block):
        named.append("bibliographic_token")
    if len(_LOCAL_INITIAL.findall(block)) >= 2:
        named.append("initials")
    if _LOCAL_ENTRY_MARKER.search(block):
        named.append("entry_marker")

    found = len(named)
    return {
        "signal": (
            LOCAL_BLOCK_WARNING if found >= LOCAL_BLOCK_MIN_SIGNALS
            else LOCAL_BLOCK_CLEAR
        ),
        "signals_found": found,
        "signals_named": named,
        "signals_looked_for": list(LOCAL_BLOCK_SIGNAL_NAMES),
        "block_chars": LOCAL_BLOCK_CHARS,
        "signals_required": LOCAL_BLOCK_MIN_SIGNALS,
        "rule": LOCAL_BLOCK_RULE,
    }


# ---------------------------------------------------------------------------
# the region adapter: defensive, because region.py may not be installed
# ---------------------------------------------------------------------------


def _region_module() -> Any:
    """`science2code.region`, or None when this build does not carry it.

    Imported here rather than at module scope so that this module answers on a
    build without it, and so that a test can install a stub in sys.modules.
    """
    try:
        import science2code.region as module
    except Exception:
        return None
    return module


def region_available() -> str:
    """Whether the region model is importable, as a label rather than a flag."""
    return "REGION_MODEL_PRESENT" if _region_module() is not None else "REGION_MODEL_ABSENT"


def _status_name(obj: Any) -> str | None:
    if obj is None:
        return None
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(obj, str):
        return obj
    value = getattr(obj, "value", None)
    if isinstance(value, str):
        return value
    return None


def _region_own_status(region: Any) -> str | None:
    return _status_name(getattr(region, "status", None)) or _status_name(region)


def _region_start(region: Any) -> int | None:
    """Where the reference region begins, or None when none was bounded."""
    if region is None:
        return None
    for attribute in ("start_char", "start", "start_pos", "reference_start", "begin"):
        value = getattr(region, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    if isinstance(region, (tuple, list)) and region:
        first = region[0]
        if isinstance(first, int) and not isinstance(first, bool) and first >= 0:
            return first
    if isinstance(region, dict):
        value = region.get("start")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


_UNDETECTED_REGION_STATUSES = frozenset({REGION_UNKNOWN, "REGION_MODEL_INAPPLICABLE"})


def region_status(raw_text: str, raw_offset: int | None) -> tuple[str, int | None, str | None]:
    """Classify a RAW offset against the reference region of RAW `raw_text`.

    RAW, on both arguments, and the distinction is load-bearing. The region
    model reads `pdftotext` reading-order output and every offset it returns
    indexes exactly the string it was given, so handing it the normalised
    rendering would hand back offsets into a different string. The caller maps
    a normalised offset back through the index map first; `_raw_offset` in the
    server does that.

    Returns (status label, region start or None, region version or None). The
    status is always a member of REGION_STATUS_NAMES: anything the region model
    hands back that is not in that closed set degrades to REGION_UNKNOWN rather
    than reaching a caller as a label this server never described, which is the
    same rule the outcome vocabulary follows.
    """
    norm_text, offset = raw_text, raw_offset
    module = _region_module()
    if module is None:
        return REGION_UNKNOWN, None, None

    version = getattr(module, "REGION_VERSION", None)
    version = str(version) if isinstance(version, str) else None

    detect = getattr(module, "detect_region", None)
    region = None
    if detect is not None:
        try:
            region = detect(norm_text)
        except Exception:
            region = None

    start = _region_start(region)
    own = _region_own_status(region)
    if own in _UNDETECTED_REGION_STATUSES:
        start = None

    if offset is None:
        return (REGION_UNKNOWN if start is None else own or REGION_UNKNOWN), start, version

    classify = getattr(module, "classify_offset", None)
    if classify is not None:
        for args in ((region, offset), (norm_text, offset)):
            try:
                name = _status_name(classify(*args))
            except Exception:
                continue
            if name in REGION_STATUS_NAMES:
                return name, start, version
    return REGION_UNKNOWN, start, version


# ---------------------------------------------------------------------------
# the two blocks that ride on every located response
# ---------------------------------------------------------------------------


def citation_marker_block(
    norm_text: str,
    offset: int | None,
    length: int | None,
    *,
    style: str = MARKER_STYLE_NOT_ASSESSED,
    context_chars: int = CITATION_CONTEXT_CHARS,
    body_floor: int = 0,
) -> dict[str, Any]:
    """The extended citation-marker block for one located span.

    Reports four counts rather than two, because a bracketed number and an
    author-year parenthetical have different precision behind them (1.00 and
    0.956) and merging them would hide which one fired. Reports them inside the
    span and beside it separately, because a quote that stops one character
    before its marker looks clean and is the case that has caught a caller out.
    Reports the document's own sentence around the span, because that is the
    unit the marker attaches to and the only primitive here with a published
    precision of 1.00.
    """
    empty_counts = {
        "in_located_characters": 0,
        "in_surrounding_characters": 0,
        "bracketed_in_located_characters": 0,
        "bracketed_in_surrounding_characters": 0,
        "author_year_in_located_characters": 0,
        "author_year_in_surrounding_characters": 0,
    }
    block: dict[str, Any] = dict(empty_counts)
    block.update(
        {
            "context_chars": int(context_chars),
            "marker_style": style if style in MARKER_STYLES else MARKER_STYLE_NOT_ASSESSED,
            "marker_style_rule": MARKER_STYLE_RULE,
            "sentence": None,
            "sentence_rule": SENTENCE_RULE,
            "markers_version": MARKERS_VERSION,
            "ceiling": ATTRIBUTION_CEILING,
        }
    )
    if not isinstance(norm_text, str) or offset is None or length is None:
        return block

    start = max(0, int(offset))
    end = min(len(norm_text), start + max(0, int(length)))

    # Scan the located span and its context as ONE region, then classify each
    # marker by whether it overlaps the located interval. Counting three
    # disjoint slices (before, located, after) loses a marker that straddles a
    # boundary: "[12]" split as "[1" inside and "2]" outside is a whole marker
    # in neither slice, so it vanishes from both counts. That is worse than the
    # off-by-one it was meant to surface, so overlap classification replaces it.
    # The header floor keeps both the marker context and, below, the sentence
    # from reaching back into a toolchain header that sits before the body.
    floor = max(0, min(int(body_floor), start))
    region_start = max(floor, start - context_chars)
    region_end = min(len(norm_text), end + context_chars)
    region = norm_text[region_start:region_end]

    def _split(spans: list[tuple[int, int]]) -> tuple[int, int]:
        located_count = surrounding_count = 0
        for ms, me in spans:
            a, b = ms + region_start, me + region_start
            if a < end and b > start:  # overlaps the located interval
                located_count += 1
            else:
                surrounding_count += 1
        return located_count, surrounding_count

    b_located, b_surrounding = _split(bracketed_spans(region))
    ay_located, ay_surrounding = _split(author_year_spans(region))

    block["bracketed_in_located_characters"] = b_located
    block["bracketed_in_surrounding_characters"] = b_surrounding
    block["author_year_in_located_characters"] = ay_located
    block["author_year_in_surrounding_characters"] = ay_surrounding
    block["in_located_characters"] = (
        block["bracketed_in_located_characters"]
        + block["author_year_in_located_characters"]
    )
    block["in_surrounding_characters"] = (
        block["bracketed_in_surrounding_characters"]
        + block["author_year_in_surrounding_characters"]
    )

    span = sentence_span(norm_text, start, end, floor_offset=floor)
    if span is not None:
        # `document_text` and not some new name: that key is already the one
        # this package exempts from its own prose rules, because the value
        # under it is the document's words and not the server's, and a real
        # paper may well contain a word the server is forbidden to write.
        # Reusing it keeps the exemption list exactly as wide as it was.
        block["sentence"] = {
            "document_text": norm_text[span[0]:span[1]],
            "char_interval": {
                "start_pos": span[0],
                "end_pos": span[1],
                "basis": "normalised_document",
            },
            "rule": SENTENCE_RULE,
        }
    return block


def reference_region_block(
    status: str,
    *,
    region_version: str | None,
    local_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The reference-region block that rides on every located response.

    Carries the two signals side by side and never merges them. A caller told
    "reference-like: 0.83" cannot tell which test fired, and the two need
    different actions: region membership is precise and silent when the model
    did not apply, while the local-block test is noisy and always answers.
    """
    if status not in REGION_STATUS_NAMES:
        status = REGION_UNKNOWN
    block: dict[str, Any] = {
        "status": status,
        "status_vocabulary": sorted(REGION_STATUS_NAMES),
        "region_version": region_version,
        "rule": REGION_RULE,
    }
    block["local_block"] = dict(local_block) if local_block else local_block_signals("", None)
    return block
