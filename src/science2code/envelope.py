"""Response envelopes for science2code, and the rules that keep them honest.

This module is stdlib only and imports nothing from the rest of the package.
That is deliberate: the refusal contract is the part of this server that most
needs to be testable without an MCP client, without a corpus, and without a PDF
extractor. Everything here is a pure function of its arguments.


WHY A BOOLEAN RETURN IS BANNED
------------------------------
The locating ladder has four outcomes. A boolean has two values. So any
projection of the ladder onto a boolean merges at least two outcomes, and every
available merge is wrong:

  * Merging T3_LOCATED into `true` lets a paraphrase, or a quote that was
    mistyped, be presented downstream as a passage the document contains. That
    is the exact error this server exists to prevent.
  * Merging T3_LOCATED into `false` tells a caller that a passage which really
    is in the document is a fabrication, and makes that indistinguishable from
    "this document has no text layer at all", which needs a different action
    from the human.

There is no third merge, so there is no correct boolean. `validate()` therefore
rejects a boolean anywhere in a response envelope, not as style but as the
enforcement of that argument.


THE FIVE ANTI-BLUR RULES
------------------------
Each is enforced by `validate()`, which every builder in this module calls
before returning, and each has a test. They are structural, not advisory,
because a written rule that depends on someone remembering it is precisely what
failed on 2026-08-28 and caused this project to exist.

  1. A refusal carries no field a caller can lift and paste as a quote. No key
     anywhere in any envelope may be named `quote`, `passage`, `text` or
     `excerpt`. A relocated passage carries `document_text` (the name says
     whose words these are), `your_text` and `char_diff` instead.
  2. The word "verbatim" occurs only inside the two VERBATIM_* outcome names.
     Never in prose, never in a reason, never in a field name.
  3. NOT_LOCATABLE carries `char_interval: null` explicitly, following Google
     LangExtract's semantics, alongside `best_score` and `scorer`, so a caller
     can say how close it got without that reading as partial success.
  4. No system-generated string may contain `supports`, `proves`,
     `demonstrates`, `confirms`, `refutes`, `shows that` or `validates`. Those
     words turn a locator into an assertion. Only the fields carrying the
     document's own words, the caller's own words, or an identifier, path or
     operating-system message this server did not write are exempt. See
     CALLER_OR_DOCUMENT_FIELDS and FOREIGN_TEXT_FIELDS for the two lists and
     the argument for each.
  5. No key anywhere may name the owner of a claim. See
     ATTRIBUTION_FIELD_NAMES and ATTRIBUTION_CEILING_SENTENCE.


WHY RULE 5 EXISTS, AND WHY IT IS STRUCTURAL RATHER THAN ADVISORY
----------------------------------------------------------------
A paper's body says "oranges are good [12]". The sentence is that paper's,
character for character, and the CLAIM is reference 12's. Measured over this
corpus, 13.9% of body sentences carry a citation marker inside the located
span and 23.6% carry one within 64 characters, so this is not an edge case: it
is roughly one located passage in four.

The obvious next move is to say whose claim it is, and that move is closed.
Trained human annotators judging citation accuracy agree at
Cohen's kappa 0.18 to 0.31, which is slight-to-fair: two experts with the same
sentence and the same cited paper in front of them reach different answers most
of the time.
The one dedicated attribution study reaches Krippendorff's alpha .654 against
a human ceiling of .806. The question is above the decidability ceiling of any
test that could run here, so a field answering it would be a number this
project cannot stand behind, printed in a shape a caller would act on.

So the sibling `markers` module reports COUNTS of markers, LABELS from closed
sets, and the SPAN of the sentence the marker sits in, and rule 5 makes the
missing field impossible to add by accident rather than merely discouraged. A
written rule that depends on someone remembering it is what failed on
2026-08-28.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

__all__ = [
    "SERVER_VERSION",
    "Outcome",
    "OUTCOME_NAMES",
    "VERBATIM_OUTCOMES",
    "REFUSAL_OUTCOMES",
    "QUOTE_FIELD_NAMES",
    "BANNED_TERMS",
    "CALLER_OR_DOCUMENT_FIELDS",
    "FOREIGN_TEXT_FIELDS",
    "ATTRIBUTION_FIELD_NAMES",
    "ATTRIBUTION_CEILING_SENTENCE",
    "CITATION_CONTEXT_CHARS",
    "CITATION_MARKER_RULE",
    "count_citation_markers",
    "citation_markers",
    "CHAR_DIFF_LEGEND",
    "INTERPRETATION_NOTICE",
    "CEILING_SENTENCE",
    "EnvelopeViolation",
    "provenance",
    "absence",
    "char_interval",
    "verbatim",
    "relocated",
    "not_locatable",
    "source_not_held",
    "source_unknown",
    "corpus_unavailable",
    "ok",
    "validate",
    "iter_strings",
]

SERVER_VERSION = "science2code/0.1.0"


class Outcome(str, enum.Enum):
    """The closed vocabulary. No response ever carries a string outside it.

    Deliberately not a boolean, and deliberately not an open string. A closed
    set means a caller can enumerate every answer this server can give, and a
    test can prove no code path invented a ninth.
    """

    #: The normalised string occurs literally in the document. No threshold was
    #: involved, so a paraphrase cannot reach this outcome by construction.
    VERBATIM_EXACT = "VERBATIM_EXACT"

    #: It occurs literally once intra-word hyphens are removed and case is
    #: folded. That is the signature of damage the PDF extractor did, not
    #: damage the quote did. Still character identity, still no threshold.
    VERBATIM_RELAXED_EXTRACTOR_DAMAGE = "VERBATIM_RELAXED_EXTRACTOR_DAMAGE"

    #: A passage above threshold was located, and the caller's string is not
    #: what the document says. Carries no field a caller can paste as a quote.
    PASSAGE_RELOCATED_QUOTE_DIFFERS = "PASSAGE_RELOCATED_QUOTE_DIFFERS"

    #: Nothing above threshold. A third outcome, not a failure, and not an
    #: accusation. `char_interval` is null and `best_score` is reported.
    NOT_LOCATABLE = "NOT_LOCATABLE"

    #: The document is known but no usable text is held for it, for example a
    #: scan with no text layer. A boundary, not an error.
    SOURCE_NOT_HELD = "SOURCE_NOT_HELD"

    #: No document with that identifier is held locally.
    SOURCE_UNKNOWN = "SOURCE_UNKNOWN"

    #: The local corpus could not be read at all.
    CORPUS_UNAVAILABLE = "CORPUS_UNAVAILABLE"

    #: A locating query ran to completion. Says nothing about how many
    #: locations were found; `hit_count` says that, and zero is a result.
    OK = "OK"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


OUTCOME_NAMES: frozenset[str] = frozenset(o.value for o in Outcome)

VERBATIM_OUTCOMES: tuple[Outcome, ...] = (
    Outcome.VERBATIM_EXACT,
    Outcome.VERBATIM_RELAXED_EXTRACTOR_DAMAGE,
)

REFUSAL_OUTCOMES: frozenset[Outcome] = frozenset(
    {
        Outcome.PASSAGE_RELOCATED_QUOTE_DIFFERS,
        Outcome.NOT_LOCATABLE,
        Outcome.SOURCE_NOT_HELD,
        Outcome.SOURCE_UNKNOWN,
        Outcome.CORPUS_UNAVAILABLE,
    }
)

# Rule 1. Field names a caller could plausibly lift and paste as if it were
# something the document said.
QUOTE_FIELD_NAMES: frozenset[str] = frozenset({"quote", "passage", "text", "excerpt"})

# Rule 4. Verbs that turn a locator into an assertion about whether a passage
# is evidence for a claim, which is not mechanically decidable.
BANNED_TERMS: tuple[str, ...] = (
    "supports",
    "proves",
    "demonstrates",
    "confirms",
    "refutes",
    "shows that",
    "validates",
)

# Rule 5. Field names that would name the owner of a claim. The substring
# "attribut" catches the whole family at once, including the ones nobody has
# thought of yet, which is the point: the rule has to hold against a future
# edit made in good faith by someone who has not read the argument above.
ATTRIBUTION_FIELD_NAMES: tuple[str, ...] = (
    "attribut",
    "claim_owner",
    "claimed_by",
    "cited_work",
    "whose_claim",
    "source_of_claim",
    "belongs_to",
)

#: Rule 5 stated in prose, carried on every block that reports a marker so the
#: counts cannot be read as the beginning of an answer to a question this
#: server does not answer. Note it names no banned verb.
ATTRIBUTION_CEILING_SENTENCE = (
    "A citation marker beside a passage means the claim may belong to another "
    "work. Which work, and whether the claim really is that work's, is not "
    "decided here and is not decidable at the accuracy that would make it "
    "worth printing: trained human annotators agree on citation accuracy at "
    "Cohen's kappa 0.18 to 0.31, and the one dedicated attribution study "
    "reaches alpha .654 against a human ceiling of .806. Read the sentence in "
    "the document."
)

# Values under these keys were not written by this server. `document_text`
# carries the document's own characters, `your_text` and `query` carry the
# caller's, and `char_diff` carries both interleaved. A real paper may well
# contain the word "supports" or the word "verbatim", and censoring the
# document would be a worse fault than any it prevents. Every other string in
# an envelope is system generated and is checked by rules 2 and 4.
CALLER_OR_DOCUMENT_FIELDS: frozenset[str] = frozenset(
    {"document_text", "your_text", "query", "char_diff"}
)

# Rules 2 and 4 police the server's OWN prose. These keys carry text the server
# did not write and cannot rewrite without lying about it: an identifier that
# came out of the user's own filename, a filesystem path, the message of an
# exception raised by the operating system, and the `field` slug of an absence,
# which embeds a document identifier so that an absence names which document it
# is about.
#
# Why this exemption exists, recorded because it looks like a hole and is not.
# A real corpus holds papers whose identifiers contain the word "demonstrates",
# and a corpus directory can be called anything at all. Checking those strings
# meant that every answer about such a paper raised EnvelopeViolation instead of
# returning a refusal, so a legitimate paper became permanently unanswerable and
# a caller could crash the server by naming one. Neither the identifier nor the
# path is an assertion this server made about a passage, which is the only thing
# rule 4 exists to prevent, so they are exempted rather than censored. The
# server's own prose lives in `reason`, and `reason` stays fully checked: no
# foreign text may be interpolated into it.
FOREIGN_TEXT_FIELDS: frozenset[str] = frozenset(
    {"paper_id", "also_occurs_in", "field", "detail"}
)

#: What `char_diff` actually looks like, described once so the legend and the
#: producer cannot drift apart. The producer is `_char_diff` in the anchor
#: module, and a test pins this legend against its real output. An earlier
#: legend described a wholly different notation, one with bracket markers that
#: the producer has never emitted, so a caller who read the legend and then
#: looked for those markers would have found none and could have concluded the
#: two texts matched.
CHAR_DIFF_LEGEND = (
    "char_diff lists one entry per difference. Each entry gives the offset in "
    "your_text where the difference starts and the characters just before it. "
    "Under that entry, a line whose first character is a minus sign holds "
    "characters your_text has that the document does not have at that point, "
    "and a line whose first character is a plus sign holds characters the "
    "document has that your_text does not. Where the two are identical after "
    "normalisation the listing says so in words instead."
)

#: How far either side of a located span the citation-marker test looks.
#: Measured on a 45-document corpus of scientific PDFs: over 449 located body
#: sentences, a marker sat inside the located characters in 15.8% of cases and
#: inside or within 64 characters in 26.1%. A window of 30 reported 22.9% and
#: one of 120 reported 30.1%. 64 was chosen because it is the smallest window
#: that reaches back past a reference-list number and the author initials that
#: follow it, which is the case where the marker is furthest from the span and
#: matters most.
CITATION_CONTEXT_CHARS = 64

# A bracketed reference number: [12], [3, 4], [7-9]. Dashes are already folded
# to the ASCII hyphen by the normaliser before this ever runs.
_BRACKET_MARKER = re.compile(r"\[\s*\d{1,3}(?:\s*[,;-]\s*\d{1,3})*\s*\]")

# A parenthesised year, which is what an author-year citation ends with:
# (2019), (Smith et al., 2019), (Smith and Jones 2019a). The 60-character
# ceiling keeps it from spanning a whole parenthetical clause that merely ends
# in a date.
_PAREN_YEAR_MARKER = re.compile(r"\([^()]{0,60}\b(?:1[5-9]|20)\d{2}[a-z]?\s*\)")

#: Stated once, and carried on every response that reports a location, so that
#: the counts cannot be read as a judgement about what is being cited. Note it
#: names no banned verb: this is a character test, and the sentence describing
#: it must not assert anything the test does not decide.
CITATION_MARKER_RULE = (
    "A citation marker is a bracketed reference number such as [12] or [3, 4], "
    "or a parenthesised year such as (2019) or (Smith et al., 2019). Two "
    "counts are reported: markers among the located characters, and markers in "
    "the %d characters either side of them. A sentence carrying one attributes "
    "its claim to another work, so the words are this document's and the claim "
    "may not be. This is a character test over the located span and its "
    "immediate context. A count of zero means the test found none, not that "
    "the claim originates here, and this server does not resolve a marker to "
    "the work it points at." % CITATION_CONTEXT_CHARS
)

_VERBATIM_OUTCOME_NAMES: tuple[str, ...] = tuple(o.value for o in VERBATIM_OUTCOMES)

#: The ceiling, stated once and reused, so a tool description and a response
#: cannot drift apart. Note that it names no banned verb: the sentence that
#: denies an assertion must not smuggle one in.
CEILING_SENTENCE = (
    "It cannot judge whether a passage is evidence for a claim, and it cannot "
    "detect a relevant work that was never read. Both are human judgements."
)

INTERPRETATION_NOTICE = (
    "science2code reports whether an exact string occurs at a given location in "
    "a locally held document. " + CEILING_SENTENCE + " Only the VERBATIM_EXACT "
    "and VERBATIM_RELAXED_EXTRACTOR_DAMAGE outcomes assert character identity "
    "with the document."
)


class EnvelopeViolation(RuntimeError):
    """A response broke the refusal contract and must not leave the server.

    Raised rather than logged. An envelope that violates the contract is worse
    than no answer, because the whole point is that a caller can trust the
    shape without reading the code that produced it.
    """


# ---------------------------------------------------------------------------
# small pieces
# ---------------------------------------------------------------------------


def provenance(
    normaliser: str,
    verifier: str,
    scorer: str,
    server: str = SERVER_VERSION,
) -> dict[str, str]:
    """Name every component whose version changes what an offset means.

    A stored offset is valid only while the normaliser is unchanged, so the
    normaliser version travels with every answer rather than being looked up.
    """
    return {
        "normaliser": str(normaliser),
        "verifier": str(verifier),
        "scorer": str(scorer),
        "server": str(server),
    }


def absence(field: str, reason: str, detail: str | None = None) -> dict[str, str]:
    """One entry for `not_available`: a thing that is missing, and why.

    Nothing is silently omitted. If a field is absent from a response there is
    a line here naming it, so a caller never has to guess whether an absence
    means "no" or "not asked".

    `reason` is this server's own prose and is held to rules 2 and 4, so no
    identifier, path or operating-system error message may be interpolated into
    it. Anything of that kind goes in `detail`, which is exempt because this
    server did not write it. See FOREIGN_TEXT_FIELDS.
    """
    entry = {"field": str(field), "reason": str(reason)}
    if detail is not None:
        entry["detail"] = str(detail)
    return entry


def count_citation_markers(text: str) -> int:
    """How many citation markers occur in `text`. See CITATION_MARKER_RULE.

    Pure string matching, no judgement. What a marker is lives here, next to
    the sentence that describes it to the caller, so the description and the
    test cannot drift apart.

    THIS IS THE LENIENT COUNTER AND IS NO LONGER WHAT THE SERVER CALLS. The
    author-year half of it accepts any parenthesis that ends in a year, which
    measures 77.0% precision, and its worst failure mode is the journal
    running head "Scientific Data |   (2026) 13:936", which every page of a
    modern journal PDF carries. The sibling `markers` module holds the measured
    replacement: bracketed at 72/72 precision, strict author-year at 95.6% at
    1.000 recall, and a label for the citation style this extractor cannot see
    at all. This function stays because an envelope built by an earlier version
    of this package used it and its numbers must keep meaning what they meant.
    """
    if not isinstance(text, str) or not text:
        return 0
    return len(_BRACKET_MARKER.findall(text)) + len(_PAREN_YEAR_MARKER.findall(text))


def citation_markers(located: str, before: str = "", after: str = "") -> dict[str, Any]:
    """The citation-marker block that rides on every located passage.

    THE PROBLEM THIS EXISTS FOR. A paper says "eating oranges is good [12]".
    That whole sentence is in the paper, character for character, so it reaches
    a character-identity outcome and every field in the response is true. A
    caller can nonetheless read the answer as "this paper says eating oranges
    is good", when the paper said someone else did. Measured on a 45-document
    corpus, 15.8% of located body sentences carry a marker inside the located
    characters and 26.1% carry one inside or within CITATION_CONTEXT_CHARS.

    The worst case is the one the caller cannot see: a quote that stops one
    character before the marker. The located text then looks clean and the
    "[12]" that owns the claim is just outside it. That is why the count of
    markers in the surrounding characters is reported separately rather than
    folded into one number: a marker beside the span is a different fact from a
    marker inside it, and merging them would hide the case that is invisible.

    This server does NOT resolve a marker to the work it points at. Reporting
    the count is a character test; deciding whose claim it is remains the
    caller's, which is the same boundary the whole server keeps.
    """
    return {
        "in_located_characters": count_citation_markers(located),
        "in_surrounding_characters": count_citation_markers(before)
        + count_citation_markers(after),
        "context_chars": CITATION_CONTEXT_CHARS,
        "rule": CITATION_MARKER_RULE,
    }


def char_interval(start: int | None, end: int | None) -> dict[str, Any] | None:
    """A LangExtract-style interval, or None when nothing was located.

    `basis` is carried because these offsets index the normalised document,
    not the bytes on disk, and a caller that assumed otherwise would seek to
    the wrong place.
    """
    if start is None or end is None:
        return None
    return {
        "start_pos": int(start),
        "end_pos": int(end),
        "basis": "normalised_document",
    }


def _base(
    outcome: Outcome,
    provenance_block: Mapping[str, str],
    not_available: Sequence[Mapping[str, str]] | None,
) -> dict[str, Any]:
    return {
        "outcome": outcome.value,
        "interpretation_notice": INTERPRETATION_NOTICE,
        "provenance": dict(provenance_block),
        "not_available": [dict(a) for a in (not_available or ())],
    }


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def verbatim(
    outcome: Outcome,
    *,
    paper_id: str,
    document_text: str,
    start_pos: int,
    end_pos: int,
    score: float | None,
    scorer: str,
    provenance_block: Mapping[str, str],
    page: int | None = None,
    citation_marker_block: Mapping[str, Any] | None = None,
    reference_region_block: Mapping[str, Any] | None = None,
    not_available: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """A character-identity outcome: the document's own characters, located."""
    if outcome not in VERBATIM_OUTCOMES:
        raise EnvelopeViolation(
            "verbatim() accepts only the two character-identity outcomes, got %r"
            % (outcome,)
        )
    payload = _base(outcome, provenance_block, not_available)
    payload.update(
        {
            "paper_id": str(paper_id),
            "document_text": document_text,
            "char_interval": char_interval(start_pos, end_pos),
            "score": score,
            "scorer": str(scorer),
            "page": page,
        }
    )
    if citation_marker_block is not None:
        payload["citation_markers"] = dict(citation_marker_block)
    if reference_region_block is not None:
        payload["reference_region"] = dict(reference_region_block)
    return validate(payload)


def relocated(
    *,
    paper_id: str,
    document_text: str,
    your_text: str,
    char_diff: str | None,
    start_pos: int,
    end_pos: int,
    best_score: float | None,
    scorer: str,
    provenance_block: Mapping[str, str],
    page: int | None = None,
    citation_marker_block: Mapping[str, Any] | None = None,
    reference_region_block: Mapping[str, Any] | None = None,
    not_available: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """A passage was located and the caller's string is not what it says.

    The three fields are named so that no reading of them produces a quote:
    `document_text` is whose words these are, `your_text` is whose words those
    were, and `char_diff` is the distance between them. There is no field here
    that a caller can lift.
    """
    payload = _base(Outcome.PASSAGE_RELOCATED_QUOTE_DIFFERS, provenance_block, not_available)
    payload.update(
        {
            "paper_id": str(paper_id),
            "document_text": document_text,
            "your_text": your_text,
            "char_diff": char_diff,
            "char_diff_legend": CHAR_DIFF_LEGEND,
            "char_interval": char_interval(start_pos, end_pos),
            "best_score": best_score,
            "scorer": str(scorer),
            "page": page,
        }
    )
    if citation_marker_block is not None:
        payload["citation_markers"] = dict(citation_marker_block)
    if reference_region_block is not None:
        payload["reference_region"] = dict(reference_region_block)
    return validate(payload)


def not_locatable(
    *,
    your_text: str,
    best_score: float | None,
    scorer: str,
    provenance_block: Mapping[str, str],
    paper_id: str | None = None,
    not_available: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Nothing above threshold anywhere that was searched.

    `char_interval` is present and explicitly null rather than omitted, which
    is LangExtract's convention: ungroundable is a distinct, reportable state,
    not a missing field and not a failure.
    """
    payload = _base(Outcome.NOT_LOCATABLE, provenance_block, not_available)
    payload.update(
        {
            "paper_id": paper_id,
            "your_text": your_text,
            "char_interval": None,
            "best_score": best_score,
            "scorer": str(scorer),
        }
    )
    return validate(payload)


def source_not_held(
    *,
    paper_id: str | None,
    reason: str,
    provenance_block: Mapping[str, str],
    your_text: str | None = None,
    not_available: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """The document is known locally but no usable characters are held for it."""
    payload = _base(Outcome.SOURCE_NOT_HELD, provenance_block, not_available)
    payload.update({"paper_id": paper_id, "reason": str(reason), "char_interval": None})
    if your_text is not None:
        payload["your_text"] = your_text
    return validate(payload)


def source_unknown(
    *,
    paper_id: Any,
    reason: str,
    provenance_block: Mapping[str, str],
    your_text: str | None = None,
    not_available: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """No document with that identifier is held locally."""
    payload = _base(Outcome.SOURCE_UNKNOWN, provenance_block, not_available)
    payload.update({"paper_id": paper_id, "reason": str(reason), "char_interval": None})
    if your_text is not None:
        payload["your_text"] = your_text
    return validate(payload)


def corpus_unavailable(
    *,
    reason: str,
    provenance_block: Mapping[str, str],
    your_text: str | None = None,
    not_available: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """The local corpus could not be read, so no question about it was answered."""
    payload = _base(Outcome.CORPUS_UNAVAILABLE, provenance_block, not_available)
    payload.update({"reason": str(reason), "char_interval": None})
    if your_text is not None:
        payload["your_text"] = your_text
    return validate(payload)


def ok(
    *,
    query: str,
    hits: Sequence[Mapping[str, Any]],
    searched: Sequence[str],
    provenance_block: Mapping[str, str],
    not_available: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """A locating query ran to completion.

    `hit_count` of zero is a result, not an error: the phrase does not occur in
    what is held. Each hit carries its own outcome from the same closed
    vocabulary, so a hit can never be read as stronger than its own tier.
    """
    payload = _base(Outcome.OK, provenance_block, not_available)
    payload.update(
        {
            "query": query,
            "hit_count": len(hits),
            "hits": [dict(h) for h in hits],
            "searched_document_count": len(searched),
        }
    )
    return validate(payload)


# ---------------------------------------------------------------------------
# enforcement
# ---------------------------------------------------------------------------


def _walk(
    node: Any, key: str | None = None, path: str = "$"
) -> Iterator[tuple[str, str | None, Any]]:
    """Yield (path, immediate key, value) for every node in the envelope."""
    yield path, key, node
    if isinstance(node, Mapping):
        for k, v in node.items():
            yield from _walk(v, str(k), "%s.%s" % (path, k))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from _walk(v, key, "%s[%d]" % (path, i))


def iter_strings(payload: Mapping[str, Any], system_only: bool = True) -> Iterator[tuple[str, str]]:
    """Yield (path, string) for strings in an envelope.

    With `system_only`, values written by the caller, copied out of the
    document, or carried in from a filename, a path or an operating-system
    error are skipped, and every remaining string is one this server wrote.
    """
    for path, key, value in _walk(payload):
        if not isinstance(value, str):
            continue
        if system_only and (key in CALLER_OR_DOCUMENT_FIELDS
                            or key in FOREIGN_TEXT_FIELDS):
            continue
        yield path, value


def _verbatim_uses_are_outcome_names(text: str) -> bool:
    """Rule 2. Every occurrence of the word must be inside an outcome name.

    The occurrence must be the WHOLE outcome name and not merely start with
    one, or the rule reads a prefix as a licence: "VERBATIM_EXACTLY what the
    paper says" begins with VERBATIM_EXACT and is prose, which is the one thing
    the rule forbids. The character after the name has to be something other
    than a letter, a digit or an underscore, and the end of the string counts.
    """
    lowered = text.lower()
    i = lowered.find("verbatim")
    while i != -1:
        matched = max(
            (len(name) for name in _VERBATIM_OUTCOME_NAMES
             if text.startswith(name, i)),
            default=0,
        )
        if not matched:
            return False
        after = i + matched
        if after < len(text) and (text[after].isalnum() or text[after] == "_"):
            return False
        i = lowered.find("verbatim", i + len("verbatim"))
    return True


def validate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce the whole contract, or raise. Returns the payload for chaining.

    Every builder above ends by calling this, so there is no path from this
    module to a caller that skips it.
    """
    if not isinstance(payload, Mapping):
        raise EnvelopeViolation("an envelope must be a mapping, got %s" % type(payload).__name__)

    outcome = payload.get("outcome")
    if outcome not in OUTCOME_NAMES:
        raise EnvelopeViolation(
            "outcome %r is outside the closed vocabulary %s"
            % (outcome, sorted(OUTCOME_NAMES))
        )

    if payload.get("interpretation_notice") != INTERPRETATION_NOTICE:
        raise EnvelopeViolation("every response carries the constant interpretation_notice")

    prov = payload.get("provenance")
    _PROVENANCE_KEYS = {"normaliser", "verifier", "scorer", "server"}
    if not isinstance(prov, Mapping) or not _PROVENANCE_KEYS <= set(prov):
        raise EnvelopeViolation(
            "provenance must name the normaliser, verifier, scorer and server"
        )

    if not isinstance(payload.get("not_available"), list):
        raise EnvelopeViolation("not_available must be a list, empty if nothing is absent")

    for path, key, value in _walk(payload):
        # The boolean ban. See the module docstring for the argument.
        if isinstance(value, bool):
            raise EnvelopeViolation(
                "a boolean at %s: the ladder has four outcomes and every "
                "projection onto two values merges outcomes that must stay "
                "apart" % path
            )
        if key is not None:
            # A field NAME is always system generated, so rules 2 and 4 apply
            # to it whatever the value under it is exempt from.
            if not _verbatim_uses_are_outcome_names(key):
                raise EnvelopeViolation(
                    "the field name %r at %s uses the word outside the two "
                    "VERBATIM_* outcome names" % (key, path)
                )
            for term in BANNED_TERMS:
                if term in key.lower():
                    raise EnvelopeViolation(
                        "the field name %r at %s contains the banned term %r"
                        % (key, path, term)
                    )
        if key is not None:
            # Rule 5. A field naming whose claim a passage carries would be an
            # answer to a question that sits above this server's decidability
            # ceiling, and it would look exactly like the fields around it that
            # are mechanically true. See ATTRIBUTION_CEILING_SENTENCE.
            lowered_key = key.lower()
            for banned_key in ATTRIBUTION_FIELD_NAMES:
                if banned_key in lowered_key:
                    raise EnvelopeViolation(
                        "field %r at %s names the owner of a claim. Human "
                        "annotators agree on that question at Cohen's kappa "
                        "0.18 to 0.31, so this server reports the marker, the "
                        "region and the sentence, and stops there"
                        % (key, path)
                    )
        if key is not None and key in QUOTE_FIELD_NAMES:
            # Rule 1, enforced on every envelope rather than only on refusals,
            # because a field a caller can lift should not exist at all.
            raise EnvelopeViolation(
                "field %r at %s is named so a caller could paste it as a quote"
                % (key, path)
            )
        # A nested outcome, for example inside a hit, is held to the same
        # closed vocabulary as the top-level one.
        if key == "outcome" and isinstance(value, str) and value not in OUTCOME_NAMES:
            raise EnvelopeViolation(
                "nested outcome %r at %s is outside the vocabulary" % (value, path)
            )

    for path, text in iter_strings(payload, system_only=True):
        if not _verbatim_uses_are_outcome_names(text):
            raise EnvelopeViolation(
                "the word appears at %s outside the two VERBATIM_* outcome names" % path
            )
        lowered = text.lower()
        for term in BANNED_TERMS:
            if term in lowered:
                raise EnvelopeViolation(
                    "banned term %r at %s: this server locates strings, it does "
                    "not assert what a passage does" % (term, path)
                )

    if outcome == Outcome.NOT_LOCATABLE.value:
        # Rule 3, checked positively: the key must be present and explicitly
        # null, so a caller can tell "nothing was located" from "not reported".
        if "char_interval" not in payload or payload["char_interval"] is not None:
            raise EnvelopeViolation("NOT_LOCATABLE must carry char_interval explicitly null")
        for required in ("best_score", "scorer"):
            if required not in payload:
                raise EnvelopeViolation("NOT_LOCATABLE must carry %s" % required)

    return dict(payload)
