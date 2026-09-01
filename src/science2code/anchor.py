"""The anchoring ladder: locate a quoted passage in a source document.

Four tiers, in order. Only the first two ever assert "verbatim".

    T1_EXACT          the normalised quote occurs literally in the normalised
                      document.
    T2_RELAXED        it occurs literally in the MATCH FORM (intra-word hyphen
                      deleted, casefolded). Still character identity, still no
                      threshold.
    T3_LOCATED        the best fuzzy window scores >= t_locate. The verdict is
                      NOT "verbatim". It is "passage relocated, quote text
                      differs", and it always carries a character diff.
    T4_NOT_LOCATABLE  nothing scores at or above t_locate.

Why the ladder has this shape
-----------------------------
T1 and T2 are character-identity tests with NO similarity threshold. Nothing
that is merely *similar* can reach them. That is what makes the false-positive
rate for "a paraphrase accepted as a verbatim quote" 0% BY CONSTRUCTION rather
than by tuning a number that a later corpus could move.

The measurement behind that choice, on 400 exactly-anchored quotes perturbed
into roughly 3600 positive and negative cases:

    fabricated attribution (a passage from a different paper)   <= 0.536
    genuinely damaged true quote (word dropped, typo, digit)    >= 0.819
    margin                                                       0.283  (difflib)
    margin                                                       0.202  (rapidfuzz)

Those figures, and every other number quoted anywhere in this module, were
measured on a private 44-document corpus of scientific PDFs which cannot be
redistributed, because the papers are paywalled or licence-limited. A reader
cannot re-run them against the same inputs, and the run itself left no script
or log, so they cannot be re-derived here either. They are kept because they
are the evidence the threshold was chosen on, not because they are
reproducible. The BEHAVIOUR they justify is pinned by the test suite, which
uses short inline fixtures and opens no file.

So a fabricated attribution and a damaged true quote ARE separable, and t_locate
separates them. A synonym paraphrase is NOT separable from a damaged true quote
by any similarity score: the two distributions overlap. The system therefore
never tries. A paraphrase can at best reach T3, whose verdict already says the
quote text differs from the document.

One fold is not allowed to reach T1 or T2, and it is worth saying why the
exception exists at all. The normaliser applies NFKC, which is what makes a
formula quotable, and NFKC discards position: it folds "10\u2076" to "106" and
"10\u207b\u00b3" to "10-3". A quote and a document that agree only after that
fold are not the same string, they are different numbers, and an identity
verdict over them would be exactly the false attribution this ladder exists to
refuse. So the superscript, subscript and fraction characters of the quote are
compared against those of the located document span before either identity tier
may be returned, and a span that only matched because of the fold comes back as
T3_LOCATED with a character diff over the RAW forms. That check can only ever
demote, never promote, so it cannot open a path to a verdict the ladder would
not otherwise have reached.

T4 is named NOT_LOCATABLE, not MISS, deliberately. It is a third outcome,
distinct from both success and invalidation, following the `char_interval = None`
semantics of Google LangExtract. A paper with no text layer (an image-only scan)
lands in T4 for every quote and must never be read as "the quote failed".

Offsets
-------
`offset_norm` and `length_norm` index the NORMALISED haystack, which is what
`science2code.normalise.normalise` produces. To render a normalised offset as a span in
the original file, use the index map that `normalise` returns alongside the
text. Offsets are a cache. `quote_raw` is the primary key. See `anchor_record`.

Scorers
-------
The default is `difflib.SequenceMatcher.ratio` from the standard library. It
separates the classes better than rapidfuzz (0.283 against 0.202) and is about
30% slower, which does not matter at 2.5 ms per lookup. `scorer="rapidfuzz"` is
available behind an optional import. The two scales are NOT interchangeable:
paraphrase means differ by about 0.07 between them, so the scorer name is stored
with every result and every record.
"""

from __future__ import annotations

import dataclasses
import difflib
import enum
import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Any

from .normalise import (
    MATCHFORM_VERSION,
    NORMALISER_FINGERPRINT,
    NORMALISER_VERSION,
    fingerprint_file,
    match_form,
    normalise,
)

try:  # optional, never required
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
except ImportError:  # pragma: no cover - the default path is pure stdlib
    _rapidfuzz_fuzz = None

__all__ = [
    "Tier",
    "Anchor",
    "PreparedText",
    "locate",
    "prepare",
    "anchor_record",
    "record_status",
    "record_is_stale",
    "reanchor_record",
    "VERIFIER_VERSION",
    "VERIFIER_FINGERPRINT",
    "T_LOCATE_DEFAULT",
    "T_LOCATE_EQUIVALENT_RANGE",
    "SCORERS",
    "STATUS_FRESH",
    "STATUS_STALE",
    "W3C_ANNOTATION_CONTEXT",
]

#: Bumped from 1.0.0 when the ladder stopped letting a lossy NFKC fold reach a
#: character-identity tier. A record stored under 1.0.0 may name a tier this
#: code would no longer give, so it reads STALE and re-anchors from its raw
#: quote, which is what the version contract is for. STALE never means the
#: citation was wrong.
VERIFIER_VERSION = "verify/1.1.0"

#: A digest of THIS module's behaviour, stored alongside VERIFIER_VERSION on
#: every anchor record. Same backstop, same argument, same failure mode as
#: NORMALISER_FINGERPRINT: see THE FINGERPRINT BACKSTOP in
#: `science2code.normalise`. The version literal above is hand maintained, and
#: a hand maintained literal fails silently. If the ladder's behaviour moves
#: (a tier condition, the seed-and-extend parameters, the scorer table, the
#: diff format) and nobody edits the literal, every record stored afterwards
#: claims a verifier it does not have and reads FRESH for ever, so the
#: re-anchoring the whole design rests on quietly stops firing.
#:
#: The fingerprint moves whenever the code moves and holds still for comments,
#: docstrings and reformatting. A mismatch means STALE, which means re-anchor
#: from `quote_raw`. It never means the citation was wrong.
VERIFIER_FINGERPRINT = fingerprint_file(globals().get("__file__"))

#: Default fuzzy locating threshold. On the perturbation set the measurement
#: was made on, every value in T_LOCATE_EQUIVALENT_RANGE gives the same
#: outcomes: no fabricated attribution reaches 0.55 and no damaged true quote
#: falls below 0.72. 0.65 sits near the midpoint of that measured margin.
#:
#: THE RANGE IS NOT AN EQUIVALENCE FOR ARBITRARY STRINGS, and an earlier
#: wording that said so was too strong. The measurement covered quotes
#: perturbed from real anchors and passages lifted from a different paper. A
#: string that is in no held document at all is neither, and it can land in
#: the gap: on a 45-document corpus, the title of a paper the corpus does not
#: hold scored 0.610 against an unrelated window, which is NOT_LOCATABLE at
#: 0.65 and PASSAGE_RELOCATED_QUOTE_DIFFERS at 0.55. Lowering the threshold
#: therefore does change answers, and it changes them in the direction that
#: names a location for a string the corpus does not contain.
T_LOCATE_DEFAULT = 0.65
T_LOCATE_EQUIVALENT_RANGE = (0.55, 0.72)

SCORERS = ("difflib", "rapidfuzz")

STATUS_FRESH = "FRESH"
STATUS_STALE = "STALE"

W3C_ANNOTATION_CONTEXT = "http://www.w3.org/ns/anno.jsonld"

_WORD = re.compile(r"[A-Za-z0-9]+")

# Seed-and-extend tuning. These are search parameters, not verdict parameters:
# widening them can only find a better window, never change T1 or T2.
_SEED_WORDS_FALLBACK = (4, 3, 2, 1)
_MAX_SEEDS = 10
_MAX_CANDIDATES = 300
_SMALL_HAYSTACK = 20_000
_REFINE_TOP = 5
#: Upper bound on windows-scored times needle-length-squared in _best_window.
#: At about 4e8 a worst-case lookup stays well under a second. Above a needle
#: of roughly 1,150 characters this reduces the window count below
#: _MAX_CANDIDATES; below it the cap is never reached and behaviour is
#: unchanged from before the bound existed.
_WINDOW_SCORE_BUDGET = 400_000_000
#: Most seed n-grams whose document frequency is counted. Counting is
#: O(needle-words times haystack), so an unbounded needle drives it; a normal
#: quote has far fewer grams than this and is unaffected. Sampled evenly across
#: the needle so the seeds still span the whole quote.
_MAX_GRAMS = 256

_DIFF_CONTEXT = 24
_DIFF_MAX_SEGMENT = 60
_DIFF_MAX_OPCODES = 20


class Tier(enum.Enum):
    """The rung of the ladder a quote reached.

    Deliberately not a boolean. See `locate`.
    """

    T1_EXACT = "T1_EXACT"
    T2_RELAXED = "T2_RELAXED"
    T3_LOCATED = "T3_LOCATED"
    T4_NOT_LOCATABLE = "T4_NOT_LOCATABLE"

    @property
    def is_verbatim(self) -> bool:
        """True only for the two character-identity tiers."""
        return self in (Tier.T1_EXACT, Tier.T2_RELAXED)


@dataclasses.dataclass(frozen=True)
class Anchor:
    """The outcome of one lookup.

    tier         which rung was reached.
    offset_norm  start of the located span in the normalised haystack, or None.
    length_norm  length of that span in the normalised haystack, or None. For
                 T2 this is the length of the DOCUMENT span, which can differ
                 from the length of the quote, because the match form deletes
                 intra-word hyphens on both sides.
    score        1.0 for T1 and T2 (identity), the window score for T3, and the
                 best score seen for T4 so a caller can report how close it got.
    diff         a character diff of quote against document. T3 only.
    scorer       the name of the scorer that produced `score`. Stored because
                 difflib and rapidfuzz scales are not interchangeable.
    """

    tier: Tier
    offset_norm: int | None
    length_norm: int | None
    score: float | None
    diff: str | None
    scorer: str

    @property
    def is_verbatim(self) -> bool:
        """True only for T1 and T2. Never infer this from `score`."""
        return self.tier.is_verbatim


@dataclasses.dataclass(frozen=True)
class PreparedText:
    """A haystack with its normalisation cached.

    Anchoring many quotes into one document should normalise it once. `locate`
    accepts either a raw string or one of these.
    """

    raw: str
    norm: str
    norm_index_map: Sequence[int]
    match: str
    match_index_map: Sequence[int]


def prepare(text: str) -> PreparedText:
    """Normalise a haystack once, for repeated lookups."""
    norm, index_map = normalise(text)
    match = match_form(norm)
    return PreparedText(
        raw=text,
        norm=norm,
        norm_index_map=index_map,
        match=match,
        match_index_map=_match_index_map(norm, match),
    )


# --------------------------------------------------------------------------
# scorers
# --------------------------------------------------------------------------


def _difflib_ratio(a: str, b: str) -> float:
    # autojunk is left at its default. The measured margin of 0.283 was
    # obtained with the default heuristic in place; turning it off would
    # change the scale and invalidate T_LOCATE_DEFAULT.
    return difflib.SequenceMatcher(None, a, b).ratio()


def _rapidfuzz_ratio(a: str, b: str) -> float:
    return _rapidfuzz_fuzz.ratio(a, b) / 100.0


def _resolve_scorer(name: str) -> Callable[[str, str], float]:
    if name == "difflib":
        return _difflib_ratio
    if name == "rapidfuzz":
        if _rapidfuzz_fuzz is None:
            raise ValueError(
                "scorer='rapidfuzz' requested but rapidfuzz is not installed. "
                "Install it, or use the stdlib default scorer='difflib'."
            )
        return _rapidfuzz_ratio
    raise ValueError(
        "unknown scorer %r, expected one of %s" % (name, ", ".join(SCORERS))
    )


# --------------------------------------------------------------------------
# match form index mapping
# --------------------------------------------------------------------------


def _match_index_map(norm: str, match: str) -> list[int]:
    """Map each match-form offset back to an offset in the normalised text.

    `match_form` only deletes characters (the intra-word hyphen) and casefolds
    them, so the mapping is monotonic and can be recovered by a single forward
    walk. `science2code.normalise.match_form` returns only the folded string,
    so this module recovers the map rather than asking for one.

    The walk is driven by the SOURCE character, not the folded one, because a
    casefold can expand: U+00DF folds to "ss", so one normalised character can
    own two match-form offsets. Walking the folded side one character at a time
    desynchronises on the first such character and never recovers.

    The normalise module also offers `match_form_with_map`, which returns this
    map directly. This module deliberately depends only on the minimal agreed
    contract (`match_form(text) -> str`) and reconstructs the map, so a change
    to that helper cannot move an offset here. The two were checked against
    each other over 526k offsets across six extracted papers (from the same
    private corpus as every other measurement here) and agreed on every one,
    and `_match_span` verifies the recovered span against the document before
    any T2 verdict is returned, so a drift would fail closed rather than
    mislocate.
    """
    index_map: list[int] = [-1] * len(match)
    i = 0
    j = 0
    n = len(norm)
    m = len(match)
    while i < n and j < m:
        folded = norm[i].casefold()
        if folded and match.startswith(folded, j):
            for _ in range(len(folded)):
                index_map[j] = i
                j += 1
        # else: match_form deleted this character (the intra-word hyphen).
        i += 1
    if j < m:
        # Alignment ran out, so match_form did something beyond delete and
        # casefold. Report no map rather than a wrong one: a wrong offset on a
        # T2 verdict is worse than no T2 verdict at all.
        return []
    return index_map


def _match_span(
    prepared: PreparedText, quote_match: str, search_from: int = 0
) -> tuple[int, int] | None:
    """Span in the normalised haystack of a literal match-form hit, or None.

    `search_from` is an offset into the MATCH form, so a caller can walk
    successive occurrences: pass the previous hit's match-form position plus
    one to find the next.
    """
    at = prepared.match.find(quote_match, search_from)
    if at < 0:
        return None
    index_map = prepared.match_index_map
    if not index_map or at + len(quote_match) > len(index_map):
        return None
    start = index_map[at]
    end = index_map[at + len(quote_match) - 1] + 1
    if match_form(prepared.norm[start:end]) == quote_match:
        return start, end, at
    # Repair a small drift rather than trusting the walk blindly.
    for delta_start in (0, -1, 1, -2, 2, -3, 3, -4, 4):
        for delta_end in (0, 1, -1, 2, -2, 3, -3, 4, -4):
            a = start + delta_start
            b = end + delta_end
            if 0 <= a < b <= len(prepared.norm):
                if match_form(prepared.norm[a:b]) == quote_match:
                    return a, b, at
    # Fail closed. A T2 verdict asserts "verbatim", so it may not be returned
    # with an offset that has not been verified against the document text. The
    # lookup falls through to T3, which reports a score and a diff.
    return None


def _casefold_hazard_next_to_digit(quote_norm: str, doc_span: str) -> bool:
    """A case difference on a letter touching a digit changes a unit prefix.

    T2 forgives a case change, because the extractor sometimes recases a
    heading or a sentence start, and case there is presentation. But a letter
    directly beside a digit is a unit prefix, where case is meaning: 10 mW is a
    billionth of 10 MW, 500 Mb is an eighth of 500 MB. When the quote and the
    document span are equal only under casefold and differ at such a position,
    a T2 identity verdict would assert two different quantities are the same
    characters, which is the false verbatim this ladder exists to refuse.

    Scoped to the equal-length, pure-case case (no intra-word hyphen in play),
    which is where a unit prefix lives; a hyphen is never part of a unit.

    Known residual, on the record: a unit separated from its number by a whole
    WORD, "5 base mW" against "5 base MW", is not caught. That is not valid unit
    notation (a unit attaches to its number directly or across one space), and
    catching it would demote ordinary prose like "5 more Studies" against
    "5 more studies", so it is left open deliberately rather than traded for
    false demotions across the corpus.
    """
    if len(quote_norm) != len(doc_span):
        return False
    if quote_norm.casefold() != doc_span.casefold():
        return False
    s = quote_norm
    n = len(s)
    for i, (q, d) in enumerate(zip(s, doc_span, strict=False)):
        if q == d:
            continue
        # A digit ANYWHERE in the recased letter's own maximal alphanumeric
        # token: "10mW", "12mS", "5Mbps", "H2O". The letter carries a unit or a
        # formula bound to that number, so case is the quantity, not
        # presentation. A letter between the digit and the recased one used to
        # let this through, so the whole token is checked, not just the
        # neighbours.
        a = i
        while a > 0 and s[a - 1].isalnum():
            a -= 1
        b = i
        while b < n and s[b].isalnum():
            b += 1
        if any(c.isdigit() for c in s[a:b]):
            return True
        # The recased letter's whitespace-delimited token follows a number
        # token across a single run of spaces: "10 mW", "500 MB".
        j = i
        while j > 0 and not s[j - 1].isspace():
            j -= 1
        k = j
        while k > 0 and s[k - 1].isspace():
            k -= 1
        if k > 0 and s[k - 1].isdigit():
            return True
    return False


# --------------------------------------------------------------------------
# seed and extend
# --------------------------------------------------------------------------


def _seed_offsets(haystack_lower: str, needle_lower: str) -> list[int]:
    """Candidate window starts, from the rarest word n-grams of the needle.

    Not a full scan. Word n-grams of the needle are located with `str.find`,
    ranked by how often they occur in the haystack so the rarest go first, and
    each hit is turned back into a candidate window start by subtracting the
    n-gram's offset within the needle. About 2.5 ms per lookup against a 90 kB
    document.
    """
    spans = [(m.start(), m.end()) for m in _WORD.finditer(needle_lower)]
    for width in _SEED_WORDS_FALLBACK:
        if len(spans) < width:
            continue
        grams = [
            (spans[i][0], needle_lower[spans[i][0] : spans[i + width - 1][1]])
            for i in range(len(spans) - width + 1)
        ]
        # Counting every gram across the whole haystack is O(needle-words times
        # haystack), the real cost lever on a long argument: a 5,000-character
        # needle is ~1,000 words, each counted across the document. We only need
        # a handful of RARE grams to seed, so an evenly spaced sample bounds the
        # count calls while still spanning the whole quote. A normal quote has
        # far fewer grams than the cap and is untouched.
        if len(grams) > _MAX_GRAMS:
            step = len(grams) / _MAX_GRAMS
            grams = [grams[int(i * step)] for i in range(_MAX_GRAMS)]
        scored = [
            (haystack_lower.count(gram), offset, gram)
            for offset, gram in grams
        ]
        scored = [row for row in scored if row[0]]
        if not scored:
            continue
        scored.sort()
        candidates: set[int] = set()
        for _count, offset, gram in scored[:_MAX_SEEDS]:
            at = haystack_lower.find(gram)
            while at >= 0 and len(candidates) < _MAX_CANDIDATES:
                candidates.add(max(0, at - offset))
                at = haystack_lower.find(gram, at + 1)
        if candidates:
            return sorted(candidates)
    # No shared word at all. On a small haystack a strided sweep is cheap and
    # keeps short quotes locatable; on a large one, give up rather than scan.
    if len(haystack_lower) <= _SMALL_HAYSTACK:
        stride = max(1, len(needle_lower) // 4)
        return list(range(0, max(1, len(haystack_lower)), stride))
    return []


def _lower_keep_length(text: str) -> str:
    """Lowercase without changing the string's length.

    `str.lower()` is not length preserving: "İ".lower() is two characters,
    so a single such letter before a passage shifts every offset computed in
    the lowered string by one. `_best_window` reports offsets from the lowered
    haystack but slices the ORIGINAL, so any length change skews the offset,
    the served window, the prefix and suffix, and the diff. A character whose
    lowercase is not exactly one character is left as it stands: an aligned
    offset matters more here than folding a dotted capital I, and case is only
    a scoring convenience at T3, never an identity claim.
    """
    return "".join(
        low if len(low := ch.lower()) == 1 else ch for ch in text
    )


def _best_window(
    haystack: str, needle: str, score: Callable[[str, str], float]
) -> tuple[float, int, str]:
    """Best-scoring window of `haystack` for `needle`.

    Returns (score, offset, window_text), or (0.0, -1, "") if no candidate.
    Scoring is case-insensitive: character identity is what T1 and T2 are for.

    Two phases. Every candidate is scored once at zero padding, then only the
    few best are re-scored at wider paddings, which is where an insertion in
    the document is absorbed. Padding refinement only ever matters near the
    true location, and the true location always ranks at zero padding, so this
    costs about a third of scoring every candidate at every padding.
    """
    haystack_lower = _lower_keep_length(haystack)
    needle_lower = _lower_keep_length(needle)
    width = len(needle_lower)
    # Each candidate is scored with an O(width squared) comparison, so scoring
    # every seed against a long needle is a denial-of-service lever: a caller
    # who sends a long string that reaches T3 pays width squared per window
    # across up to _MAX_CANDIDATES windows. Cap the number of windows so the
    # product width-squared-times-windows stays bounded. For an ordinary quote
    # the cap is far above _MAX_CANDIDATES and nothing changes; it bites only on
    # a needle long enough to be a resource hazard, where examining fewer
    # windows costs only best-effort recall on a fuzzy relocation that already
    # is not a verbatim verdict. The server enforces a length ceiling too; this
    # is the second, scale-independent lever that holds even if that is raised.
    candidate_cap = max(30, _WINDOW_SCORE_BUDGET // max(1, width * width))
    seeds = _seed_offsets(haystack_lower, needle_lower)[:candidate_cap]
    scored: list[tuple[float, int]] = []
    for start in seeds:
        b = min(len(haystack), start + width)
        if start >= b:
            continue
        scored.append((score(needle_lower, haystack_lower[start:b]), start))
    if not scored:
        return 0.0, -1, ""
    scored.sort(reverse=True)

    best_score, best_at = scored[0]
    best_window = haystack[best_at : min(len(haystack), best_at + width)]
    for _value, start in scored[:_REFINE_TOP]:
        for pad in (width // 5, width // 2):
            a = max(0, start - pad // 2)
            b = min(len(haystack), start + width + pad)
            if a >= b:
                continue
            value = score(needle_lower, haystack_lower[a:b])
            if value > best_score:
                best_score = value
                best_at = a
                best_window = haystack[a:b]
    return best_score, best_at, best_window


# --------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------


def _truncate(text: str) -> str:
    if len(text) <= _DIFF_MAX_SEGMENT:
        return text
    return text[:_DIFF_MAX_SEGMENT] + "..."


def _char_diff(quote: str, document: str) -> str:
    """A compact character diff, quote against the located document window.

    `-` is text present in the quote, `+` is what the document has instead.
    """
    matcher = difflib.SequenceMatcher(None, quote, document)
    lines = ["char diff: - quote, + document"]
    shown = 0
    hidden = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if shown >= _DIFF_MAX_OPCODES:
            hidden += 1
            continue
        context = quote[max(0, i1 - _DIFF_CONTEXT) : i1]
        lines.append("  @%d after %r" % (i1, _truncate(context)))
        if i2 > i1:
            lines.append("    - %r" % (_truncate(quote[i1:i2]),))
        if j2 > j1:
            lines.append("    + %r" % (_truncate(document[j1:j2]),))
        shown += 1
    if hidden:
        lines.append("  ... %d further difference(s) not shown" % hidden)
    if shown == 0:
        lines.append("  (no character differences: the texts are identical)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the one lossy fold that character identity may not be reached through
# --------------------------------------------------------------------------

#: Compatibility decompositions that move a character out of its position and
#: so change what a number means. NFKC folds "10\u2076" to "106" and
#: "10\u207b\u00b3" to "10-3", which are different numbers by five and six
#: orders of magnitude. The same fold flattens a vulgar fraction and an ordinal
#: indicator.
_POSITIONAL_TAGS = ("<super>", "<sub>", "<fraction>")


def _positional_marks(text: str) -> tuple[str, ...]:
    """The superscript, subscript and fraction characters in `text`, in order.

    Everything else NFKC folds is safe for a character-identity verdict: a
    ligature, a fullwidth form, a math alphanumeric and a no-break space all
    fold to the letters a reader would type, which is exactly why the fold is
    there. A circled digit and a squared unit were looked at and left alone
    too: U+2460 folds to "1" and U+33A1 to "m2", and in both the fold keeps
    the meaning. The three tags above do not. They discard a position that
    carries the meaning, so two strings that are character-identical after the
    fold can be different numbers. A fraction is included for the same reason
    even though a fraction folds to its own value: "2\u00bd" folds to "21/2",
    which is two and a half becoming twenty-one halves.
    """
    return tuple(
        ch for ch in text
        if unicodedata.decomposition(ch).startswith(_POSITIONAL_TAGS)
    )


def _raw_of(prepared: PreparedText, offset: int, length: int) -> str:
    """The document's own characters behind a span of the normalised text."""
    index_map = prepared.norm_index_map
    if offset < 0 or length <= 0 or offset >= len(index_map):
        return ""
    end_norm = offset + length
    start_raw = index_map[offset]
    end_raw = index_map[end_norm] if end_norm < len(index_map) else len(prepared.raw)
    # A single source character can expand to several normalised characters
    # (a vulgar fraction to "1/2", a ligature to "fi"). When the matched span
    # ends INSIDE such an expansion, index_map at the boundary points back at
    # the shared source character, so the raw span would stop just before it
    # and the fold guard would never see the fraction. Force the source
    # character of the LAST matched normalised character to sit inside the raw
    # span. In the ordinary one-to-one case this is a no-op, because there
    # index_map[end_norm - 1] + 1 already equals index_map[end_norm].
    last_src = index_map[min(end_norm - 1, len(index_map) - 1)]
    if last_src + 1 > end_raw:
        end_raw = last_src + 1
    if end_raw <= start_raw:
        return ""
    return prepared.raw[start_raw:end_raw]


def _identity_survives_the_fold(
    prepared: PreparedText, quote: str, offset: int, length: int
) -> bool:
    """Does this span still equal the quote once the lossy fold is undone?

    Called before T1 and before T2, which are the only two tiers that assert
    character identity. It compares the superscript, subscript and fraction
    characters of the quote against those of the document span. If they differ,
    the two strings became equal only because NFKC threw away a position, so
    they are not the same string and no tier that says so may be returned.

    Attack this closes, run against the real ladder: a document reading
    "Throughput reached 10\u2076 operations per second" returned
    VERBATIM_EXACT for the quote "Throughput reached 106 operations per
    second". The number in the quote is a million times smaller than the number
    in the paper, and the outcome asserted they were the same characters.
    """
    return _positional_marks(quote) == _positional_marks(
        _raw_of(prepared, offset, length)
    )


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------


def locate(
    haystack: str | PreparedText,
    quote: str,
    t_locate: float = T_LOCATE_DEFAULT,
    scorer: str = "difflib",
) -> Anchor:
    """Walk the ladder and return an Anchor. Never returns None.

    A miss is not an error: it comes back as a T4_NOT_LOCATABLE Anchor, so a
    caller has to look at the tier to find out what happened.

    The return type is a Tier, NEVER a bool, and that is load bearing. A
    boolean return is exactly how a T3_LOCATED gets silently mistaken for a
    pass: the passage was found, so the call looks successful, while the quote
    text actually differs from the document. Preventing that confusion is the
    reason this whole system exists. Only `Tier.T1_EXACT` and `Tier.T2_RELAXED`
    assert "verbatim" (see `Anchor.is_verbatim`); T3 asserts only "the passage
    is here and your string differs, here is the diff".

    `haystack` may be a raw string or a `PreparedText` from `prepare()`, which
    is what to use when anchoring many quotes into one document.

    Raises ValueError for an unknown scorer name, or for scorer="rapidfuzz"
    when rapidfuzz is not installed. It never raises for a failed lookup.
    """
    score_fn = _resolve_scorer(scorer)
    prepared = haystack if isinstance(haystack, PreparedText) else prepare(haystack)

    quote_norm = normalise(quote)[0]
    if not quote_norm.strip() or not prepared.norm:
        return Anchor(Tier.T4_NOT_LOCATABLE, None, None, None, None, scorer)

    # T1: character identity in the normalised text. No threshold. Walk EVERY
    # occurrence: a quote may appear once where the fold guard demotes it (a
    # superscript source) and again where it holds (a plain-digit source), and
    # the identity verdict the document genuinely supports must win over the
    # first occurrence that happens not to.
    qlen = len(quote_norm)
    first_at = prepared.norm.find(quote_norm)
    at = first_at
    while at >= 0:
        if _identity_survives_the_fold(prepared, quote, at, qlen):
            return Anchor(Tier.T1_EXACT, at, qlen, 1.0, None, scorer)
        at = prepared.norm.find(quote_norm, at + 1)
    if first_at >= 0:
        return _folded_apart(prepared, quote, first_at, qlen, score_fn, scorer)

    # T2: character identity in the match form. Still no threshold. This
    # survives damage the EXTRACTOR did (a compound hyphen eaten at a line
    # break, a case change), not damage the quote did. Walk every match-form
    # occurrence too, for the same reason, and refuse a case difference that
    # lands on a unit prefix.
    quote_match = match_form(quote_norm)
    if quote_match:
        first_span: tuple[int, int] | None = None
        search_from = 0
        while True:
            span = _match_span(prepared, quote_match, search_from)
            if span is None:
                break
            start, end, match_at = span
            if first_span is None:
                first_span = (start, end)
            doc_span = prepared.norm[start:end]
            if _identity_survives_the_fold(
                prepared, quote, start, end - start
            ) and not _casefold_hazard_next_to_digit(quote_norm, doc_span):
                return Anchor(Tier.T2_RELAXED, start, end - start, 1.0, None, scorer)
            search_from = match_at + 1
        if first_span is not None:
            start, end = first_span
            return _folded_apart(prepared, quote, start, end - start, score_fn, scorer)

    # T3: fuzzy relocation. The verdict is not "verbatim".
    best_score, best_at, window = _best_window(prepared.norm, quote_norm, score_fn)
    if best_at >= 0 and best_score >= t_locate:
        return Anchor(
            Tier.T3_LOCATED,
            best_at,
            len(window),
            best_score,
            _char_diff(quote_norm, window),
            scorer,
        )

    # T4: not locatable. A third outcome, neither success nor invalidation.
    # The best score is reported so a caller can say how close it got, but no
    # offset is ever handed back, because nothing was located.
    return Anchor(Tier.T4_NOT_LOCATABLE, None, None, best_score, None, scorer)


def _folded_apart(
    prepared: PreparedText,
    quote: str,
    offset: int,
    length: int,
    score_fn: Callable[[str, str], float],
    scorer: str,
) -> Anchor:
    """The T3 verdict for a span that only matched because of a lossy fold.

    T3 and not T4: the passage really is there, at this offset, and refusing to
    say where it is would be less useful and no safer. T3 and not T1 or T2: the
    quote is not what the document says at that offset, which is precisely what
    PASSAGE_RELOCATED_QUOTE_DIFFERS means. The diff is taken over the RAW forms
    rather than the normalised ones, because in the normalised forms there is
    nothing left to see: the fold is the whole difference.

    The score is measured between the raw forms too, so it reports the real
    distance rather than the 1.0 the normalised comparison would give. The
    threshold plays no part: this span was reached by identity, so it is
    located whatever `t_locate` says.
    """
    raw_span = _raw_of(prepared, offset, length)
    return Anchor(
        Tier.T3_LOCATED,
        offset,
        length,
        score_fn(quote.lower(), raw_span.lower()),
        _char_diff(quote, raw_span),
        scorer,
    )


# --------------------------------------------------------------------------
# W3C Web Annotation record
# --------------------------------------------------------------------------


def anchor_record(
    quote_raw: str,
    haystack: str | PreparedText,
    *,
    source: str | None = None,
    t_locate: float = T_LOCATE_DEFAULT,
    scorer: str = "difflib",
    context_chars: int = 32,
) -> dict[str, Any]:
    """Emit a storable anchor record using W3C Web Annotation selectors.

    The shape follows the W3C Web Annotation Data Model (W3C Recommendation,
    23 February 2017) rather than a bespoke schema, because inventing an
    anchor format when a standard already exists buys nothing and costs every
    reader who already knows the standard.

    Two selectors are emitted on the target:

      TextQuoteSelector    exact, prefix, suffix. AUTHORITATIVE.
      TextPositionSelector start, end. A SPEED HINT ONLY.

    That split is exactly how the Hypothesis client treats them: the position
    is tried first because it is cheap, and the quote is what decides.

    `quote_raw` is the primary key and the position is a cache. A normaliser,
    match-form or verifier version change therefore degrades a stored record to
    STALE (see `record_status`), and STALE means "re-anchor from quote_raw"
    (see `reanchor_record`). It NEVER means the citation is invalid. Measured
    on a private, non-redistributable 44-document corpus of scientific PDFs:
    bumping the normaliser invalidated 396 of 400 stored offsets, and
    re-anchored 400 of 400 from the raw quote in about 5 seconds corpus-wide.

    start and end index the NORMALISED text produced by the recorded normaliser
    version, which is precisely why a normaliser bump makes them stale.

    The record carries BOTH a hand-maintained version literal AND a digest of
    the actual code, for the normaliser (`normaliser`, `normaliser_fingerprint`)
    and for the verifier (`verifier`, `verifier_fingerprint`) alike. Storing
    only the literal would mean that a behaviour change nobody version-bumped
    produced records claiming a version they do not have, and every one of them
    would read as FRESH forever. The fingerprints close that hole: they move
    whenever the code moves, so a forgotten bump surfaces as a STALE record
    rather than as silence. The verifier gets the same guard as the normaliser
    because it carries the same risk: this module decides which tier a quote
    reaches, and a silent change to that decision is a citation that stopped
    being checked the way the record says it was.
    """
    prepared = haystack if isinstance(haystack, PreparedText) else prepare(haystack)
    anchor = locate(prepared, quote_raw, t_locate=t_locate, scorer=scorer)
    quote_norm = normalise(quote_raw)[0]

    prefix = ""
    suffix = ""
    located_text = None
    if anchor.offset_norm is not None and anchor.length_norm is not None:
        start = anchor.offset_norm
        end = start + anchor.length_norm
        prefix = prepared.norm[max(0, start - context_chars) : start]
        suffix = prepared.norm[end : end + context_chars]
        located_text = prepared.norm[start:end]

    # W3C TextQuoteSelector.exact is the text OF THE DOCUMENT at the selected
    # position, and prefix/suffix already come from the document around it. At
    # T1 the located text and the normalised quote are the same string, but at
    # T2 (a case or hyphen difference) and T3 (a fuzzy relocation) they differ,
    # so writing the quote into `exact` produced a selector whose
    # prefix+exact+suffix was NOT a substring of the document and could not be
    # re-anchored by any standard client. The document's own characters go in
    # `exact`; the caller's quote is kept, distinctly, under refs.quote_raw.
    quote_selector: dict[str, Any] = {
        "type": "TextQuoteSelector",
        "exact": located_text if located_text is not None else quote_norm,
        "prefix": prefix,
        "suffix": suffix,
    }
    selectors: list[dict[str, Any]] = [quote_selector]
    if anchor.offset_norm is not None and anchor.length_norm is not None:
        selectors.append(
            {
                "type": "TextPositionSelector",
                "start": anchor.offset_norm,
                "end": anchor.offset_norm + anchor.length_norm,
            }
        )

    target: dict[str, Any] = {"selector": selectors}
    if source is not None:
        target["source"] = source

    return {
        "@context": W3C_ANNOTATION_CONTEXT,
        "type": "Annotation",
        "target": target,
        "refs": {
            # The primary key. Everything else can be rebuilt from this.
            "quote_raw": quote_raw,
            "tier": anchor.tier.value,
            "is_verbatim": anchor.is_verbatim,
            "score": anchor.score,
            "diff": anchor.diff,
            "located_text": located_text,
            "authoritative_selector": "TextQuoteSelector",
            "position_selector_is_cache": True,
            "normaliser": NORMALISER_VERSION,
            "normaliser_fingerprint": NORMALISER_FINGERPRINT,
            "matchform": MATCHFORM_VERSION,
            "verifier": VERIFIER_VERSION,
            "verifier_fingerprint": VERIFIER_FINGERPRINT,
            "scorer": anchor.scorer,
        },
    }


def record_status(record: dict[str, Any]) -> str:
    """STATUS_FRESH if every stored version AND the fingerprint match.

    STALE is not invalid. It means the cached position can no longer be trusted
    and the record must be re-anchored from `quote_raw`.

    Both fingerprints, the normaliser's and the verifier's, are checked
    alongside the version strings, which is the whole point of them: a
    behaviour change that nobody version-bumped still degrades the record to
    STALE. A record written before a fingerprint existed carries no such key
    and therefore reads STALE, which is the correct and cheap answer, because
    such a record cannot be shown to be fresh.
    """
    refs = record.get("refs", {})
    current = {
        "normaliser": NORMALISER_VERSION,
        "normaliser_fingerprint": NORMALISER_FINGERPRINT,
        "matchform": MATCHFORM_VERSION,
        "verifier": VERIFIER_VERSION,
        "verifier_fingerprint": VERIFIER_FINGERPRINT,
    }
    for key, value in current.items():
        if refs.get(key) != value:
            return STATUS_STALE
    return STATUS_FRESH


def record_is_stale(record: dict[str, Any]) -> bool:
    """True when a stored record needs re-anchoring. See `record_status`."""
    return record_status(record) == STATUS_STALE


def reanchor_record(
    record: dict[str, Any],
    haystack: str | PreparedText,
    *,
    t_locate: float = T_LOCATE_DEFAULT,
    scorer: str | None = None,
    context_chars: int = 32,
) -> dict[str, Any]:
    """Rebuild a record from its `quote_raw`, keeping any caller-owned keys.

    This is the whole answer to a version bump: never invalidate a citation
    because an offset moved, re-anchor it from the raw quote. Keys the caller
    added to the record (an id, a task reference) are carried through; only
    `target` and `refs` are replaced.
    """
    refs = record.get("refs", {})
    quote_raw = refs.get("quote_raw")
    if not isinstance(quote_raw, str) or not quote_raw:
        raise ValueError(
            "record has no refs.quote_raw, so it cannot be re-anchored. "
            "quote_raw is the primary key of an anchor record."
        )
    source = record.get("target", {}).get("source")
    fresh = anchor_record(
        quote_raw,
        haystack,
        source=source,
        t_locate=t_locate,
        scorer=scorer if scorer is not None else refs.get("scorer", "difflib"),
        context_chars=context_chars,
    )
    out = dict(record)
    out["@context"] = fresh["@context"]
    out["type"] = fresh["type"]
    out["target"] = fresh["target"]
    out["refs"] = fresh["refs"]
    return out
