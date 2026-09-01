"""Quote-anchoring normaliser for scientific source documents.

A quote anchor stores an offset into a normalised rendering of a source
document. This module defines that rendering. It is a pure function of a
single string: no file I/O, no network, no mutable global state, stdlib only,
so it runs on a bare Python with nothing installed.

`normalise()` returns the normalised text AND a per-character index map back
into the input, so any normalised offset can always be rendered as a raw span
in the source file. That map is what makes an anchor auditable: a stored
offset can be walked back to the exact bytes it came from.


WHERE THE NUMBERS IN THIS DOCSTRING CAME FROM
---------------------------------------------
Every percentage and count quoted below was measured on a private 44-document
corpus of scientific PDFs which cannot be redistributed, because the papers are
paywalled or licence-limited. A reader of this repository therefore cannot
re-run those figures against the same inputs. They are kept here because they
are the evidence for each design choice, not because they are reproducible.
What IS reproducible is the BEHAVIOUR the numbers justify: the test suite pins
every stage on short inline fixtures and reads no corpus file at all.


THE VERSION CONTRACT
--------------------
`NORMALISER_VERSION` is load-bearing. A stored anchor records the version that
produced it. When this module changes in a way that moves offsets, bump the
version. A reader that finds an anchor whose recorded version differs from
`NORMALISER_VERSION` must RE-ANCHOR that quote (re-run the search under the
current normaliser and rewrite the offset), NOT invalidate it. A version
mismatch is a stale coordinate system, never evidence that the quote is wrong.
The same contract holds for `MATCHFORM_VERSION` and the T2 relaxed form.

Bump the patch digit for a change that cannot move any offset (comments, type
hints, a faster loop). Bump the minor digit for a change to the fold tables or
the stage behaviour. Bump the major digit for a change to the stage set or to
the meaning of the index map.


THE FINGERPRINT BACKSTOP
------------------------
`NORMALISER_VERSION` is a hand-maintained string literal, and a hand-maintained
literal fails silently. If the behaviour here changes and nobody edits the
literal, every anchor stored afterwards records a version it does not actually
have, no record ever reads as stale, and the re-anchoring mechanism the whole
design rests on stops firing without anyone being told.

`NORMALISER_FINGERPRINT` is the backstop. It is a short digest, computed once
at import, of this module's own abstract syntax tree with docstrings removed.
It therefore moves when the CODE moves, and holds still when only comments,
docstrings, blank lines or formatting move. Both strings are written onto every
anchor record, so a forgotten version bump is caught by the fingerprint rather
than by nothing at all.

A fingerprint mismatch means precisely what a version mismatch means: the
record is STALE and must be re-anchored from its raw quote. It is never
evidence that the citation was wrong.


THE PIPELINE
------------
Stages run in this order, and each one is justified by a measurement quoted
below so the reasoning is inspectable rather than asserted. `stages=` disables
any individual stage, which is what makes the ablation re-runnable by anyone
holding a corpus of their own.

S0  Invisible-character strip. Removes U+00AD SOFT HYPHEN, U+200B ZERO WIDTH
    SPACE, U+200C ZERO WIDTH NON-JOINER, U+200D ZERO WIDTH JOINER, U+2060 WORD
    JOINER, U+FEFF ZERO WIDTH NO-BREAK SPACE and U+180E MONGOLIAN VOWEL
    SEPARATOR. Runs BEFORE NFKC and is not optional. NFKC does not remove
    these: U+200B has no compatibility decomposition, so it survives NFKC
    untouched. Measured: the corpus holds 2,222 ZERO WIDTH SPACE characters
    across 3 documents, 1,730 of them in a single document, interleaved
    character by character inside URLs. On quotes drawn from those 3 documents
    and retyped without the zero-width characters, S0 on gives 403/403 exact
    matches and S0 off gives 0/403. It is the difference between 100% and 0%.
    Known limit, on the record: U+200C and U+200D are orthographic in some
    scripts, so deleting them can merge sequences a reader would keep apart, a
    Perso-Arabic "mikhaham" written with a ZERO WIDTH NON-JOINER against the
    joined form, or a ZERO WIDTH JOINER emoji sequence against its parts. In
    scientific prose that is vanishingly rare and the URL case above is common,
    so S0 stays; the tradeoff is stated rather than hidden.

S1  NFKC, applied CLUSTER-WISE (one starter plus its trailing combining
    marks, plus any following character whose own decomposition begins with a
    combining mark) so composition cannot silently shift the index map.
    Cluster-wise output is identical to whole-string
    `unicodedata.normalize("NFKC", t)` on all 44 documents; the test suite
    keeps that as an assertion on inline fixtures. NFKC and not NFC because
    4,185 Mathematical Alphanumeric Symbols in 7 documents fold to plain
    letters under NFKC and not at all under NFC, and that fold is what makes
    a formula quotable: on quotes whose math alphanumerics are typed as plain
    letters, NFKC gives 605/605 and NFC gives 0/605. NFKC is deliberately
    lossy in two documented ways: U+00B2 becomes "2" and U+00BA becomes "o".
    Both folds stay, because an anchor is a locator and the raw span it maps
    back to is the evidence. They are NOT allowed to reach a verdict that
    asserts character identity, though, and an earlier version let them. A
    document reading "10\u2076 operations" returned an exact-identity verdict
    for the quote "106 operations", which is a different number by six orders
    of magnitude. `science2code.anchor` now compares the superscript,
    subscript and fraction characters of the quote against those of the
    document span before T1 or T2 may be returned, and reports a located
    passage with a character diff instead when they differ. The normaliser is
    unchanged by that, which is why this version literal has not moved.

S2  Punctuation fold. Curly and angle quotes to "'" and '"', every dash
    variant and U+2212 MINUS SIGN to "-", U+2026 to "...", U+2044 to "/",
    U+037E to ";", and U+00A0 / U+2000 to U+200A / U+202F / U+205F / U+3000 to
    a single U+0020. Worth 16.2 points corpus-wide: on quotes retyped with
    ASCII punctuation, the full pipeline scores 90.1% exact and dropping S2
    scores 73.9%. On the subset of quotes that actually contain foldable
    punctuation the gap is 99.9% against 0.6%.

S3  De-hyphenate across a line break. A letter, then "-", then optional
    spaces or tabs, then a SINGLE newline, then optional spaces or tabs, then
    a LOWERCASE letter: the hyphen and the break are deleted. It must not jump
    a blank line, because a blank line means intervening page furniture (a
    page number, a running head). Worth 20.8 points: 90.1% with S3 against
    69.3% without. The lowercase condition is measured, not assumed: on 5,294
    line-break hyphen sites labelled against an independent reading-order
    rendering of the same PDF, always dropping the hyphen errs on 0.00% and
    always keeping it errs on 100%, so dropping is right; the residual risk is
    a real compound split at a line break, which S3 joins wrongly and which
    the T2 match form below recovers.

S4  Whitespace collapse. Any run of whitespace, form feed and vertical tab
    included, becomes a single U+0020, and the ends are stripped. Worth 5.1
    points: 90.1% against 85.0%.

CASE IS PRESERVED. `normalise()` never casefolds. Casefolding belongs only to
`match_form()`, which is a comparison form and is never displayed and never
stored as a quote.


THE T2 RELAXED MATCH FORM
-------------------------
`match_form()` derives a second, more aggressive form from the normalised
text: it deletes hyphens that sit between two letters, deletes a hyphen that
sits between a letter and a space followed by a letter (the same hyphen with
the line break still beside it, which is what a caller who ran their own
extractor hands over), and casefolds. It exists
to survive damage the EXTRACTOR did, not damage the quote did. pdftotext's own
reading-order de-hyphenation drops the hyphen of a genuine compound at a line
break, so the extracted text holds "longterm" where the quote an agent types is
"long-term". Measured: on 613 such quotes across all 44 documents, exact
matching recovers 0 and the T2 form recovers 609 (99.3%). Deleting the hyphen
on BOTH sides cannot be wrong, where a rule that decides whether to keep it
errs on 1.27% of labelled sites.

The match form is for character-identity matching only. Offsets found in it
are mapped back through `match_form_with_map()` to raw source offsets before
anything is stored or shown.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import unicodedata
from collections.abc import Iterable

NORMALISER_VERSION = "norm/1.1.0"
MATCHFORM_VERSION = "match/1.1.0"

__all__ = [
    "NORMALISER_VERSION",
    "NORMALISER_FINGERPRINT",
    "MATCHFORM_VERSION",
    "FINGERPRINT_UNAVAILABLE",
    "fingerprint_source",
    "fingerprint_file",
    "ALL_STAGES",
    "S0_INVISIBLE",
    "S1_NFKC",
    "S2_PUNCTUATION",
    "S3_DEHYPHENATE",
    "S4_WHITESPACE",
    "normalise",
    "normalise_text",
    "match_form",
    "match_form_with_map",
    "match_fold",
]

# ---------------------------------------------------------------------------
# The fingerprint backstop. See THE FINGERPRINT BACKSTOP in the module
# docstring for why a hand-maintained version literal is not enough on its own.
# ---------------------------------------------------------------------------
#: Reported instead of a digest when this module was shipped without its
#: source (a zipimport, a frozen build). Comparable like any other value: it
#: is stable, so two records that both carry it still agree with each other.
FINGERPRINT_UNAVAILABLE = "unavailable"

_FINGERPRINT_LENGTH = 12


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove every docstring node in place, so prose cannot move a digest."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def fingerprint_source(source: str) -> str:
    """Digest what a Python source string DOES, ignoring how it is written.

    The digest is taken over the abstract syntax tree, with docstrings deleted
    and without position attributes. So it is blind to comments, docstrings,
    blank lines and reformatting, and it moves for any change to a literal, a
    table, an expression or the control flow.

    Public rather than private so that a maintainer can fingerprint an edit
    before committing it and see for themselves whether it is behavioural.

    Raises SyntaxError if `source` does not parse.
    """
    tree = _strip_docstrings(ast.parse(source))
    return hashlib.sha256(
        ast.dump(tree).encode("utf-8")).hexdigest()[:_FINGERPRINT_LENGTH]


def fingerprint_file(path: str | None) -> str:
    """Fingerprint the Python source at `path`. Never raises.

    Costs a few milliseconds, once, at import. A self-check that could break
    an import would be worse than the problem it guards against, so every
    failure path returns FINGERPRINT_UNAVAILABLE instead.

    Public and takes a path rather than being hardcoded to this module,
    because `science2code.anchor` needs exactly the same backstop for
    VERIFIER_VERSION. One implementation, so the two cannot drift into
    disagreeing about what a fingerprint is.

    A module fingerprints itself with `fingerprint_file(globals().get(
    "__file__"))`. `globals().get` rather than a bare `__file__`, because a
    frozen or zipimported build may not define it at all.
    """
    if not path:  # pragma: no cover - no source to read
        return FINGERPRINT_UNAVAILABLE
    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
    except OSError:  # pragma: no cover - source not shipped
        return FINGERPRINT_UNAVAILABLE
    try:
        return fingerprint_source(source)
    except (SyntaxError, ValueError, RecursionError):  # pragma: no cover
        return FINGERPRINT_UNAVAILABLE


#: A digest of this module's behaviour, stored alongside NORMALISER_VERSION on
#: every anchor record. A mismatch means STALE, exactly as a version mismatch
#: does, and never means the quote was wrong.
NORMALISER_FINGERPRINT = fingerprint_file(globals().get("__file__"))


S0_INVISIBLE = 0
S1_NFKC = 1
S2_PUNCTUATION = 2
S3_DEHYPHENATE = 3
S4_WHITESPACE = 4

ALL_STAGES = (S0_INVISIBLE, S1_NFKC, S2_PUNCTUATION, S3_DEHYPHENATE,
              S4_WHITESPACE)

_STAGE_NAMES = {
    S0_INVISIBLE: "S0 invisible-character strip",
    S1_NFKC: "S1 NFKC",
    S2_PUNCTUATION: "S2 punctuation fold",
    S3_DEHYPHENATE: "S3 de-hyphenate across line breaks",
    S4_WHITESPACE: "S4 whitespace collapse",
}


# ---------------------------------------------------------------------------
# S0 tables
# ---------------------------------------------------------------------------
# Characters that carry no glyph and no meaning for a quote, and that NFKC
# leaves in place. Escapes, not literals, so the source stays greppable.
INVISIBLE: frozenset[str] = frozenset({
    "\u00ad",  # SOFT HYPHEN
    "\u180e",  # MONGOLIAN VOWEL SEPARATOR
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE (byte order mark)
})


# ---------------------------------------------------------------------------
# S2 tables
# ---------------------------------------------------------------------------
def _build_punct_fold() -> dict[str, str]:
    """Build the S2 fold table. Called once at import; the result is frozen."""
    table: dict[str, str] = {}

    single_quotes = (
        "\u2018"  # LEFT SINGLE QUOTATION MARK
        "\u2019"  # RIGHT SINGLE QUOTATION MARK
        "\u201a"  # SINGLE LOW-9 QUOTATION MARK
        "\u201b"  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
        "\u2032"  # PRIME
        "\u2035"  # REVERSED PRIME
        "\u00b4"  # ACUTE ACCENT
        "\u0060"  # GRAVE ACCENT
        "\u02bc"  # MODIFIER LETTER APOSTROPHE
        "\u02b9"  # MODIFIER LETTER PRIME
    )
    double_quotes = (
        "\u201c"  # LEFT DOUBLE QUOTATION MARK
        "\u201d"  # RIGHT DOUBLE QUOTATION MARK
        "\u201e"  # DOUBLE LOW-9 QUOTATION MARK
        "\u201f"  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
        "\u2033"  # DOUBLE PRIME
        "\u2036"  # REVERSED DOUBLE PRIME
        "\u00ab"  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
        "\u00bb"  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
        "\u02ba"  # MODIFIER LETTER DOUBLE PRIME
        "\u3003"  # DITTO MARK
    )
    dashes = (
        "\u2010"  # HYPHEN
        "\u2011"  # NON-BREAKING HYPHEN
        "\u2012"  # FIGURE DASH
        "\u2013"  # EN DASH
        "\u2014"  # EM DASH
        "\u2015"  # HORIZONTAL BAR
        "\u2212"  # MINUS SIGN
        "\u2043"  # HYPHEN BULLET
        "\ufe58"  # SMALL EM DASH
        "\ufe63"  # SMALL HYPHEN-MINUS
        "\uff0d"  # FULLWIDTH HYPHEN-MINUS
    )
    spaces = (
        "\u00a0"  # NO-BREAK SPACE
        "\u2000\u2001\u2002\u2003\u2004\u2005"  # EN QUAD to FOUR-PER-EM SPACE
        "\u2006\u2007\u2008\u2009\u200a"  # SIX-PER-EM SPACE to HAIR SPACE
        "\u202f"  # NARROW NO-BREAK SPACE
        "\u205f"  # MEDIUM MATHEMATICAL SPACE
        "\u3000"  # IDEOGRAPHIC SPACE
    )

    for ch in single_quotes:
        table[ch] = "'"
    for ch in double_quotes:
        table[ch] = '"'
    for ch in dashes:
        table[ch] = "-"
    for ch in spaces:
        table[ch] = " "

    table["\u2026"] = "..."  # HORIZONTAL ELLIPSIS
    table["\u2044"] = "/"    # FRACTION SLASH
    table["\u037e"] = ";"    # GREEK QUESTION MARK
    return table


PUNCT_FOLD: dict[str, str] = _build_punct_fold()

# Every one of these folds to "-" in S2 and NONE of them is ever a word break
# at the end of a line. S3 deletes a hyphen that sits at a line break because
# there a hyphen is usually the extractor's syllable break; an em dash, en
# dash, figure dash, horizontal bar or minus sign at the end of a line is
# punctuation BETWEEN two whole words, so deleting it fuses them.
#
# Measured on a 45-document corpus of real scientific PDFs: 29 sites across 14
# documents. Before this set existed, "difficult to achieve\u2014\nin which case"
# normalised to "difficult to achievein which case", a word that is in no
# paper, and every quote spanning such a site was unreachable at T1 and T2
# because the characters the quote holds are not in the rendering at all.
# S3 now deletes the line break and KEEPS the dash, which is what the page
# shows the reader.
NOT_A_LINE_BREAK_HYPHEN: frozenset[str] = frozenset({
    "\u2012",  # FIGURE DASH
    "\u2013",  # EN DASH
    "\u2014",  # EM DASH
    "\u2015",  # HORIZONTAL BAR
    "\u2212",  # MINUS SIGN
    "\u2043",  # HYPHEN BULLET
    "\ufe58",  # SMALL EM DASH
})

# U+2010 HYPHEN, U+2011 NON-BREAKING HYPHEN, U+FE63 SMALL HYPHEN-MINUS and
# U+FF0D FULLWIDTH HYPHEN-MINUS are deliberately absent: each is a hyphen, so
# each can be a line-break hyphen and S3 may delete it.
assert all(PUNCT_FOLD.get(ch) == "-" for ch in NOT_A_LINE_BREAK_HYPHEN)

# U+00AD SOFT HYPHEN is deliberately NOT in PUNCT_FOLD. It is an invisible and
# belongs to S0 alone, so that each stage owns exactly one behaviour and an
# ablation that switches S0 off measures S0 and nothing else.
assert "\u00ad" not in PUNCT_FOLD


# ---------------------------------------------------------------------------
# S3 tables
# ---------------------------------------------------------------------------
# Horizontal whitespace tolerated between the hyphen and the newline, and
# between the newline and the continuation. Form feed and vertical tab are
# excluded on purpose: they mean a page or column break, which is exactly the
# case where the continuation is not a continuation.
_H_SPACE: frozenset[str] = frozenset({" ", "\t", "\r"})

_MARK_CATEGORIES: frozenset[str] = frozenset({"Mn", "Mc", "Me"})

# Conjoining Hangul jamo that attach to the syllable in front of them: the
# vowel (V) and trailing consonant (T) jamo, plus the Unicode 5.2 extensions.
# These are the one case where canonical composition joins two STARTERS, so a
# category test alone would split a syllable the whole-string algorithm joins.
# The leading (L) jamo are deliberately absent: they open a syllable.
_JAMO_RANGES: tuple[tuple[int, int], ...] = (
    (0x1161, 0x1175),  # V, HANGUL JUNGSEONG A to I
    (0x11A8, 0x11C2),  # T, HANGUL JONGSEONG KIYEOK to HIEUH
    (0xD7B0, 0xD7C6),  # V extended
    (0xD7CB, 0xD7FB),  # T extended
)


@functools.lru_cache(maxsize=8192)
def _continues_cluster(ch: str, form: str) -> bool:
    """True when `ch` must stay attached to the cluster before it.

    A combining mark obviously must. So must any character whose own
    normalisation BEGINS with a combining mark, because splitting there would
    let the cluster-wise pass miss a composition that whole-string
    normalisation would perform. U+FF9E HALFWIDTH KATAKANA VOICED SOUND MARK
    is the canonical example: its NFKC image is U+3099, which composes with
    the preceding kana. Conjoining Hangul jamo are the third case, handled by
    range because their composition is algorithmic rather than tabulated.

    Every canonical two-character composite in the Unicode data whose second
    character is a starter has category Mc or Mn, so the category test covers
    all of them; the test suite asserts that by enumeration rather than trust.
    """
    code = ord(ch)
    for lo, hi in _JAMO_RANGES:
        if lo <= code <= hi:
            return True
    if unicodedata.category(ch) in _MARK_CATEGORIES:
        return True
    decomposed = unicodedata.normalize(form, ch)
    return bool(decomposed) and unicodedata.category(decomposed[0]) in _MARK_CATEGORIES


def _resolve_stages(stages: Iterable[int]) -> frozenset[int]:
    """Validate the requested stage set and return it as a frozenset."""
    try:
        active = frozenset(stages)
    except TypeError as exc:
        raise TypeError("stages must be an iterable of stage numbers") from exc
    unknown = active - frozenset(ALL_STAGES)
    if unknown:
        raise ValueError(
            "unknown normaliser stage(s) %s; valid stages are %s"
            % (sorted(unknown), ", ".join(
                "%d (%s)" % (s, _STAGE_NAMES[s]) for s in ALL_STAGES))
        )
    return active


def normalise(text: str, *, stages: Iterable[int] = ALL_STAGES) -> tuple[str, list[int]]:
    """Return (normalised_text, index_map).

    index_map[k] is the offset in `text` of the character that produced
    normalised character k. len(index_map) == len(normalised_text), the map is
    monotone non-decreasing, and every entry is a valid index into `text`.

    Where one input character produces several output characters (U+2026
    becoming three dots) every output character maps to that one input offset.
    Where several input characters produce one output character (a base letter
    plus a combining acute composing into one precomposed letter) that output
    character maps to the offset of the first input character of the cluster.

    `stages` selects which stages run and exists so the per-stage ablation
    behind this module's docstring can be re-run. The default is every stage.
    Note that the cluster pass always applies a CANONICAL normalisation: with
    stage 1 active it is NFKC, without it it is NFC. NFC is form-independent
    and lossless, so an ablation of stage 1 isolates the compatibility fold
    rather than removing normalisation altogether. That is the semantics the
    published ablation was measured under.

    Pure: same input, same output, no side effects.
    """
    if not isinstance(text, str):
        raise TypeError("normalise() expects str, got %s" % type(text).__name__)

    active = _resolve_stages(stages)
    form = "NFKC" if S1_NFKC in active else "NFC"

    # --- S0 invisible-character strip, whole string, before any composition.
    # It runs first so that an invisible sitting between a base letter and its
    # combining mark cannot break the cluster the NFKC pass is about to build.
    chars: list[str]
    idxs: list[int]
    if S0_INVISIBLE in active:
        chars = []
        idxs = []
        for i, ch in enumerate(text):
            if ch not in INVISIBLE:
                chars.append(ch)
                idxs.append(i)
    else:
        chars = list(text)
        idxs = list(range(len(text)))

    # --- S1 normalisation, cluster-wise so the index map survives composition.
    out_chars: list[str] = []
    out_idxs: list[int] = []
    n = len(chars)
    i = 0
    while i < n:
        j = i + 1
        while j < n and _continues_cluster(chars[j], form):
            j += 1
        cluster = "".join(chars[i:j])
        folded = unicodedata.normalize(form, cluster)
        if folded == cluster:
            # Untouched, so keep the precise per-character map.
            out_chars.extend(chars[i:j])
            out_idxs.extend(idxs[i:j])
        else:
            start = idxs[i]
            out_chars.extend(folded)
            out_idxs.extend([start] * len(folded))
        i = j
    chars, idxs = out_chars, out_idxs

    # --- S2 punctuation fold.
    # `was_dash` runs alongside and marks each output character that S2 turned
    # into "-" from a character that is NOT a hyphen. S3 reads it, because
    # after the fold the two are the same character and the distinction cannot
    # be recovered from `chars` alone.
    was_dash: list[bool] = [False] * len(chars)
    if S2_PUNCTUATION in active:
        out_chars = []
        out_idxs = []
        out_dash: list[bool] = []
        for ch, ix in zip(chars, idxs, strict=True):
            replacement = PUNCT_FOLD.get(ch)
            if replacement is None:
                out_chars.append(ch)
                out_idxs.append(ix)
                out_dash.append(False)
            else:
                out_chars.extend(replacement)
                out_idxs.extend([ix] * len(replacement))
                out_dash.extend(
                    [ch in NOT_A_LINE_BREAK_HYPHEN] * len(replacement)
                )
        chars, idxs, was_dash = out_chars, out_idxs, out_dash

    # --- S3 de-hyphenate across a SINGLE line break.
    if S3_DEHYPHENATE in active:
        out_chars = []
        out_idxs = []
        n = len(chars)
        i = 0
        while i < n:
            ch = chars[i]
            if ch == "-" and i > 0 and chars[i - 1].isalpha():
                j = i + 1
                while j < n and chars[j] in _H_SPACE:
                    j += 1
                if j < n and chars[j] == "\n":
                    j += 1
                    # Only horizontal space may follow. A second newline is
                    # not in _H_SPACE, so a blank line stops the scan here and
                    # the hyphen is kept.
                    while j < n and chars[j] in _H_SPACE:
                        j += 1
                    if j < n and chars[j].islower():
                        if was_dash[i]:
                            # An em dash, en dash or minus sign that happens to
                            # sit at a line end. It is punctuation between two
                            # whole words, never a syllable break, so the break
                            # goes and the dash stays. Deleting it here fused
                            # "achieve" and "in" into a word no paper contains.
                            out_chars.append(ch)
                            out_idxs.append(idxs[i])
                            i = j
                            continue
                        i = j  # drop the hyphen and the whole break run
                        continue
            out_chars.append(ch)
            out_idxs.append(idxs[i])
            i += 1
        chars, idxs = out_chars, out_idxs

    # --- S4 whitespace collapse and strip.
    if S4_WHITESPACE in active:
        out_chars = []
        out_idxs = []
        prev_ws = True  # seeded True so leading whitespace is suppressed
        for ch, ix in zip(chars, idxs, strict=True):
            if ch.isspace():
                if not prev_ws:
                    out_chars.append(" ")
                    out_idxs.append(ix)
                    prev_ws = True
            else:
                out_chars.append(ch)
                out_idxs.append(ix)
                prev_ws = False
        while out_chars and out_chars[-1] == " ":
            out_chars.pop()
            out_idxs.pop()
        chars, idxs = out_chars, out_idxs

    return "".join(chars), idxs


def normalise_text(text: str, *, stages: Iterable[int] = ALL_STAGES) -> str:
    """`normalise()` without the index map. Convenience for tests and studies."""
    return normalise(text, stages=stages)[0]


# ---------------------------------------------------------------------------
# T2 relaxed match form
# ---------------------------------------------------------------------------
def match_fold(norm_text: str, *, hyphen_fold: bool = True,
               case_fold: bool = True) -> tuple[str, list[int]]:
    """Apply the T2 fold to text that is ALREADY normalised.

    Returns (match_text, index_map) where index_map indexes into `norm_text`.
    Callers that already hold a normalised haystack and its map into the raw
    source use this and compose the two maps themselves. Callers holding raw
    text use `match_form_with_map()`.

    The flags exist for the same reason `stages=` does: so the two components
    of the fold can be ablated separately.
    """
    if not isinstance(norm_text, str):
        raise TypeError("match_fold() expects str, got %s" % type(norm_text).__name__)

    out_chars: list[str] = []
    out_idxs: list[int] = []
    n = len(norm_text)
    i = 0
    while i < n:
        ch = norm_text[i]
        if (hyphen_fold and ch == "-" and 0 < i < n - 1
                and norm_text[i - 1].isalpha() and norm_text[i + 1].isalpha()):
            i += 1  # an intra-word hyphen, deleted on both sides of a match
            continue
        if (hyphen_fold and ch == "-" and i > 0 and i + 2 < n
                and norm_text[i - 1].isalpha() and norm_text[i + 1] == " "
                and norm_text[i + 2].isalpha()):
            # The same intra-word hyphen with the line break still beside it.
            # A caller who extracted the PDF themselves, or copied from a
            # viewer, hands over "action- able" where a reading-order
            # extraction holds "actionable". That is the extractor's damage in
            # the caller's copy rather than in this corpus, which is the same
            # damage T2 exists for, seen from the other side. Deleting both
            # characters on BOTH sides of the comparison is the same safe
            # direction as the rule above: it cannot decide wrongly, where a
            # rule that chose whether to keep the hyphen errs on 1.27% of
            # labelled sites.
            #
            # Measured on 240 sentences lifted from an INDEPENDENT extraction
            # of 12 corpus PDFs, so that no quote was copied out of the text
            # this server holds: 31 of them, 12.9%, differed from the document
            # by nothing but this, and each came back as a located passage
            # with a character diff rather than as character identity.
            i += 2
            continue
        if case_fold:
            # Per character, so the map stays exact. Casefold can lengthen
            # (U+00DF becomes "ss"), which the map handles the same way S2
            # handles an ellipsis.
            folded = ch.casefold()
            out_chars.extend(folded)
            out_idxs.extend([i] * len(folded))
        else:
            out_chars.append(ch)
            out_idxs.append(i)
        i += 1
    return "".join(out_chars), out_idxs


def match_form(text: str) -> str:
    """The T2 relaxed comparison form.

    `normalise()`, then delete intra-word hyphens and casefold. Used only for
    character-identity matching. It is never displayed and never stored as a
    quote: an offset found in this form is mapped back to a raw source span
    first, via `match_form_with_map()`.
    """
    return match_fold(normalise(text)[0])[0]


def match_form_with_map(text: str) -> tuple[str, list[int]]:
    """`match_form()` plus an index map back into the ORIGINAL `text`.

    index_map[k] is the offset in `text` of the character that produced match
    character k, composed through both the normaliser and the T2 fold. Same
    invariants as `normalise()`: length matches, monotone non-decreasing, every
    entry a valid index into `text`.
    """
    norm, norm_map = normalise(text)
    match, match_map = match_fold(norm)
    return match, [norm_map[k] for k in match_map]
