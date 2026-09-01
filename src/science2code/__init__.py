"""science2code: locate a quoted passage in a scientific source document, or
say plainly that it could not be located.

The core is two stdlib-only modules and has no required runtime dependency at
all, which is deliberate. A stored anchor is only as durable as the code that
can still read it, so the code that reads it must import on a bare Python.

    normalise   the rendering an offset indexes into, plus an index map back
                to the raw source, plus the version and fingerprint strings
                that say when a stored offset has gone stale.
    anchor      the four-tier ladder, and the W3C Web Annotation record that
                stores a result.

The one thing worth knowing before calling anything here: `locate()` returns a
`Tier`, never a boolean. `Tier.T1_EXACT` and `Tier.T2_RELAXED` are the only
two that assert "verbatim". `Tier.T3_LOCATED` means the passage was found and
the quoted text DIFFERS from it, which is not a pass. `Tier.T4_NOT_LOCATABLE`
is a third outcome, neither success nor invalidation.

    from science2code import locate, Tier

    anchor = locate(document_text, quoted_passage)
    if anchor.is_verbatim:
        ...        # the quote occurs in the document, character for character
    elif anchor.tier is Tier.T3_LOCATED:
        ...        # the passage is at anchor.offset_norm; anchor.diff says how
                   # the quote differs from it. Do not call this verbatim.
    else:
        ...        # not locatable. Report it; do not paper over it.

One name to be careful with. This package re-exports the FUNCTION `normalise`,
which shadows the SUBMODULE `science2code.normalise` as an attribute. So:

    from science2code import normalise            # the function
    from science2code.normalise import normalise  # also the function
    from science2code.normalise import match_form # the submodule's contents

all behave as expected, but `import science2code.normalise as m` does NOT hand
back the module. Code that genuinely needs the module object should ask for it
by name: `sys.modules["science2code.normalise"]`, or
`importlib.import_module("science2code.normalise")`.

The other submodules of this package (the MCP server, the CLI, extraction,
corpus handling) are deliberately NOT imported here. They carry optional
dependencies, and importing this package must never require them.
"""

from .anchor import (
    SCORERS,
    STATUS_FRESH,
    STATUS_STALE,
    T_LOCATE_DEFAULT,
    T_LOCATE_EQUIVALENT_RANGE,
    VERIFIER_FINGERPRINT,
    VERIFIER_VERSION,
    W3C_ANNOTATION_CONTEXT,
    Anchor,
    PreparedText,
    Tier,
    anchor_record,
    locate,
    prepare,
    reanchor_record,
    record_is_stale,
    record_status,
)
from .normalise import (
    ALL_STAGES,
    FINGERPRINT_UNAVAILABLE,
    MATCHFORM_VERSION,
    NORMALISER_FINGERPRINT,
    NORMALISER_VERSION,
    fingerprint_file,
    fingerprint_source,
    match_fold,
    match_form,
    match_form_with_map,
    normalise,
    normalise_text,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # normalise
    "ALL_STAGES",
    "FINGERPRINT_UNAVAILABLE",
    "MATCHFORM_VERSION",
    "NORMALISER_FINGERPRINT",
    "NORMALISER_VERSION",
    "fingerprint_file",
    "fingerprint_source",
    "match_fold",
    "match_form",
    "match_form_with_map",
    "normalise",
    "normalise_text",
    # anchor
    "Anchor",
    "PreparedText",
    "SCORERS",
    "STATUS_FRESH",
    "STATUS_STALE",
    "T_LOCATE_DEFAULT",
    "T_LOCATE_EQUIVALENT_RANGE",
    "Tier",
    "VERIFIER_FINGERPRINT",
    "VERIFIER_VERSION",
    "W3C_ANNOTATION_CONTEXT",
    "anchor_record",
    "locate",
    "prepare",
    "reanchor_record",
    "record_is_stale",
    "record_status",
]
