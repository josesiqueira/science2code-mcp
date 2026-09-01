# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches a first release.

## [Unreleased]

Nothing has been published to an index and no version has been tagged, so
everything below is unreleased. `pyproject.toml` and `science2code.__version__`
already read `0.1.0`: that is the version this code WOULD be released as, not a
claim that it was. This section records what has landed in the repository, not
what is planned.

### Added

- Normalisation and the four-tier anchoring ladder (`science2code.normalise`,
  `science2code.anchor`), stdlib only, so a stored anchor stays readable on a
  bare Python.
- Response envelopes with a closed outcome vocabulary
  (`science2code.envelope`). A boolean is rejected anywhere in a response,
  because the ladder has four outcomes and every projection of four onto two
  merges outcomes that must not be merged.
- PDF extraction via `pdftotext` in reading order (`science2code.extract`).
- The corpus: a directory of PDFs plus a generated `manifest.json`
  (`science2code.corpus`), with a write helper that cannot truncate an existing
  file.
- Command line: `science2code index` and `science2code status`. There is no
  `serve` subcommand: the MCP server is `python -m science2code.server` and
  takes its corpus directory from `SCIENCE2CODE_CORPUS`.
- README and quickstart documentation.
- Continuous integration: lint and tests on Python 3.10 to 3.13, with
  `poppler-utils` installed so extraction runs for real, plus a house-style check
  that rejects em dashes and en dashes.
- `tests/test_mcp_protocol.py`: the MCP layer tested as a server rather than as
  a library. It spawns `python -m science2code.server` and speaks hand written
  JSON-RPC to its standard input, so the tool list, the descriptions, the
  annotations, the ordering and the shape of every response are checked in the
  bytes a client actually reads. `char_interval: null` is checked there because
  that is the only place a serialiser that dropped null keys could be caught.
  Every test in it skips cleanly when fastmcp is absent, and CI now installs the
  `mcp` extra so they run.
- Contributor guide.

### Changed

- Corpus membership is a directory of PDFs plus a `manifest.json` beside them.
  An earlier design made a tag in a reference manager the only source of truth.
  That was reversed after an audit found one real user's reference library was
  72 percent disjoint from the corpus actually being worked with, which would
  have given the tool's first user the worst onboarding of anyone. A reference
  manager becomes an optional adapter later, not a dependency now.
- PDF extraction uses `pdftotext` in reading order. `-layout` is rejected on
  measurement: over a 44-paper census and 879 quote attempts, page-correct
  locatability was 30.5 percent under `-layout` against 80.7 percent under
  reading order, and on two-column papers 1.6 percent against 83.0 percent.
  That census ran on a corpus that cannot be redistributed and left no script
  in this repository, so it cannot be reproduced from here and the figures are
  indicative rather than independently checkable.

### Fixed

- Every response that names a location now carries `citation_markers`, two
  counts: markers among the located characters, and markers in the 64
  characters either side. A paper writing "eating oranges is good [12]" gives
  a character-identity outcome in which every field is true and whose claim
  belongs to reference 12, and the worst case said nothing at all: a quote
  stopping one character before the marker looked clean. Measured on a
  45-document corpus over 449 located body sentences, 15.8 percent carried a
  marker inside the located characters and 26.1 percent inside or beside them.
  The marker is not resolved to the work it points at, which stays deferred.
- The normaliser deleted an em dash, en dash or minus sign that fell at a line
  break, because the punctuation fold had already turned it into a hyphen and
  the de-hyphenation stage could no longer tell the two apart. "achieve\u2014\nin
  which case" rendered as "achievein which case", a word in no paper. The line
  break still goes and the dash now stays. Measured on the same private corpus
  run that left no script here, so the counts are not reproducible from this
  repository: most affected quotes moved from the relaxed tier to exact
  identity, where the relaxed tier had been blaming the extractor for damage
  the normaliser did. `NORMALISER_VERSION` is `norm/1.1.0`.
- The T2 relaxed form now also deletes a hyphen that still has its line break
  beside it, "action- able", which is what a caller who ran their own
  extractor or copied from a viewer hands over. Measured on sentences lifted
  from an independent extraction of corpus PDFs, so that no quote was copied
  out of the text this server holds: character identity rose materially, with a
  small refusal rate and a remainder of genuine differences the independent
  rendering introduced. The measurement script is not part of this repository,
  so the exact rates are not reproducible here. `MATCHFORM_VERSION` is
  `match/1.1.0`.
- The claim that every threshold from 0.55 to 0.72 gives identical outcomes
  was too strong and is now scoped to the perturbation set it was measured on.
  A title of a paper the corpus does not hold scored 0.610 against an
  unrelated window: NOT_LOCATABLE at the default, a located passage at 0.55.
- The declared MCP dependency floor was wrong and the server it advertised
  could not start. `fastmcp>=2.0` allowed every release from 2.0.0 to 2.2.6, on
  which `build_server()` raises `TypeError: FastMCP.tool() got an unexpected
  keyword argument 'annotations'`, because the tool decorator grew
  `annotations` in 2.2.7. The floor is now 2.10, measured by installing each
  release and calling `build_server()` against it: 2.2.7 for `annotations`,
  2.5.0 for `mask_error_details`, 2.10.0 for `version`.
- The MCP handshake reported fastmcp's own version as this server's, so a
  client that asked which science2code it was talking to was told "3.4.7".
  `build_server()` now passes `version`, and `serverInfo.version` agrees with
  `provenance.server`.
- An unhandled exception inside a tool had its own message forwarded to the
  client. Measured on the wire, an `OSError` raised inside `verify_quote` put an
  absolute local path in the client's content block. `build_server()` now sets
  `mask_error_details`, so the detail stays in the server's stderr where the
  operator can see it and a transcript cannot. Argument validation errors are
  unaffected and still name the offending argument.
- A misconfigured `SCIENCE2CODE_CORPUS` escaped the closed vocabulary.
  `_open_corpus` resolved the root before entering its guard, and
  `Path("~nosuchuser/papers").expanduser()` raises, so the exception left the
  tool and reached the client as a protocol-level error carrying a bare string
  rather than as `CORPUS_UNAVAILABLE`. Resolution is now inside the guard.
- The staleness absence said "disagree in 1 places".
- `find_passage` and `verify_quote` no longer discard the first page of a text
  file that has no extraction header. The header boundary was inferred by
  counting form feeds, and `pdftotext` writes one after the last page too, so a
  hand made dump counted the same as a headed document and its whole first page
  was suppressed as metadata. The header sentinel decides it now, which is the
  same test `extract.split_document` applies.
- `VERIFIER_VERSION` gained the fingerprint backstop that `NORMALISER_VERSION`
  already had. Both are hand maintained literals that fail silently, and the
  verifier decides which tier a quote reached, so a behaviour change nobody
  version bumped left every later record reading FRESH for ever. Anchor records
  now carry `verifier_fingerprint`, and `record_status` checks it.
- `Corpus.body()` returns a paper's own characters with the extraction header
  removed, and `Corpus.text()` says in its docstring why matching against the
  header is the wrong-author error this project exists to prevent. `text()`
  behaviour is unchanged.
- `ruff check src tests`, which CI runs and `CONTRIBUTING.md` tells you to run,
  passed for the first time. It reported 306 findings on the previous commit.
  The line length is now 100, which is what the code is written at; f-string
  conversion is not enforced; every remaining finding was fixed.
- Documentation corrections, each checked by running the code: the MCP
  registration in `docs/QUICKSTART.md` named a `science2code serve` subcommand
  that does not exist; the `verify_quote` example in `README.md` used
  `your_text`, which is a response field, where the argument is `text`; the
  `find_passage` example used `paper_id` where the argument is `paper_ids` and
  takes a list; the worked `find_passage` example queried "superseded", which is
  ten characters and comes back `NOT_LOCATABLE` under the twelve character
  floor; `char_diff` was described in wdiff notation and is not; no document
  mentioned that the server needs the `mcp` extra installed.
- Docstrings that pointed at `refs_anchor.py`, a file in another repository,
  now point at `src/science2code/anchor.py`, which holds the same logic and is
  here.

### Removed

- Two anchoring numbers were withdrawn from the design rationale on 2026-08-29
  because they appeared nowhere in the repository and no run producing them was
  recorded. Under this project's own rule an unrecorded measurement cannot be
  cited.

### Deferred, and not built

Listed so that nobody mistakes them for shipped behaviour.

- Structured PDF extractors (GROBID, MinerU). Evaluated and deferred to a later
  phase, which is why there is no Docker image and no model download.
- Any adapter that reads design decisions out of a project's own documents,
  including Architecture Decision Records. Not implemented.
- A reference manager adapter for building a corpus.
- OCR for PDFs with no text layer. The tool reports the boundary rather than
  crossing it.
