# Contributing

Thanks for looking. This is a small, deliberately narrow tool, and the narrowness
is the design rather than a stage it will grow out of. The fastest way to have a
contribution accepted is to know what is in scope before you start.

## What is in scope

- Correctness of extraction and anchoring.
- New **adapters** that fill a corpus folder from another source, following
  the read-only, folder-as-truth pattern of the Zotero connector in
  `src/science2code/connectors/`.
- Portability: Windows and macOS paths, `pdftotext` version differences, PDFs
  that break the extractor.
- Making a refusal more legible. A `NOT_LOCATABLE` or `SOURCE_NOT_HELD` response
  that does not explain itself is a bug.
- Documentation that is wrong, stale, or assumes knowledge a newcomer lacks.

## What is out of scope, permanently

- **Anything that decides whether a passage is evidence for a claim.** Trained
  human annotators agree on that at Krippendorff alpha .69 to .79, and the best
  automatic judge in AttributionBench reaches about 80 percent macro-F1. A patch
  that adds such a verdict, however hedged, will be declined. The README's
  ceiling section is the full argument. The response validator enforces this: a
  set of assertion verbs is banned from every system-generated string in an
  envelope, and a response that carries one is refused rather than logged.
- **Anything that detects omitted contrary evidence.** No automated method
  exists.
- **An LLM call inside the server.** The server is a string-processing tool. Its
  value is that its answers are reproducible and that it cannot hallucinate,
  and both properties come from there being no model in the loop.
- **Deleting, moving or rewriting a user's files.** See the invariant below.

If you think one of these should change, open an issue and argue it before
writing code. Do not open a pull request that quietly crosses one of these lines.

## The invariant

**This tool never deletes a local file.**

A PDF in the corpus directory with no entry in `manifest.json` is reported, never
removed. An empty manifest produces a report, not a wipe. Any patch that
introduces a call to `os.remove`, `Path.unlink`, `shutil.rmtree` or an equivalent
against a path outside a temporary directory will be declined without further
discussion.

Adapters hold to a stricter version of the same rule: they read a project and do
not write to it.

## Setting up

You need Python 3.10 or newer and `poppler-utils`.

```bash
sudo apt-get install poppler-utils   # or: brew install poppler
cd science2code-mcp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the checks the way CI runs them:

```bash
ruff check src tests
pytest -q
```

CI runs on Python 3.10, 3.11, 3.12 and 3.13. It installs `poppler-utils` so that
extraction tests really extract.

## Tests

- Every bug fix comes with a test that fails before it and passes after.
- Tests must not depend on a private corpus. CI has no papers in it and never
  will. If a test needs a PDF, generate it at test time. Note that `.gitignore`
  excludes `*.pdf` on purpose, because papers are paywalled or licence-limited
  and must never reach git. Committing a fixture PDF therefore needs an explicit
  exception in `.gitignore`, and that is a maintainer decision to settle in an
  issue before you write the test.
- A test must assert the tier or the outcome, never the score. Inferring
  character identity from a similarity number is the exact mistake this project
  exists to prevent, and a test that does it will teach someone to do it in
  production.

## Evidence rules

These are unusual, and they are the point of the project.

1. **Do not invent a number.** Every measurement in the documentation names the
   command or the paper it came from. If you cannot say where a number came
   from, it does not go in.
2. **Do not invent a citation.** If you are unsure a paper says what you think it
   says, either check it or leave it out. "I recall reading" is not a source.
3. **Mark what you could not check.** Write `UNVERIFIED` and leave it there. An
   honest gap is a contribution. A confident guess is a defect.
4. **Do not document a feature that does not ship.** Phase 2 work is labelled as
   not built. A reader must never have to run the code to find out whether the
   docs are describing it or wishing for it.

If your change adopts something from published work or from another project,
name the source with a venue and an identifier in your pull request, so the
provenance of the design stays on the record.

## Style

- **Never use an em dash or an en dash.** Not in code, comments, documentation,
  commit messages or pull request descriptions. Use commas, colons, parentheses,
  or separate sentences. For ranges use "to", or a plain hyphen where that is the
  convention. CI enforces this and will fail the build.
- `ruff` decides formatting arguments. Run it before pushing.
- Comments explain why, not what. A comment recording a measurement or a rejected
  alternative is worth more than one narrating the next line.

## Pull requests

- One concern per pull request.
- Say what you changed and what you verified, with the command you ran.
- If you did not verify something, say that too. Nobody is annoyed by "I could
  not test this on Windows". People are annoyed by finding that out later.
- Green CI is necessary and not sufficient. A reviewer will ask where your
  numbers came from.

## Reporting a bug

Include the `pdftotext -v` output, the Python version, the operating system, and,
if it involves a specific PDF, whether that PDF can be shared. If it cannot,
describe it: page count, one column or two, scanned or born-digital. A PDF that
breaks extraction is a valuable bug report even when the file itself cannot leave
your machine.
