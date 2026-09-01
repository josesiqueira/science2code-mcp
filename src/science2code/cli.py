"""Command line entry point.

Two commands, both of which take the corpus directory as their only required
argument::

    science2code index <dir>     extract every PDF, write texts and manifest.json
    science2code status <dir>    what is held, what is stale, what has no text layer

Neither command deletes anything, ever. Where a file ought to go away, the
command that would remove it is printed for a human to run and copy. The
precedent is deliberate: a tool that tidies up on your behalf eventually tidies
up something you wanted.

Exit codes: 0 clean, 1 finished but with problems worth a human's attention,
2 the command could not run at all.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from . import corpus as corpus_mod
from .corpus import (
    Corpus,
    CorpusError,
    ManifestError,
    Paper,
    WriteRefused,
    build_manifest,
    dump_manifest,
    find_pdfs,
    manifest_path,
    paper_id_from_filename,
    safe_write_text,
    unique_paper_id,
)
from .extract import (
    MIN_CHARS_PER_PAGE,
    ExtractError,
    blank_pages,
    build_document,
    describe_pages,
    extract_pages,
    extractor_version,
    measure,
    sha256_file,
)

OK = 0
PROBLEMS = 1
CANNOT_RUN = 2


def _out(line: str = "") -> None:
    print(line)


def _err(line: str) -> None:
    print(line, file=sys.stderr)


def _load_existing(root: Path) -> Corpus | None:
    """Load the manifest if there is one, so hand written metadata survives.

    A manifest that exists but does not validate stops the run. It is never
    ignored and never overwritten, because a file the tool cannot read may be
    a file the tool did not write.
    """
    if not manifest_path(root).is_file():
        return None
    return Corpus.load(root)


def _prior_index(existing: Corpus | None, root: Path) -> tuple[dict, dict, set]:
    by_path: dict = {}
    by_sha: dict = {}
    taken: set = set()
    if existing is None:
        return by_path, by_sha, taken
    for paper in existing.papers():
        by_path[str(paper.pdf_path.resolve())] = paper
        by_sha.setdefault(paper.pdf_sha256, paper)
        taken.add(paper.paper_id)
    return by_path, by_sha, taken


def cmd_index(args: argparse.Namespace) -> int:
    root = Path(args.directory).expanduser()
    if not root.is_dir():
        _err("science2code index: not a directory: {}".format(root))
        return CANNOT_RUN
    root = root.resolve()

    existing = _load_existing(root)
    by_path, by_sha, taken = _prior_index(existing, root)

    pdfs = find_pdfs(root)
    _out("science2code index: {}".format(root))
    _out("  {} PDF(s) found, extractor {}".format(len(pdfs), extractor_version()))
    if not pdfs:
        _out("  nothing to extract. Put some PDFs in this directory and run again.")

    papers: list = []
    seen_pdfs: set = set()
    claimed_texts: dict = {}
    problems = 0
    counts = {"written": 0, "unchanged": 0, "no_text": 0, "refused": 0}

    for pdf in pdfs:
        resolved = str(pdf.resolve())
        seen_pdfs.add(resolved)
        pdf_sha = sha256_file(pdf)
        prior = by_path.get(resolved) or by_sha.get(pdf_sha)
        if prior is not None and prior.paper_id not in {p.paper_id for p in papers}:
            paper_id = prior.paper_id
        else:
            paper_id = unique_paper_id(
                paper_id_from_filename(pdf), taken | {p.paper_id for p in papers}
            )
        taken.add(paper_id)

        base = Paper(
            paper_id=paper_id,
            title=prior.title if prior else None,
            authors=list(prior.authors) if prior else [],
            year=prior.year if prior else None,
            venue=prior.venue if prior else None,
            doi=prior.doi if prior else None,
            pdf_path=pdf,
            text_path=None,
            pdf_sha256=pdf_sha,
            text_sha256=None,
            pages=None,
            held=False,
            note=None,
        )

        try:
            pages = extract_pages(pdf)
        except ExtractError as exc:
            problems += 1
            _out("  FAILED         {}".format(pdf.name))
            _out("                 {}".format(exc))
            papers.append(replace(base, note="extraction failed: {}".format(exc)))
            continue

        chars, n_pages, per_page = measure(pages)
        text_path = pdf.with_suffix(".txt")

        if per_page < args.floor:
            counts["no_text"] += 1
            note = ("no text layer: {} characters per page over {} page(s), "
                    "floor is {}".format(per_page, n_pages, args.floor))
            _out("  NO TEXT LAYER  {}".format(pdf.name))
            _out("                 {}".format(note))
            _out("                 Leaving the text file absent on purpose. "
                 "Quote this paper by hand.")
            if text_path.is_file():
                problems += 1
                _out("                 WARNING: a text file for it already exists.")
                _out("                 It cannot have come from this PDF. "
                     "Check it, then remove it yourself:")
                _out("                     rm '{}'".format(text_path))
            papers.append(replace(base, pages=n_pages, held=False, note=note))
            continue

        # Two PDFs in one directory whose names differ only outside the stem,
        # say x.pdf and x.PDF, want the same sidecar. Whichever loses the race
        # must not end up pointed at the winner's text, because that text is a
        # different paper's words under this paper's identifier.
        earlier = claimed_texts.get(str(text_path))
        if earlier is not None:
            problems += 1
            note = ("sidecar path {} is already claimed by paper {}".format(
                text_path.name, earlier))
            _out("  COLLISION      {}".format(pdf.name))
            _out("                 wants {}, which paper {} already claims."
                 .format(text_path.name, earlier))
            _out("                 Two PDFs cannot share one text file. "
                 "Rename one of them yourself.")
            _out("                 Nothing was written and nothing was removed.")
            papers.append(replace(base, pages=n_pages, held=False, note=note))
            continue
        claimed_texts[str(text_path)] = paper_id

        meta = base.meta()
        try:
            document = build_document(pdf, pages, meta)
        except ExtractError as exc:
            problems += 1
            _out("  FAILED         {}".format(pdf.name))
            _out("                 {}".format(exc))
            papers.append(replace(base, pages=n_pages, note=str(exc)))
            continue

        note = None
        refused = False
        try:
            written = safe_write_text(text_path, document, overwrite=args.force)
        except WriteRefused as exc:
            problems += 1
            counts["refused"] += 1
            refused = True
            _out("  REFUSED        {}".format(text_path.name))
            for line in str(exc).splitlines():
                _out("                 {}".format(line))
            # The file on disk is left exactly as it is, and it is also not
            # adopted as this paper's text. Recording it as held would hand a
            # hand edited file, or a different paper's text, back to a caller
            # as this paper's own characters.
            _out("                 Not recording it as this paper's text. "
                 "science2code will not quote from it.")
            note = ("a text file exists at {} but differs from a fresh "
                    "extraction of this PDF, so it is not held".format(text_path.name))
        else:
            if written:
                counts["written"] += 1
                _out("  extracted      {}  ({} page(s), {} characters)".format(
                    pdf.name, n_pages, chars))
            else:
                counts["unchanged"] += 1
                _out("  unchanged      {}".format(pdf.name))
            blank = blank_pages(pages)
            if blank:
                problems += 1
                note = "no text on page(s) {} of {}".format(
                    describe_pages(blank), n_pages)
                _out("                 PARTIAL: {} of {} page(s) carry no text: {}"
                     .format(len(blank), n_pages, describe_pages(blank)))
                _out("                 They are kept as empty pages so later "
                     "page numbers stay correct. Quote those pages by hand.")

        held = (not refused) and text_path.is_file()
        papers.append(replace(
            base,
            # The path is recorded even when the file is not held, so that
            # status can name the file that is sitting there unused.
            text_path=text_path if (held or refused) else None,
            text_sha256=sha256_file(text_path) if held else None,
            pages=n_pages,
            held=held,
            note=note,
        ))

    # A manifest entry whose PDF is gone keeps its hand written metadata. It is
    # reported, never dropped on the tool's own initiative.
    if existing is not None:
        for paper in existing.papers():
            if str(paper.pdf_path.resolve()) in seen_pdfs:
                continue
            problems += 1
            _out("  MISSING PDF    {}".format(paper.paper_id))
            _out("                 recorded at {}, not found now".format(paper.pdf_path))
            _out("                 Keeping the manifest entry so its metadata "
                 "survives. Edit manifest.json yourself to drop it.")
            papers.append(paper)

    manifest = build_manifest(papers, root, extractor_version())
    path = manifest_path(root)
    try:
        changed = safe_write_text(path, dump_manifest(manifest), overwrite=True)
    except CorpusError as exc:
        _err("science2code index: {}".format(exc))
        return CANNOT_RUN

    held = sum(1 for paper in papers if paper.held)
    _out("  manifest       {} ({})".format(path, "updated" if changed else "unchanged"))
    _out("  {} paper(s): {} held, {} without text".format(
        len(papers), held, len(papers) - held))
    _out("  {} extracted, {} unchanged, {} without a text layer, {} refused".format(
        counts["written"], counts["unchanged"], counts["no_text"], counts["refused"]))
    if problems:
        _out("  {} item(s) need a human. Nothing was deleted.".format(problems))
    return PROBLEMS if problems else OK


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.directory).expanduser()
    if not root.is_dir():
        _err("science2code status: not a directory: {}".format(root))
        return CANNOT_RUN
    root = root.resolve()

    corpus = Corpus.load(root)
    papers = corpus.papers()
    held = corpus.held()
    unheld = corpus.unheld()

    _out("science2code status: {}".format(root))
    _out("  manifest {}, extractor {}".format(
        corpus_mod.MANIFEST_VERSION, corpus.extractor or "not recorded"))
    _out("  {} paper(s): {} held, {} without text".format(
        len(papers), len(held), len(unheld)))

    if unheld:
        _out()
        _out("  no text available ({}):".format(len(unheld)))
        for paper in unheld:
            _out("    {}".format(paper.paper_id))
            _out("      {}".format(paper.note or "no reason recorded"))

    # A held paper can still carry a caveat, for example pages with no text
    # layer inside a document that cleared the average. Printing the note only
    # for unheld papers would hide exactly the cases a quoter needs to know.
    flagged = [paper for paper in held if paper.note]
    if flagged:
        _out()
        _out("  held with a caveat ({}):".format(len(flagged)))
        for paper in flagged:
            _out("    {}".format(paper.paper_id))
            _out("      {}".format(paper.note))

    stale = corpus.is_stale()
    if stale:
        _out()
        _out("  stale ({}):".format(len(stale)))
        for line in stale:
            _out("    {}".format(line))

    indexed = {str(paper.pdf_path.resolve()) for paper in papers}
    unindexed = [pdf for pdf in find_pdfs(root) if str(pdf.resolve()) not in indexed]
    if unindexed:
        _out()
        _out("  in the directory but not in the manifest ({}):".format(len(unindexed)))
        for pdf in unindexed:
            _out("    {}".format(pdf.relative_to(root)))
        _out("    Run: science2code index '{}'".format(root))

    if not (unheld or flagged or stale or unindexed):
        _out()
        _out("  everything held and current.")
    _out()
    _out("  This command never deletes anything. Where a file should go, the "
         "command to remove it is printed for you to run.")
    return PROBLEMS if (stale or unindexed) else OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="science2code",
        description="Ground code development in scientific papers. "
                    "A corpus is a directory of PDFs plus a manifest.",
        epilog="This tool never deletes a file. Where one should be removed, "
               "it prints the command for you to run.",
    )
    subparsers = parser.add_subparsers(dest="command")

    index = subparsers.add_parser(
        "index", help="extract every PDF and write manifest.json")
    index.add_argument("directory", help="the corpus directory")
    index.add_argument(
        "--force", action="store_true",
        help="replace a text file that differs from a fresh extraction. "
             "Without this the differing file is left alone and reported.")
    index.add_argument(
        "--floor", type=int, default=MIN_CHARS_PER_PAGE, metavar="N",
        help="characters per page below which a PDF is treated as having no "
             "text layer (default: %(default)s)")
    index.set_defaults(func=cmd_index)

    status = subparsers.add_parser(
        "status", help="what is held, what is stale, what has no text layer")
    status.add_argument("directory", help="the corpus directory")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return CANNOT_RUN
    try:
        return args.func(args)
    except ManifestError as exc:
        _err("science2code: invalid manifest")
        _err("  {}".format(exc))
        _err("  Nothing was written. Fix that field and run again.")
        return CANNOT_RUN
    except (CorpusError, ExtractError) as exc:
        _err("science2code: {}".format(exc))
        return CANNOT_RUN
    except KeyboardInterrupt:
        _err("science2code: interrupted. Nothing was deleted.")
        return CANNOT_RUN


if __name__ == "__main__":
    sys.exit(main())
