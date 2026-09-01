"""The corpus: a directory of PDFs plus a manifest.

A corpus is a folder you point the tool at. That is the whole onboarding
story, and it is deliberate. A reference manager was considered as the source
of truth and rejected on measurement: in the audit that prompted this project,
the author's own Zotero library was 72 percent disjoint from the corpus he
actually cites from (7 of 47 matched by file hash, 12 by title, 6 of 35 by
DOI). Making a reference manager authoritative would hand every new user the
worst possible first hour. A Zotero adapter can be added later, on top, as an
optional importer.

The manifest is the source of metadata. It is a plain JSON file a human can
open and correct, and a corrected title flows straight into the next extraction
because :mod:`science2code.extract` takes its header fields from the caller and
never from the PDF.

The hard invariant, enforced structurally rather than by convention:

**This tool never deletes a file.**

There is no deletion primitive imported anywhere in this package, a test walks
the AST of every module to keep it that way, and every write goes through
:func:`safe_write_text`, which creates a new temporary file with ``O_EXCL`` and
never opens an existing path for writing. When something ought to be removed,
the tool prints the exact command and lets the human run it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .extract import PAGE_BREAK, sha256_file, split_document

#: Bump when the manifest layout changes in a way an old reader would misread.
MANIFEST_VERSION = "corpus/1"

#: The manifest file name inside the corpus directory.
MANIFEST_NAME = "manifest.json"

_SHA_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9._+-]+")


class CorpusError(RuntimeError):
    """Something is wrong with the corpus directory or its manifest."""


class ManifestError(CorpusError):
    """The manifest is invalid. The offending field is named in the message."""


class WriteRefused(CorpusError):
    """A write was refused because it would have destroyed an existing file."""


@dataclass(frozen=True)
class Paper:
    """One paper. ``held`` false means known but with no text available."""

    paper_id: str
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    pdf_path: Path
    text_path: Path | None
    pdf_sha256: str
    text_sha256: str | None
    pages: int | None
    held: bool
    note: str | None = None

    def meta(self) -> dict:
        """The metadata dictionary handed to the extractor for the header."""
        return {
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "paper_id": self.paper_id,
        }


# ---------------------------------------------------------------------------
# writing


def safe_write_text(path: Path, content: str, *, overwrite: bool = False) -> bool:
    """Write ``content`` to ``path`` without ever truncating an existing file.

    The content goes to a new temporary file in the same directory, opened with
    ``O_CREAT | O_EXCL`` so it cannot land on top of anything, and is then moved
    into place with :func:`os.replace`, which is atomic within a directory.

    Returns True if the file was written, False if the file already held
    exactly this content and was left alone. Raises :class:`WriteRefused` if
    the path exists with different content and ``overwrite`` is false. The
    refusal message carries the command a human can run to resolve it; this
    function will not run that command itself.
    """
    path = Path(path)
    parent = path.parent
    if not parent.is_dir():
        raise CorpusError("no such directory: {}".format(parent))
    if path.is_symlink():
        # A symbolic link is a file the user made, and it is one this function
        # cannot reason about: os.replace would swap the link itself for a
        # regular file, and a dangling link does not answer exists(), so the
        # differs check above would never run. Refuse the whole shape, with
        # and without overwrite, and name it.
        raise WriteRefused(
            "{} is a symbolic link, refusing to touch it.\n"
            "  This tool writes regular files only, and replacing a link is a "
            "way to lose one.\n"
            "  Look at where it points, then remove the link yourself if you "
            "want the file rebuilt:\n"
            "    rm '{}'".format(path, path))
    if path.exists():
        if not path.is_file():
            raise WriteRefused("not a regular file, refusing to touch it: {}".format(path))
        try:
            existing = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            existing = None
        if existing == content:
            return False
        if not overwrite:
            raise WriteRefused(
                "{} already exists and differs.\n"
                "  This tool never destroys a file you may have edited.\n"
                "  Inspect it, then remove it yourself if you want it rebuilt:\n"
                "    rm '{}'\n"
                "  or re-run the command with --force.".format(path, path)
            )
    tmp = parent / ".{}.{}.part".format(path.name, os.getpid())
    if tmp.exists():
        raise WriteRefused(
            "a leftover temporary file is in the way: {}\n"
            "  Remove it yourself: rm '{}'".format(tmp, tmp)
        )
    # os.open with O_EXCL and no O_TRUNC: this call cannot shorten any file
    # that already exists, it can only create a new one.
    handle = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as exc:
        raise CorpusError(
            "failed while writing {}: {}\n"
            "  The partial file was left in place on purpose. "
            "Remove it yourself: rm '{}'".format(tmp, exc, tmp)
        ) from exc
    os.replace(tmp, path)
    return True


# ---------------------------------------------------------------------------
# identifiers


def _served_path_is_contained(path: Path, root: Path) -> bool:
    """Whether a served file stays inside the corpus, checked at read time.

    Closes two windows the load-time check cannot: a sidecar turned into a
    symlink pointing outside AFTER the manifest was validated, and a hardlink
    to an outside file, which follows no symlink and so resolves under the root
    yet still exposes foreign bytes. A regular sidecar the tool wrote, the only
    kind a normal corpus holds, has one link and resolves inside, so this is
    transparent to every real corpus. Any error reading the file's metadata
    fails closed.
    """
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        if resolved != root_resolved and root_resolved not in resolved.parents:
            return False
        if path.stat().st_nlink > 1:
            return False
    except OSError:
        return False
    return True


def _hash_matches(path: Path, recorded: str | None) -> bool | None:
    """Whether the file's hash equals the recorded one.

    True if they match, False if they differ, and None if the file could not
    be read at all. A caller that wants to fail toward "stale" treats anything
    that is not True as a problem, so an unreadable file is never mistaken for
    an agreeing one and never turned into an exception that hides every other
    paper's staleness.
    """
    try:
        return sha256_file(path) == recorded
    except OSError:
        return None


def paper_id_from_filename(pdf: Path) -> str:
    """Generate an id from a filename when the user has not supplied one.

    There is no ``REF-nn`` scheme and no renumbering. The filename stem is the
    id, with only characters that would make a path or a JSON key awkward
    folded to an underscore.
    """
    stem = Path(pdf).stem.strip()
    candidate = _ID_SAFE_RE.sub("_", stem).strip("_")
    return candidate or "paper"


def unique_paper_id(candidate: str, taken: set) -> str:
    """Disambiguate a generated id against ids already in use."""
    if candidate not in taken:
        return candidate
    index = 2
    while "{}-{}".format(candidate, index) in taken:
        index += 1
    return "{}-{}".format(candidate, index)


# ---------------------------------------------------------------------------
# manifest validation, which fails closed


def _fail(where: str, message: str) -> None:
    raise ManifestError("{}: {}".format(where, message))


def _as_str(value: object, where: str, *, allow_none: bool = False) -> str | None:
    if value is None:
        if allow_none:
            return None
        _fail(where, "required, but it is missing or null")
    if not isinstance(value, str):
        _fail(where, "expected a string, found {}".format(type(value).__name__))
    text = value.strip()
    if not text:
        if allow_none:
            return None
        _fail(where, "required, but it is empty")
    return text


def _as_int(value: object, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, "expected a whole number or null, found {}".format(
            type(value).__name__))
    return int(value)


def _as_sha(value: object, where: str, *, allow_none: bool = False) -> str | None:
    text = _as_str(value, where, allow_none=allow_none)
    if text is None:
        return None
    if not _SHA_RE.match(text):
        _fail(where, "expected a 64 character lowercase hex SHA-256, found {!r}".format(
            text[:16] + ("..." if len(text) > 16 else "")))
    return text


def _as_authors(value: object, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        _fail(where, "expected a list of strings, found a single string. "
                     "Write it as [\"{}\"]".format(value[:40]))
    if not isinstance(value, list):
        _fail(where, "expected a list of strings, found {}".format(type(value).__name__))
    authors = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            _fail("{}[{}]".format(where, index),
                  "expected a string, found {}".format(type(item).__name__))
        if item.strip():
            authors.append(item.strip())
    return authors


def _as_bool(value: object, where: str, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return default
    if not isinstance(value, bool):
        _fail(where, "expected true or false, found {}".format(type(value).__name__))
    return value


def _corpus_path(
    value: str, where: str, root: Path, *, served: bool = False
) -> Path:
    """Resolve one manifest path against the corpus root, or fail closed.

    The manifest is data, not instruction. A path in it may only name a file
    inside the corpus directory, because everything downstream reads that path
    and serves what it finds as the paper's own characters. An absolute path
    or a ``..`` segment would let a hand written manifest point at any file on
    the machine and have its contents returned under a real paper's identifier,
    which is the exact failure this project exists to prevent.

    `served` is True for the text sidecar, whose CONTENTS are handed back as
    the paper's own words. A served path may not escape the corpus even through
    a symbolic link, because the whole tool rests on those characters being the
    paper's. A PDF path is only ever hashed, never served as text, and
    symlinking PDFs into a folder from a reference store is a normal way to
    build a corpus, so a PDF link that points outside is allowed.
    """
    if any(ord(ch) < 32 for ch in value):
        _fail(where, "contains a control character (NUL, newline or similar), "
                     "which no real corpus path has: {!r}".format(value))
    candidate = Path(value)
    if candidate.is_absolute() or candidate.drive or candidate.root:
        _fail(where, "must be relative to the corpus directory, found the "
                     "absolute path {!r}".format(value))
    if ".." in candidate.parts:
        _fail(where, "must stay inside the corpus directory, found {!r}".format(value))
    # A served path with no ``..`` can still escape the corpus through a
    # symbolic link: a sidecar named ``paper.txt`` that is a link to a file
    # outside the directory would have that outside file's contents served as
    # the paper's own characters. That is the manifest path-traversal defect,
    # reintroduced through a link. Resolve the target and require it to stay
    # under the resolved root. Fail closed, at load, before anything is served.
    if served:
        resolved = (root / candidate).resolve()
        root_resolved = root.resolve()
        if resolved != root_resolved and root_resolved not in resolved.parents:
            _fail(where, "resolves outside the corpus directory, which a "
                         "symbolic link can do even without a '..': {!r}"
                         .format(value))
    return root / candidate


def _paper_from_entry(entry: object, where: str, root: Path) -> Paper:
    if not isinstance(entry, dict):
        _fail(where, "expected an object, found {}".format(type(entry).__name__))
    known = {
        "paper_id", "title", "authors", "year", "venue", "doi",
        "pdf_path", "text_path", "pdf_sha256", "text_sha256",
        "pages", "held", "note",
    }
    unknown = sorted(set(entry) - known)
    if unknown:
        _fail("{}.{}".format(where, unknown[0]),
              "unknown field. Known fields are: {}".format(", ".join(sorted(known))))
    pdf_rel = _as_str(entry.get("pdf_path"), "{}.pdf_path".format(where))
    text_rel = _as_str(entry.get("text_path"), "{}.text_path".format(where),
                       allow_none=True)
    return Paper(
        paper_id=_as_str(entry.get("paper_id"), "{}.paper_id".format(where)),
        title=_as_str(entry.get("title"), "{}.title".format(where), allow_none=True),
        authors=_as_authors(entry.get("authors"), "{}.authors".format(where)),
        year=_as_int(entry.get("year"), "{}.year".format(where)),
        venue=_as_str(entry.get("venue"), "{}.venue".format(where), allow_none=True),
        doi=_as_str(entry.get("doi"), "{}.doi".format(where), allow_none=True),
        pdf_path=_corpus_path(pdf_rel, "{}.pdf_path".format(where), root),
        text_path=(_corpus_path(text_rel, "{}.text_path".format(where), root,
                                served=True)
                   if text_rel else None),
        pdf_sha256=_as_sha(entry.get("pdf_sha256"), "{}.pdf_sha256".format(where)),
        text_sha256=_as_sha(entry.get("text_sha256"), "{}.text_sha256".format(where),
                            allow_none=True),
        pages=_as_int(entry.get("pages"), "{}.pages".format(where)),
        held=_as_bool(entry.get("held"), "{}.held".format(where), default=False),
        note=_as_str(entry.get("note"), "{}.note".format(where), allow_none=True),
    )


def _relative(path: Path, root: Path) -> str:
    """The manifest form of a path: relative to the corpus root.

    The unresolved path is tried first, so that a PDF which is a symbolic link
    into someone's reference folder still records as ``paper.pdf`` rather than
    as the absolute path of its target. :func:`_corpus_path` refuses to load an
    absolute path, so a value this function cannot make relative would produce
    a manifest the tool could not read back.
    """
    path = Path(path)
    root = Path(root)
    for candidate in (path, path.resolve()):
        for base in (root, root.resolve()):
            try:
                return str(candidate.relative_to(base))
            except ValueError:
                continue
    return str(path)


def entry_from_paper(paper: Paper, root: Path) -> dict:
    """Serialise one paper for the manifest, with paths relative to the root."""
    entry = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "authors": list(paper.authors),
        "year": paper.year,
        "venue": paper.venue,
        "doi": paper.doi,
        "pdf_path": _relative(paper.pdf_path, root),
        "text_path": _relative(paper.text_path, root) if paper.text_path else None,
        "pdf_sha256": paper.pdf_sha256,
        "text_sha256": paper.text_sha256,
        "pages": paper.pages,
        "held": paper.held,
    }
    if paper.note:
        entry["note"] = paper.note
    return entry


def build_manifest(papers: list, root: Path, extractor: str | None = None) -> dict:
    """The manifest document. Deliberately free of timestamps, so that
    indexing an unchanged corpus twice produces an identical file."""
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "extractor": extractor,
        "papers": [entry_from_paper(paper, root) for paper in papers],
    }
    return manifest


def dump_manifest(manifest: dict) -> str:
    """Render the manifest as the exact text written to disk."""
    return json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


# ---------------------------------------------------------------------------
# the corpus


class Corpus:
    """A loaded corpus: the manifest, plus lookups into the text files."""

    def __init__(self, root: Path, papers: list, extractor: str | None = None) -> None:
        self.root = Path(root)
        self.extractor = extractor
        self._papers = list(papers)
        self._by_id = {paper.paper_id: paper for paper in self._papers}

    # -- loading

    @classmethod
    def load(cls, root: Path) -> Corpus:
        """Read and validate ``manifest.json``.

        Validation fails closed: on any bad field this raises
        :class:`ManifestError` naming that field, and no partial corpus is
        returned. A caller that gets an object gets a whole one.
        """
        root = Path(root)
        if not root.is_dir():
            raise CorpusError("not a directory: {}".format(root))
        path = manifest_path(root)
        if not path.is_file():
            raise CorpusError(
                "no manifest at {}.\n"
                "  Build one with: science2code index '{}'".format(path, root)
            )
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CorpusError("cannot read {}: {}".format(path, exc)) from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManifestError(
                "{} line {} column {}: not valid JSON: {}".format(
                    path.name, exc.lineno, exc.colno, exc.msg)
            ) from exc
        return cls.from_manifest(data, root)

    @classmethod
    def from_manifest(cls, data: object, root: Path) -> Corpus:
        root = Path(root)
        if not isinstance(data, dict):
            _fail(MANIFEST_NAME, "expected a JSON object at the top level, found {}"
                  .format(type(data).__name__))
        version = _as_str(data.get("manifest_version"),
                          "{}.manifest_version".format(MANIFEST_NAME))
        if version != MANIFEST_VERSION:
            _fail("{}.manifest_version".format(MANIFEST_NAME),
                  "expected {!r}, found {!r}. This manifest was written by a "
                  "different version of the tool".format(MANIFEST_VERSION, version))
        extractor = _as_str(data.get("extractor"), "{}.extractor".format(MANIFEST_NAME),
                            allow_none=True)
        entries = data.get("papers")
        if entries is None:
            _fail("{}.papers".format(MANIFEST_NAME), "required, but it is missing")
        if not isinstance(entries, list):
            _fail("{}.papers".format(MANIFEST_NAME),
                  "expected a list, found {}".format(type(entries).__name__))
        papers = []
        seen: dict = {}
        for index, entry in enumerate(entries):
            where = "{}.papers[{}]".format(MANIFEST_NAME, index)
            paper = _paper_from_entry(entry, where, root)
            if paper.paper_id in seen:
                _fail("{}.paper_id".format(where),
                      "duplicate id {!r}, already used by papers[{}]".format(
                          paper.paper_id, seen[paper.paper_id]))
            seen[paper.paper_id] = index
            papers.append(paper)
        return cls(root, papers, extractor)

    # -- lookups

    def papers(self) -> list:
        """Every paper in the manifest, held or not, in manifest order."""
        return list(self._papers)

    def get(self, paper_id: str) -> Paper | None:
        """One paper by id, or None. Ids are matched verbatim."""
        return self._by_id.get(paper_id)

    def held(self) -> list:
        """Papers whose text is available."""
        return [paper for paper in self._papers if paper.held]

    def unheld(self) -> list:
        """Papers that are known but have no text. A first class state, not a
        gap: a paper with no text layer is quotable by hand, not missing."""
        return [paper for paper in self._papers if not paper.held]

    def note(self, paper_id: str) -> str | None:
        """Why a paper is not held, when a reason was recorded."""
        paper = self.get(paper_id)
        return paper.note if paper else None

    def text(self, paper_id: str) -> str | None:
        """The whole text file, HEADER INCLUDED, or None when not held.

        Read this before matching anything against the return value. The first
        block of the file, up to the first form feed, is a metadata header this
        toolchain wrote: a title, an author list, a DOI, a checksum. Those are
        characters the PAPER DOES NOT CONTAIN. A caller that searches this
        string without excluding the header can match a title or an author list
        and report it as something the document said, which is the wrong-author
        error this project exists to prevent, committed by the tool itself. It
        has happened once already, inside this repository.

        So:

            corpus.body(pid)     the paper's own characters. Use this.
            corpus.pages(pid)    the same, split per page.
            corpus.text(pid)     header included. Use only when you want the
                                 header, for example to read it back with
                                 `science2code.extract.header_fields`.
        """
        paper = self.get(paper_id)
        if paper is None or not paper.held or paper.text_path is None:
            return None
        if not _served_path_is_contained(paper.text_path, self.root):
            # Re-checked AT READ, not only at load. The manifest was validated
            # when the corpus was loaded, but the sidecar could have been turned
            # into a symlink or a hardlink pointing outside the corpus in the
            # window since. A served path whose real location, or whose extra
            # hardlink, leaves the corpus is refused rather than read, because
            # its bytes would be handed back as the paper's own characters.
            return None
        try:
            return paper.text_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def body(self, paper_id: str) -> str | None:
        """The paper's own characters, with the extraction header removed.

        None when not held, exactly as `text` and `pages` return None. Page
        breaks are preserved, so this is `text` minus the header block and the
        form feed that terminates it, and nothing else: no stripping, no
        folding, no repair.
        """
        text = self.text(paper_id)
        if text is None:
            return None
        header, pages = split_document(text)
        if not header:
            return text
        return PAGE_BREAK.join(pages)

    def pages(self, paper_id: str) -> list | None:
        """The pages of a paper, or None when not held.

        The header is not a page. Page ``n`` of the paper, one indexed as every
        locator states it, is ``corpus.pages(pid)[n - 1]``.
        """
        text = self.text(paper_id)
        if text is None:
            return None
        return split_document(text)[1]

    def page(self, paper_id: str, number: int) -> str | None:
        """Page ``number``, one indexed, or None if there is no such page."""
        pages = self.pages(paper_id)
        if pages is None or number < 1 or number > len(pages):
            return None
        return pages[number - 1]

    # -- integrity

    def is_stale(self) -> list:
        """Every recorded hash that no longer matches the file on disk.

        Returns one message per problem, each beginning with the paper id, so
        the offender is always named. An empty list means the manifest and the
        directory agree. A file that exists but cannot be READ (a permission
        change, say) counts as a problem, not as an exception: staleness must
        fail toward "stale" so a caller is warned, never toward silence.
        """
        problems = []
        for paper in self._papers:
            pid = paper.paper_id
            if not paper.pdf_path.is_file():
                problems.append("{}: pdf missing at {}".format(pid, paper.pdf_path))
            elif _hash_matches(paper.pdf_path, paper.pdf_sha256) is not True:
                problems.append(
                    "{}: pdf_sha256 mismatch or unreadable, {} changed since "
                    "indexing".format(pid, paper.pdf_path.name))
            if paper.held:
                if paper.text_path is None:
                    problems.append("{}: held but no text_path is recorded".format(pid))
                elif not paper.text_path.is_file():
                    problems.append("{}: text missing at {}".format(pid, paper.text_path))
                elif (paper.text_sha256
                      and _hash_matches(paper.text_path, paper.text_sha256) is not True):
                    problems.append(
                        "{}: text_sha256 mismatch or unreadable, {} changed "
                        "since indexing".format(pid, paper.text_path.name))
            elif paper.text_path is not None and paper.text_path.is_file():
                problems.append(
                    "{}: recorded as not held, but a text file exists at {}".format(
                        pid, paper.text_path))
        return problems

    def stale_paper_ids(self) -> list:
        """The paper ids whose files no longer match the manifest, sorted.

        The identifiers only, with no filename or message, so a caller can be
        told WHICH document is stale through a channel that carries no unvetted
        prose. `is_stale` remains the human-readable form.
        """
        stale = set()
        for paper in self._papers:
            pid = paper.paper_id
            if not paper.pdf_path.is_file():
                stale.add(pid)
            elif _hash_matches(paper.pdf_path, paper.pdf_sha256) is not True:
                stale.add(pid)
            if paper.held:
                if paper.text_path is None or not paper.text_path.is_file():
                    stale.add(pid)
                elif (paper.text_sha256
                      and _hash_matches(paper.text_path, paper.text_sha256) is not True):
                    stale.add(pid)
            elif paper.text_path is not None and paper.text_path.is_file():
                stale.add(pid)
        return sorted(stale)

    # -- serialisation

    def to_manifest(self) -> dict:
        return build_manifest(self._papers, self.root, self.extractor)


def manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST_NAME


def find_pdfs(root: Path) -> list:
    """Every PDF under the corpus directory, sorted, hidden paths skipped."""
    root = Path(root)
    found = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() != ".pdf" or not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        found.append(path)
    return found
