"""Tests for the manifest, the corpus, and the never delete invariant.

Fixtures are inline. No test reads anyone's real paper collection, and no test
needs a data file on disk that it did not write itself.
"""

from __future__ import annotations

import ast
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from science2code import cli  # noqa: E402
from science2code.corpus import (  # noqa: E402
    MANIFEST_VERSION,
    Corpus,
    CorpusError,
    ManifestError,
    WriteRefused,
    dump_manifest,
    paper_id_from_filename,
    safe_write_text,
    unique_paper_id,
)
from science2code.extract import HEADER_SENTINEL, sha256_file  # noqa: E402

PACKAGE = ROOT / "src" / "science2code"
OWNED = ("extract.py", "corpus.py", "cli.py")

HAVE_PDFTOTEXT = shutil.which("pdftotext") is not None
needs_pdftotext = unittest.skipUnless(
    HAVE_PDFTOTEXT, "pdftotext (poppler-utils) is not installed")


# ---------------------------------------------------------------------------
# the invariant

#: Names that destroy a file, whatever object they are reached through.
_DESTRUCTIVE = frozenset({
    "remove", "unlink", "rmtree", "rmdir", "removedirs",
    "truncate", "ftruncate", "rename", "renames",
    "move", "copyfile", "copy", "copy2", "copytree", "copyfileobj",
})

#: Reachable on any receiver at all, because no benign object has them.
_DESTRUCTIVE_ANYWHERE = frozenset({
    "unlink", "rmtree", "rmdir", "removedirs", "truncate", "ftruncate",
})

#: Modules whose members are file destroyers. Aliases are resolved, so
#: ``import os as o`` does not launder ``o.remove``.
_DANGEROUS_MODULES = frozenset({"os", "shutil", "pathlib", "os.path", "subprocess"})

#: Command names that destroy a file when handed to a shell or to exec.
_DESTRUCTIVE_COMMANDS = frozenset({
    "rm", "rmdir", "unlink", "shred", "truncate", "del", "erase", "mv",
})

#: Ways to reach a name without writing it, which a guard that reads names
#: alone would never see.
_INDIRECTION = frozenset({"__import__", "eval", "exec", "compile"})


def _aliases(tree: ast.AST) -> dict:
    """Map every local name bound to a dangerous module back to that module."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _DANGEROUS_MODULES:
                    found[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module in _DANGEROUS_MODULES:
            for alias in node.names:
                found[alias.asname or alias.name] = "{}.{}".format(
                    node.module, alias.name)
    return found


def _module_of(node: ast.AST, aliases: dict) -> str | None:
    """The dangerous module a receiver expression stands for, if any.

    Aliases are resolved first, so ``import os as o`` does not launder
    ``o.remove``. A bare ``os`` still counts even with no import in view, so
    that removing the import line cannot be the move that quiets the guard.
    """
    if not isinstance(node, ast.Name):
        return None
    if node.id in aliases:
        return aliases[node.id]
    return node.id if node.id in _DANGEROUS_MODULES else None


def _string_constants(node: ast.AST) -> list:
    return [child.value for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)]


def _parse_all(paths, sources) -> list:
    parsed = []
    for path in paths:
        path = Path(path)
        parsed.append((path.name,
                       ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
    for name, source in (sources or {}).items():
        parsed.append((name, ast.parse(source, filename=name)))
    return parsed


def scan_for_destruction(paths, sources: dict | None = None) -> list:
    """Every way a module could destroy a file, as a list of located hits.

    Exposed as a function rather than living inside one test so that the guard
    can itself be tested against payloads that once slipped past it. A guard
    nobody attacks is a guard nobody has checked.
    """
    hits = []
    for name, tree in _parse_all(paths, sources):
        aliases = _aliases(tree)
        for node in ast.walk(tree):
            where = "{}:{}".format(name, getattr(node, "lineno", 0))
            if isinstance(node, ast.ImportFrom) and node.module in _DANGEROUS_MODULES:
                for alias in node.names:
                    if alias.name in _DESTRUCTIVE:
                        hits.append("{} from {} import {}".format(
                            where, node.module, alias.name))
            elif isinstance(node, ast.Attribute) and node.attr == "__dict__":
                hits.append("{} .__dict__, which reaches any name at all".format(where))
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    if func.id in _DESTRUCTIVE:
                        hits.append("{} {}()".format(where, func.id))
                    elif func.id in _INDIRECTION:
                        hits.append("{} {}(), which reaches any name at all".format(
                            where, func.id))
                    elif func.id == "getattr":
                        for value in _string_constants(node):
                            if value in _DESTRUCTIVE:
                                hits.append("{} getattr(..., {!r})".format(where, value))
                elif isinstance(func, ast.Attribute):
                    owner = _module_of(func.value, aliases)
                    if func.attr in _DESTRUCTIVE_ANYWHERE:
                        hits.append("{} .{}()".format(where, func.attr))
                    elif owner is not None and func.attr in _DESTRUCTIVE:
                        hits.append("{} {}.{}()".format(where, owner, func.attr))
                    elif func.attr == "import_module":
                        hits.append("{} import_module(), which reaches any "
                                    "name at all".format(where))
                    elif owner in {"os", "subprocess"} or func.attr in {
                            "system", "popen", "spawnv", "execv", "execvp"}:
                        # Shelling out is how a deletion leaves no Python trace.
                        if func.attr in {"system", "popen"}:
                            hits.append("{} {}.{}()".format(where, owner, func.attr))
                            continue
                        for keyword in node.keywords:
                            if keyword.arg == "shell" and getattr(
                                    keyword.value, "value", False) is True:
                                hits.append("{} a shell was requested".format(where))
                        for value in _string_constants(node):
                            head = value.strip().split()[0] if value.strip() else ""
                            if head in _DESTRUCTIVE_COMMANDS:
                                hits.append("{} runs {!r}".format(where, value))
    return sorted(set(hits))


def scan_for_truncating_writes(paths, sources: dict | None = None) -> list:
    """Every call in these modules that could open a path for writing.

    ``os.fdopen`` is exempt: it takes a descriptor rather than a path, and the
    only descriptor these modules can make comes from the guarded ``os.open``,
    which a separate test holds to O_EXCL with no O_TRUNC.
    """
    hits = []
    for name, tree in _parse_all(paths, sources):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            where = "{}:{}".format(name, node.lineno)
            attribute = isinstance(func, ast.Attribute)
            called = func.attr if attribute else getattr(func, "id", None)
            if called in {"write_text", "write_bytes"}:
                hits.append("{} .{}()".format(where, called))
                continue
            if called != "open" or (attribute and _is_os_open(func)):
                continue
            # open(path, mode) puts the mode second; path.open(mode) puts it
            # first. Reading only the second argument missed every Path form.
            position = 0 if attribute else 1
            mode = None
            given = False
            if len(node.args) > position:
                given = True
                argument = node.args[position]
                mode = argument.value if isinstance(argument, ast.Constant) else None
            for keyword in node.keywords:
                if keyword.arg == "mode":
                    given = True
                    mode = (keyword.value.value
                            if isinstance(keyword.value, ast.Constant) else None)
            if given and mode is None:
                hits.append("{} open(..., <not a literal>), which cannot be "
                            "checked".format(where))
            elif isinstance(mode, str) and any(flag in mode for flag in "wxa+"):
                hits.append("{} open(..., {!r})".format(where, mode))
    return sorted(set(hits))


def _is_os_open(func: ast.Attribute) -> bool:
    return isinstance(func.value, ast.Name) and func.value.id == "os"


class DeletionPrimitiveTests(unittest.TestCase):
    """This tool never deletes a user's file, enforced structurally.

    A rule in a document is not a mechanism. This walks the syntax tree of
    every module in the package and fails if a deletion primitive appears at
    all, so the invariant cannot be argued away later by a hurried change.
    """

    def test_no_deletion_primitives(self) -> None:
        hits = scan_for_destruction(PACKAGE.glob("*.py"))
        self.assertEqual(hits, [], "deletion primitives found: {}".format(hits))

    def test_no_truncating_open_in_owned_modules(self) -> None:
        # open(path, "w") destroys a file with no deletion primitive in sight,
        # so banning the primitives alone would not be enough.
        hits = scan_for_truncating_writes(PACKAGE / name for name in OWNED)
        self.assertEqual(hits, [], "truncating writes found: {}".format(hits))

    def test_no_low_level_open_truncates_in_any_owned_module(self) -> None:
        # The original version of this test read corpus.py alone, so an
        # os.open(p, os.O_WRONLY | os.O_TRUNC) added to extract.py or cli.py
        # passed every guard in the suite.
        found = 0
        for name in OWNED:
            path = PACKAGE / name
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "open"
                        and _module_of(node.func.value, _aliases(tree)) == "os"):
                    continue
                found += 1
                self.assertGreaterEqual(
                    len(node.args), 2,
                    "{}:{} os.open with no flag argument".format(path.name, node.lineno))
                flags = ast.unparse(node.args[1])
                self.assertIn("O_EXCL", flags)
                self.assertIn("O_CREAT", flags)
                self.assertNotIn("O_TRUNC", flags)
        self.assertTrue(found, "expected the safe writer to use os.open")

    # -- the guard has to catch what it claims to catch

    def test_the_guard_catches_a_deletion_smuggled_past_the_obvious_form(self) -> None:
        """Each of these once passed the guard. They are the regression.

        The first entry is the plain form the guard always caught; it is kept
        so that a guard which stopped working entirely cannot pass this test.
        """
        payloads = {
            "plain": "import os\nos.remove(p)\n",
            "aliased module": "import os as o\no.remove(p)\n",
            "getattr by name": "import os\ngetattr(os, 'remove')(p)\n",
            "through __dict__": "import os\nos.__dict__['remove'](p)\n",
            "dunder import": "__import__('os').remove(p)\n",
            "importlib": "import importlib\nimportlib.import_module('os').unlink(p)\n",
            "os.truncate": "import os\nos.truncate(p, 0)\n",
            "os.ftruncate": "import os\nos.ftruncate(fd, 0)\n",
            "handle.truncate": "with open(p, 'r+') as h:\n    h.truncate(0)\n",
            "shutil.move onto it": "import shutil\nshutil.move(q, p)\n",
            "shutil.copyfile onto it": "import shutil\nshutil.copyfile(q, p)\n",
            "os.rename onto it": "import os\nos.rename(q, p)\n",
            "shelling out to rm": "import subprocess\nsubprocess.run(['rm', '-rf', p])\n",
            "os.system": "import os\nos.system('rm -rf ' + p)\n",
            "a shell at all": "import subprocess\nsubprocess.run(c, shell=True)\n",
        }
        for label, source in payloads.items():
            with self.subTest(payload=label):
                self.assertNotEqual(
                    scan_for_destruction([], sources={"probe.py": source}), [],
                    "this destroys a file and the guard let it through: {!r}".format(
                        source))

    def test_the_guard_catches_a_truncating_write_on_a_path_object(self) -> None:
        """``Path(p).open("w")`` truncates, and the guard used to miss it.

        For ``open(path, "w")`` the mode is the second argument; for
        ``path.open("w")`` it is the first. The guard only ever read the
        second, so the Path form went through untouched.
        """
        payloads = {
            "Path(p).open": "from pathlib import Path\nPath(p).open('w')\n",
            "bound path object": "p.open('w')\n",
            "append mode": "p.open('a')\n",
            "update mode": "p.open('r+')\n",
            "exclusive create": "p.open('x')\n",
            "binary write": "p.open('wb')\n",
            "keyword mode": "p.open(mode='w')\n",
            "the plain form": "open(p, 'w')\n",
            "a mode that is not a literal": "open(p, mode)\n",
        }
        for label, source in payloads.items():
            with self.subTest(payload=label):
                self.assertNotEqual(
                    scan_for_truncating_writes([], sources={"probe.py": source}), [],
                    "this truncates a file and the guard let it through: {!r}".format(
                        source))

    def test_the_guard_still_passes_what_the_package_legitimately_does(self) -> None:
        allowed = {
            "reading bytes": "open(p, 'rb')\n",
            "reading text": "open(p, 'r')\n",
            "the sanctioned writer":
                "import os\n"
                "h = os.open(t, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)\n"
                "with os.fdopen(h, 'w', encoding='utf-8') as s:\n    s.write(c)\n"
                "os.replace(t, p)\n",
            "a list remove": "names.remove(x)\n",
            "getattr for something benign": "getattr(args, 'func', None)\n",
            "running pdftotext":
                "import subprocess\nsubprocess.run(['pdftotext', '-q', f, '-'])\n",
        }
        for label, source in allowed.items():
            with self.subTest(allowed=label):
                self.assertEqual(
                    scan_for_destruction([], sources={"probe.py": source}), [], label)
                self.assertEqual(
                    scan_for_truncating_writes([], sources={"probe.py": source}), [],
                    label)


# ---------------------------------------------------------------------------
# writing


class SafeWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_writes_a_new_file(self) -> None:
        target = self.root / "a.txt"
        self.assertTrue(safe_write_text(target, "hello\n"))
        self.assertEqual(target.read_text(encoding="utf-8"), "hello\n")

    def test_identical_content_is_a_no_op(self) -> None:
        target = self.root / "a.txt"
        safe_write_text(target, "hello\n")
        before = target.stat().st_mtime_ns
        self.assertFalse(safe_write_text(target, "hello\n"))
        self.assertEqual(target.stat().st_mtime_ns, before)

    def test_a_differing_file_is_never_clobbered(self) -> None:
        target = self.root / "a.txt"
        safe_write_text(target, "mine\n")
        with self.assertRaises(WriteRefused) as caught:
            safe_write_text(target, "theirs\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "mine\n")
        message = str(caught.exception)
        self.assertIn("rm '{}'".format(target), message)
        self.assertIn("never destroys", message)

    def test_overwrite_is_possible_but_must_be_asked_for(self) -> None:
        target = self.root / "a.txt"
        safe_write_text(target, "mine\n")
        self.assertTrue(safe_write_text(target, "theirs\n", overwrite=True))
        self.assertEqual(target.read_text(encoding="utf-8"), "theirs\n")

    def test_no_temporary_file_is_left_behind_on_success(self) -> None:
        safe_write_text(self.root / "a.txt", "hello\n")
        self.assertEqual([p.name for p in self.root.iterdir()], ["a.txt"])

    def test_a_missing_directory_is_a_clear_error(self) -> None:
        with self.assertRaises(CorpusError):
            safe_write_text(self.root / "nope" / "a.txt", "hello\n")

    def test_a_dangling_symlink_is_not_silently_replaced(self) -> None:
        """A link whose target is absent answers exists() with False.

        The differs check hung off exists(), so a dangling link skipped every
        guard and os.replace swapped the user's link for a regular file with
        no warning and no force asked for. A link to a drive that is not
        mounted yet is exactly that shape.
        """
        link = self.root / "a.txt"
        os.symlink(self.root / "not-there-yet.txt", link)
        with self.assertRaises(WriteRefused) as caught:
            safe_write_text(link, "theirs\n")
        self.assertTrue(link.is_symlink(), "the user's link was replaced")
        self.assertIn("symbolic link", str(caught.exception))

    def test_a_symlink_is_refused_even_when_overwrite_was_asked_for(self) -> None:
        outside = self.root / "precious.txt"
        outside.write_text("PRECIOUS\n", encoding="utf-8")
        link = self.root / "a.txt"
        os.symlink(outside, link)
        with self.assertRaises(WriteRefused):
            safe_write_text(link, "theirs\n", overwrite=True)
        self.assertTrue(link.is_symlink())
        self.assertEqual(outside.read_text(encoding="utf-8"), "PRECIOUS\n")


# ---------------------------------------------------------------------------
# identifiers


class PaperIdTests(unittest.TestCase):
    def test_generated_from_the_filename(self) -> None:
        self.assertEqual(
            paper_id_from_filename(Path("/x/Gotel Finkelstein 1994.pdf")),
            "Gotel_Finkelstein_1994")

    def test_no_ref_nn_scheme_anywhere(self) -> None:
        generated = paper_id_from_filename(Path("/x/some paper.pdf"))
        self.assertNotRegex(generated, r"^REF-\d+$")

    def test_collisions_are_disambiguated(self) -> None:
        self.assertEqual(unique_paper_id("a", set()), "a")
        self.assertEqual(unique_paper_id("a", {"a"}), "a-2")
        self.assertEqual(unique_paper_id("a", {"a", "a-2"}), "a-3")


# ---------------------------------------------------------------------------
# the manifest


def entry(**overrides) -> dict:
    base = {
        "paper_id": "some-paper",
        "title": None,
        "authors": [],
        "year": None,
        "venue": None,
        "doi": None,
        "pdf_path": "some-paper.pdf",
        "text_path": None,
        "pdf_sha256": "0" * 64,
        "text_sha256": None,
        "pages": None,
        "held": False,
    }
    base.update(overrides)
    return base


def manifest(*entries, version: str = MANIFEST_VERSION) -> dict:
    return {
        "manifest_version": version,
        "extractor": "poppler/24.02.0 reading-order",
        "papers": list(entries),
    }


class ManifestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, data) -> Path:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def assert_names(self, data, fragment: str) -> None:
        self.write(data)
        with self.assertRaises(ManifestError) as caught:
            Corpus.load(self.root)
        self.assertIn(fragment, str(caught.exception))

    def test_a_valid_manifest_loads(self) -> None:
        self.write(manifest(entry()))
        corpus = Corpus.load(self.root)
        self.assertEqual(len(corpus.papers()), 1)

    def test_wrong_version_names_the_field(self) -> None:
        self.assert_names(manifest(entry(), version="corpus/99"), "manifest_version")

    def test_missing_version_names_the_field(self) -> None:
        data = manifest(entry())
        del data["manifest_version"]
        self.assert_names(data, "manifest_version")

    def test_papers_must_be_a_list(self) -> None:
        data = manifest()
        data["papers"] = {"a": 1}
        self.assert_names(data, "papers")

    def test_a_missing_paper_id_names_the_paper(self) -> None:
        bad = entry()
        del bad["paper_id"]
        self.assert_names(manifest(bad), "papers[0].paper_id")

    def test_a_missing_pdf_path_names_the_paper(self) -> None:
        bad = entry()
        del bad["pdf_path"]
        self.assert_names(manifest(bad), "papers[0].pdf_path")

    def test_a_bad_sha_names_the_paper(self) -> None:
        self.assert_names(manifest(entry(pdf_sha256="nope")), "papers[0].pdf_sha256")

    def test_a_bad_year_names_the_paper(self) -> None:
        self.assert_names(manifest(entry(year="1994")), "papers[0].year")

    def test_authors_as_a_bare_string_names_the_paper(self) -> None:
        self.assert_names(manifest(entry(authors="A. Author")), "papers[0].authors")

    def test_a_duplicate_id_names_both_positions(self) -> None:
        self.assert_names(
            manifest(entry(), entry(pdf_path="other.pdf")), "duplicate id")

    def test_an_unknown_field_is_named_rather_than_ignored(self) -> None:
        self.assert_names(manifest(entry(colour="blue")), "papers[0].colour")

    def test_broken_json_names_the_line(self) -> None:
        (self.root / "manifest.json").write_text("{ nope", encoding="utf-8")
        with self.assertRaises(ManifestError) as caught:
            Corpus.load(self.root)
        self.assertIn("not valid JSON", str(caught.exception))

    def test_no_manifest_says_how_to_make_one(self) -> None:
        with self.assertRaises(CorpusError) as caught:
            Corpus.load(self.root)
        self.assertIn("science2code index", str(caught.exception))

    def test_validation_fails_closed_with_no_partial_corpus(self) -> None:
        self.write(manifest(entry(), entry(paper_id="second", pdf_sha256="nope")))
        with self.assertRaises(ManifestError):
            Corpus.load(self.root)


# ---------------------------------------------------------------------------
# the corpus


class CorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.pdf = self.root / "paper one.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 pretend\n")
        self.txt = self.root / "paper one.txt"
        document = (
            "Title: A Paper\n"
            "Paper ID: my own id #1\n"
            "Note: derived\n"
            + HEADER_SENTINEL
            + "\n\x0cpage one text\x0cpage two text"
        )
        self.txt.write_text(document, encoding="utf-8")
        self.data = manifest(entry(
            paper_id="my own id #1",
            title="A Paper",
            authors=["A. Author"],
            year=1994,
            pdf_path=self.pdf.name,
            text_path=self.txt.name,
            pdf_sha256=sha256_file(self.pdf),
            text_sha256=sha256_file(self.txt),
            pages=2,
            held=True,
        ))
        (self.root / "manifest.json").write_text(
            json.dumps(self.data), encoding="utf-8")
        self.corpus = Corpus.load(self.root)

    def test_a_user_id_is_accepted_verbatim(self) -> None:
        self.assertEqual(self.corpus.papers()[0].paper_id, "my own id #1")
        self.assertIsNotNone(self.corpus.get("my own id #1"))

    def test_get_returns_none_for_an_unknown_id(self) -> None:
        self.assertIsNone(self.corpus.get("nobody"))

    def test_text_returns_the_whole_file(self) -> None:
        self.assertIn("page one text", self.corpus.text("my own id #1"))

    def test_pages_excludes_the_header(self) -> None:
        pages = self.corpus.pages("my own id #1")
        self.assertEqual(pages, ["page one text", "page two text"])

    def test_body_excludes_the_header_and_keeps_the_page_breaks(self) -> None:
        body = self.corpus.body("my own id #1")
        self.assertEqual(body, "page one text\fpage two text")

    def test_the_header_is_in_text_and_out_of_body(self) -> None:
        # The load bearing distinction. The header holds a title and an author
        # list this toolchain wrote, so anything that matches against text()
        # can match characters the paper does not contain and report them as
        # the paper's. body() is the safe default and text() is the exception.
        text = self.corpus.text("my own id #1")
        body = self.corpus.body("my own id #1")
        self.assertIn(HEADER_SENTINEL, text)
        self.assertNotIn(HEADER_SENTINEL, body)
        self.assertTrue(text.endswith(body))

    def test_body_of_a_headerless_file_is_the_whole_file(self) -> None:
        # A user who ran pdftotext by hand has no header. Nothing is stripped
        # from such a file, including its first page.
        path = self.root / "handmade.txt"
        path.write_text("page one\fpage two\f", encoding="utf-8")
        data = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        entry = dict(data["papers"][0])
        entry["paper_id"] = "handmade"
        entry["text_path"] = "handmade.txt"
        entry["text_sha256"] = None
        data["papers"] = data["papers"] + [entry]
        (self.root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
        corpus = Corpus.load(self.root)
        self.assertEqual(corpus.body("handmade"), "page one\fpage two\f")

    def test_page_is_one_indexed_like_a_locator(self) -> None:
        self.assertEqual(self.corpus.page("my own id #1", 1), "page one text")
        self.assertEqual(self.corpus.page("my own id #1", 2), "page two text")
        self.assertIsNone(self.corpus.page("my own id #1", 3))
        self.assertIsNone(self.corpus.page("my own id #1", 0))

    def test_paths_resolve_against_the_corpus_root(self) -> None:
        self.assertEqual(self.corpus.papers()[0].pdf_path, self.pdf)

    def test_a_fresh_corpus_is_not_stale(self) -> None:
        self.assertEqual(self.corpus.is_stale(), [])

    def test_a_changed_pdf_is_named(self) -> None:
        self.pdf.write_bytes(b"%PDF-1.4 different\n")
        stale = Corpus.load(self.root).is_stale()
        self.assertEqual(len(stale), 1)
        self.assertTrue(stale[0].startswith("my own id #1:"))
        self.assertIn("pdf_sha256 mismatch", stale[0])

    def test_a_changed_text_file_is_named(self) -> None:
        self.txt.write_text("edited", encoding="utf-8")
        stale = Corpus.load(self.root).is_stale()
        self.assertTrue(any("text_sha256 mismatch" in line for line in stale))

    def test_a_missing_pdf_is_named(self) -> None:
        self.data["papers"][0]["pdf_path"] = "gone.pdf"
        (self.root / "manifest.json").write_text(
            json.dumps(self.data), encoding="utf-8")
        stale = Corpus.load(self.root).is_stale()
        self.assertTrue(any("pdf missing" in line for line in stale))

    def test_manifest_round_trips(self) -> None:
        rebuilt = Corpus.from_manifest(self.corpus.to_manifest(), self.root)
        self.assertEqual(
            [p.paper_id for p in rebuilt.papers()],
            [p.paper_id for p in self.corpus.papers()])

    def test_dump_is_stable(self) -> None:
        once = dump_manifest(self.corpus.to_manifest())
        twice = dump_manifest(Corpus.load(self.root).to_manifest())
        self.assertEqual(once, twice)


class UnheldTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        pdf = self.root / "scan.pdf"
        pdf.write_bytes(b"%PDF-1.4 pretend\n")
        data = manifest(entry(
            paper_id="scan",
            pdf_path="scan.pdf",
            pdf_sha256=sha256_file(pdf),
            pages=31,
            held=False,
            note="no text layer: 1 characters per page over 31 page(s), floor is 200",
        ))
        (self.root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
        self.corpus = Corpus.load(self.root)

    def test_not_held_is_a_first_class_state(self) -> None:
        self.assertEqual([p.paper_id for p in self.corpus.unheld()], ["scan"])
        self.assertEqual(self.corpus.held(), [])

    def test_the_reason_is_reportable(self) -> None:
        self.assertIn("no text layer", self.corpus.note("scan"))

    def test_text_and_pages_are_none_not_empty(self) -> None:
        self.assertIsNone(self.corpus.text("scan"))
        self.assertIsNone(self.corpus.pages("scan"))
        self.assertIsNone(self.corpus.body("scan"))

    def test_a_text_file_that_should_not_exist_is_reported(self) -> None:
        # A sidecar for a PDF with no text layer cannot have come from that
        # PDF. The tool reports it and prints the command; it never removes it.
        self.assertEqual(self.corpus.is_stale(), [])
        data = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        data["papers"][0]["text_path"] = "scan.txt"
        (self.root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
        (self.root / "scan.txt").write_text("junk", encoding="utf-8")
        stale = Corpus.load(self.root).is_stale()
        self.assertTrue(any("not held, but a text file exists" in line for line in stale))
        self.assertTrue((self.root / "scan.txt").exists())


# ---------------------------------------------------------------------------
# the command line


def run_cli(argv: list) -> tuple:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


def make_pdf(path: Path, pages: list) -> Path:
    """A minimal one font PDF, hand assembled with the standard library."""
    def escape(text: str) -> str:
        for old, new in (("\\", "\\\\"), ("(", "\\("), (")", "\\)")):
            text = text.replace(old, new)
        return text

    n_pages = len(pages)
    font_num = 3 + 2 * n_pages
    kids = " ".join("{} 0 R".format(3 + 2 * i) for i in range(n_pages))
    bodies = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(kids, n_pages),
    ]
    for index, lines in enumerate(pages):
        bodies.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 {} 0 R >> >> /Contents {} 0 R >>".format(
                font_num, 4 + 2 * index))
        drawn = "\n".join("({}) Tj T*".format(escape(line)) for line in lines)
        stream = "BT /F1 12 Tf 72 720 Td 14 TL\n{}\nET".format(drawn)
        bodies.append("<< /Length {} >>\nstream\n{}\nendstream".format(
            len(stream), stream))
    bodies.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += "{} 0 obj\n{}\nendobj\n".format(number, body).encode("latin-1")
    xref_at = len(out)
    out += "xref\n0 {}\n".format(len(bodies) + 1).encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += "{:010d} 00000 n \n".format(offset).encode("latin-1")
    out += "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n".format(
        len(bodies) + 1, xref_at).encode("latin-1")
    path.write_bytes(bytes(out))
    return path


def body_lines(marker: str, count: int = 40) -> list:
    return ["{} line {:02d}: the quick brown fox jumps over the lazy dog.".format(
        marker, n) for n in range(count)]


@needs_pdftotext
class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        make_pdf(self.root / "good.pdf", [body_lines("alpha"), body_lines("beta")])
        make_pdf(self.root / "scan.pdf", [["hi"]])

    def test_index_then_status(self) -> None:
        code, out, _ = run_cli(["index", str(self.root)])
        self.assertEqual(code, 0, out)
        self.assertIn("extracted", out)
        self.assertIn("NO TEXT LAYER", out)
        self.assertTrue((self.root / "good.txt").exists())
        self.assertFalse((self.root / "scan.txt").exists())

        corpus = Corpus.load(self.root)
        self.assertEqual(len(corpus.papers()), 2)
        self.assertEqual([p.paper_id for p in corpus.held()], ["good"])
        self.assertEqual([p.paper_id for p in corpus.unheld()], ["scan"])
        self.assertEqual(corpus.pages("good")[0].splitlines()[0].split()[0], "alpha")

        code, out, _ = run_cli(["status", str(self.root)])
        self.assertEqual(code, 0, out)
        self.assertIn("no text available", out)
        self.assertIn("never deletes", out)

    def test_index_is_idempotent(self) -> None:
        run_cli(["index", str(self.root)])
        first = (self.root / "manifest.json").read_text(encoding="utf-8")
        code, out, _ = run_cli(["index", str(self.root)])
        self.assertEqual(code, 0, out)
        self.assertIn("unchanged", out)
        self.assertEqual((self.root / "manifest.json").read_text(encoding="utf-8"), first)

    def test_hand_edited_metadata_survives_reindexing(self) -> None:
        run_cli(["index", str(self.root)])
        path = self.root / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data["papers"]:
            if record["paper_id"] == "good":
                record["paper_id"] = "author-1994"
                record["title"] = "A Real Title"
                record["authors"] = ["A. Author"]
        path.write_text(json.dumps(data), encoding="utf-8")
        code, out, _ = run_cli(["index", str(self.root), "--force"])
        self.assertEqual(code, 0, out)
        header = (self.root / "good.txt").read_text(encoding="utf-8").split("\x0c")[0]
        self.assertIn("Title: A Real Title", header)
        self.assertIn("Paper ID: author-1994", header)

    def test_an_edited_text_file_is_reported_never_clobbered(self) -> None:
        run_cli(["index", str(self.root)])
        target = self.root / "good.txt"
        target.write_text("something a human wrote\n", encoding="utf-8")
        code, out, _ = run_cli(["index", str(self.root)])
        self.assertEqual(code, 1)
        self.assertIn("REFUSED", out)
        self.assertIn("rm '{}'".format(target), out)
        self.assertEqual(target.read_text(encoding="utf-8"), "something a human wrote\n")

    def test_a_stray_sidecar_for_a_scan_is_reported_never_removed(self) -> None:
        stray = self.root / "scan.txt"
        stray.write_text("junk that cannot have come from that pdf\n", encoding="utf-8")
        code, out, _ = run_cli(["index", str(self.root)])
        self.assertEqual(code, 1)
        self.assertIn("rm '{}'".format(stray), out)
        self.assertTrue(stray.exists())

    def test_a_missing_pdf_keeps_its_manifest_entry(self) -> None:
        run_cli(["index", str(self.root)])
        data = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        data["papers"].append(entry(paper_id="vanished", pdf_path="vanished.pdf",
                                    title="Kept By Hand"))
        (self.root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
        code, out, _ = run_cli(["index", str(self.root)])
        self.assertEqual(code, 1)
        self.assertIn("MISSING PDF", out)
        kept = Corpus.load(self.root).get("vanished")
        self.assertIsNotNone(kept)
        self.assertEqual(kept.title, "Kept By Hand")

    def test_an_invalid_manifest_stops_the_run_and_writes_nothing(self) -> None:
        run_cli(["index", str(self.root)])
        path = self.root / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["papers"][0]["pdf_sha256"] = "not a hash"
        broken = json.dumps(data)
        path.write_text(broken, encoding="utf-8")
        code, _, err = run_cli(["index", str(self.root)])
        self.assertEqual(code, 2)
        self.assertIn("pdf_sha256", err)
        self.assertIn("Nothing was written", err)
        self.assertEqual(path.read_text(encoding="utf-8"), broken)

    def test_status_on_a_directory_with_no_manifest(self) -> None:
        code, _, err = run_cli(["status", str(self.root)])
        self.assertEqual(code, 2)
        self.assertIn("science2code index", err)

    def test_a_directory_that_is_not_there(self) -> None:
        code, _, err = run_cli(["index", str(self.root / "nope")])
        self.assertEqual(code, 2)
        self.assertIn("not a directory", err)


# ---------------------------------------------------------------------------
# a manifest is data, and data does not get to name a file outside the corpus


class ManifestPathTraversalTests(unittest.TestCase):
    """A hand written manifest could point a paper at any file on the machine.

    ``root / "../../etc/passwd"`` and ``root / "/etc/passwd"`` both land
    outside the corpus, and :meth:`Corpus.text` read and returned whatever it
    found there as that paper's own characters, with ``is_stale`` reporting
    the corpus clean. That is the server answering a quote check with a file
    the corpus does not hold.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.root = self.base / "corpus"
        self.root.mkdir()
        self.outside = self.base / "outside.txt"
        self.outside.write_text("NOT A PAPER AT ALL\n", encoding="utf-8")
        (self.root / "some-paper.pdf").write_bytes(b"%PDF-1.4 pretend\n")

    def load(self, **overrides):
        data = manifest(entry(**overrides))
        (self.root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
        return Corpus.load(self.root)

    def assert_refused(self, field: str, value: str) -> None:
        with self.assertRaises(ManifestError) as caught:
            self.load(**{field: value})
        message = str(caught.exception)
        self.assertIn("papers[0].{}".format(field), message)
        self.assertIn("corpus directory", message)

    def test_a_text_path_climbing_out_with_dot_dot_is_refused(self) -> None:
        self.assert_refused("text_path", "../outside.txt")

    def test_a_text_path_climbing_out_mid_path_is_refused(self) -> None:
        self.assert_refused("text_path", "sub/../../outside.txt")

    def test_an_absolute_text_path_is_refused(self) -> None:
        self.assert_refused("text_path", str(self.outside))

    def test_a_pdf_path_climbing_out_is_refused(self) -> None:
        self.assert_refused("pdf_path", "../outside.pdf")

    def test_an_absolute_pdf_path_is_refused(self) -> None:
        self.assert_refused("pdf_path", "/etc/hostname")

    def test_nothing_outside_the_corpus_is_ever_served_as_a_paper(self) -> None:
        for value in ("../outside.txt", str(self.outside), "/etc/hostname"):
            with self.subTest(text_path=value):
                with self.assertRaises(ManifestError):
                    self.load(text_path=value, held=True)

    def test_an_ordinary_relative_path_still_loads(self) -> None:
        corpus = self.load(pdf_path="sub/some-paper.pdf")
        self.assertEqual(corpus.papers()[0].pdf_path,
                         self.root / "sub" / "some-paper.pdf")


@needs_pdftotext
class SymlinkedCorpusRoundTripTests(unittest.TestCase):
    """Confining manifest paths must not break a corpus of symbolic links.

    Symlinking papers into a folder is a normal way to build one. The manifest
    has to record such a PDF by its name inside the corpus rather than by its
    target, or the tool would write a manifest that its own loader refuses.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.store = self.base / "store"
        self.store.mkdir()
        self.root = self.base / "corpus"
        self.root.mkdir()
        make_pdf(self.store / "real.pdf", [body_lines("alpha"), body_lines("beta")])
        os.symlink(self.store / "real.pdf", self.root / "linked.pdf")

    def test_a_symlinked_pdf_records_a_path_the_loader_accepts(self) -> None:
        code, out, _ = run_cli(["index", str(self.root)])
        self.assertEqual(code, 0, out)
        data = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(data["papers"][0]["pdf_path"], "linked.pdf")
        corpus = Corpus.load(self.root)
        self.assertEqual([p.paper_id for p in corpus.held()], ["linked"])
        self.assertEqual(corpus.is_stale(), [])


# ---------------------------------------------------------------------------
# a text file the tool refused to write is not that paper's text


@needs_pdftotext
class RefusedTextIsNotAdoptedTests(unittest.TestCase):
    """The refusal has to reach the manifest, not only the terminal.

    ``index`` printed REFUSED and then recorded the paper as held, with
    ``text_sha256`` taken from the file it had just refused to write. ``status``
    then reported the corpus clean and the server would have quoted that file
    as the paper's own words.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        make_pdf(self.root / "good.pdf", [body_lines("alpha"), body_lines("beta")])
        run_cli(["index", str(self.root)])
        self.target = self.root / "good.txt"
        self.fabricated = "THE PAPER SAYS: whatever I typed here.\n"
        self.target.write_text(self.fabricated, encoding="utf-8")

    def entry_now(self) -> dict:
        data = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        return data["papers"][0]

    def test_a_hand_edited_text_file_is_not_recorded_as_held(self) -> None:
        code, out, _ = run_cli(["index", str(self.root)])
        self.assertEqual(code, 1, out)
        self.assertIn("REFUSED", out)
        record = self.entry_now()
        self.assertFalse(record["held"], "a refused file was adopted as the text")
        self.assertIsNone(record["text_sha256"],
                          "the refused file's own hash was recorded as current")
        self.assertIn("not held", record["note"])

    def test_the_corpus_will_not_hand_that_text_to_a_caller(self) -> None:
        run_cli(["index", str(self.root)])
        corpus = Corpus.load(self.root)
        paper = corpus.papers()[0]
        self.assertIsNone(corpus.text(paper.paper_id))
        self.assertIsNone(corpus.pages(paper.paper_id))
        self.assertEqual([p.paper_id for p in corpus.held()], [])

    def test_status_does_not_call_the_corpus_current(self) -> None:
        run_cli(["index", str(self.root)])
        code, out, _ = run_cli(["status", str(self.root)])
        self.assertNotIn("everything held and current", out)
        self.assertIn("no text available", out)
        self.assertEqual(code, 1, out)

    def test_the_file_itself_is_still_untouched(self) -> None:
        run_cli(["index", str(self.root)])
        self.assertEqual(self.target.read_text(encoding="utf-8"), self.fabricated)

    def test_removing_the_edited_file_restores_the_paper(self) -> None:
        run_cli(["index", str(self.root)])
        self.target.rename(self.root / "kept-by-the-test")
        code, out, _ = run_cli(["index", str(self.root)])
        self.assertEqual(code, 0, out)
        record = self.entry_now()
        self.assertTrue(record["held"])
        self.assertIsNotNone(record["text_sha256"])


@needs_pdftotext
class SidecarCollisionTests(unittest.TestCase):
    """Two PDFs whose stems match want one .txt, and only one can have it.

    ``x.pdf`` and ``x.PDF`` both map to ``x.txt``. The loser was recorded as
    held against the winner's text, so a quote from one paper came back
    verbatim under the other paper's identifier with nothing reported stale.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        make_pdf(self.root / "x.pdf", [body_lines("alpha"), body_lines("beta")])
        make_pdf(self.root / "x.PDF", [body_lines("gamma"), body_lines("delta")])

    def test_only_one_paper_claims_the_text_file(self) -> None:
        code, out, _ = run_cli(["index", str(self.root)])
        self.assertEqual(code, 1, out)
        self.assertIn("COLLISION", out)
        corpus = Corpus.load(self.root)
        held = corpus.held()
        self.assertEqual(len(held), 1, "both papers claimed the same text file")

    def test_the_loser_is_not_given_the_winners_words(self) -> None:
        run_cli(["index", str(self.root)])
        corpus = Corpus.load(self.root)
        for paper in corpus.papers():
            text = corpus.text(paper.paper_id)
            if text is None:
                continue
            source = [line for line in text.split("\x0c")[0].splitlines()
                      if line.startswith("Source PDF:")]
            self.assertEqual(source, ["Source PDF: {}".format(paper.pdf_path.name)],
                             "a paper was served another paper's text")

    def test_neither_pdf_is_touched(self) -> None:
        before = {p.name: p.read_bytes() for p in self.root.glob("*.*DF")}
        run_cli(["index", str(self.root)])
        for name, data in before.items():
            self.assertEqual((self.root / name).read_bytes(), data)


# ---------------------------------------------------------------------------
# a page with no text is reported, never passed off as an empty page


@needs_pdftotext
class PartlyScannedPaperTests(unittest.TestCase):
    """The floor is an average, so a document can clear it and still be half
    unreadable. Nine empty pages behind one dense one used to be silent."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        make_pdf(self.root / "mixed.pdf",
                 [body_lines("alpha", 120)] + [[] for _ in range(9)])

    def test_index_names_the_pages_that_carry_no_text(self) -> None:
        code, out, _ = run_cli(["index", str(self.root)])
        self.assertEqual(code, 1, out)
        self.assertIn("PARTIAL", out)
        for number in range(2, 11):
            self.assertIn(str(number), out)

    def test_the_note_reaches_the_manifest_and_status(self) -> None:
        run_cli(["index", str(self.root)])
        record = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8"))["papers"][0]
        self.assertTrue(record["held"])
        self.assertIn("no text on page", record["note"])
        _, out, _ = run_cli(["status", str(self.root)])
        self.assertIn("held with a caveat", out)
        self.assertNotIn("everything held and current", out)

    def test_the_empty_pages_are_kept_so_page_numbers_stay_right(self) -> None:
        run_cli(["index", str(self.root)])
        corpus = Corpus.load(self.root)
        pages = corpus.pages("mixed")
        self.assertEqual(len(pages), 10)
        self.assertIn("alpha line 00", pages[0])
        self.assertEqual([p.strip() for p in pages[1:]], [""] * 9)


if __name__ == "__main__":
    unittest.main()
