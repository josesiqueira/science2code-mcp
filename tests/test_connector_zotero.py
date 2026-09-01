"""Tests for the optional, read-only Zotero corpus connector.

These are hermetic: the Zotero local API is mocked, so the suite never needs
Zotero running. They pin the properties that make the connector safe, that it
is read only, that it resolves stored and linked attachments correctly, that a
re-sync reconciles the folder, and that reconcile never deletes a file the tool
did not create.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from science2code.connectors import zotero

# ---------------------------------------------------------------------------
# a fake local API, so no Zotero is needed
# ---------------------------------------------------------------------------


class FakeZotero:
    """Routes the few GET paths the connector uses to canned data."""

    def __init__(self, items=None, children=None, annotations=None, version="42"):
        self.items = items or []
        self.children = children or {}      # item_key -> [child items]
        self.annotations = annotations or []  # annotation data dicts
        self.version = version

    def __call__(self, base, path, params=None):
        params = params or {}
        if path == "/items" and params.get("itemType") == "annotation":
            return [{"data": a} for a in self.annotations], self.version
        if path == "/items":
            return list(self.items), self.version
        if path.endswith("/children"):
            key = path.split("/items/")[1].split("/children")[0]
            return list(self.children.get(key, [])), self.version
        return [], self.version


def _item(key, itemType="conferencePaper", title="T"):
    return {"key": key, "data": {"key": key, "itemType": itemType, "title": title}}


def _pdf_attachment(key, link_mode="imported_file", filename="paper.pdf", path=None):
    d = {"key": key, "itemType": "attachment", "contentType": "application/pdf",
         "linkMode": link_mode}
    if filename:
        d["filename"] = filename
    if path:
        d["path"] = path
    return {"key": key, "data": d}


def _annotation(key, parent, text, atype="highlight", page="1"):
    return {"key": key, "itemType": "annotation", "annotationType": atype,
            "annotationText": text, "annotationPageLabel": page,
            "parentItem": parent, "annotationComment": ""}


# ---------------------------------------------------------------------------
# read-only, structurally
# ---------------------------------------------------------------------------


class ReadOnlyByConstruction(unittest.TestCase):
    def test_the_only_http_method_named_in_the_module_is_get(self):
        src = Path(zotero.__file__).read_text(encoding="utf-8")
        for verb in ('method="POST"', 'method="PUT"', 'method="PATCH"', 'method="DELETE"',
                     "method='POST'", "method='DELETE'"):
            self.assertNotIn(verb, src, "a non-GET method appears in the connector")
        self.assertIn('method="GET"', src)

    def test_the_module_requests_no_write_key(self):
        src = Path(zotero.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("api-key", src)
        self.assertNotIn("apikey", src)


# ---------------------------------------------------------------------------
# attachment resolution
# ---------------------------------------------------------------------------


class ResolveAttachment(unittest.TestCase):
    def test_a_stored_file_resolves_under_storage(self):
        with TemporaryDirectory() as tmp:
            dd = Path(tmp)
            f = dd / "storage" / "ATT1" / "paper.pdf"
            f.parent.mkdir(parents=True)
            f.write_bytes(b"%PDF")
            kind, path = zotero.resolve_attachment(
                {"key": "ATT1", "linkMode": "imported_file", "filename": "paper.pdf"}, dd)
            self.assertEqual(kind, "stored")
            self.assertEqual(path, f)

    def test_a_missing_stored_file_resolves_to_none(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(zotero.resolve_attachment(
                {"key": "ATT1", "linkMode": "imported_file", "filename": "gone.pdf"}, Path(tmp)))

    def test_a_linked_file_resolves_to_its_own_path(self):
        with TemporaryDirectory() as tmp:
            ext = Path(tmp) / "elsewhere.pdf"
            ext.write_bytes(b"%PDF")
            kind, path = zotero.resolve_attachment(
                {"key": "ATT2", "linkMode": "linked_file", "path": str(ext)}, Path(tmp) / "zdata")
            self.assertEqual(kind, "linked")
            self.assertEqual(path, ext)

    def test_a_linked_url_is_not_a_local_file(self):
        self.assertIsNone(zotero.resolve_attachment(
            {"key": "A", "linkMode": "linked_url"}, Path("/nope")))


# ---------------------------------------------------------------------------
# the corpus may never be built inside the Zotero data directory
# ---------------------------------------------------------------------------


class NeverInsideDataDir(unittest.TestCase):
    def test_an_out_inside_the_data_dir_is_refused(self):
        with TemporaryDirectory() as tmp:
            dd = Path(tmp) / "Zotero"
            dd.mkdir()
            with self.assertRaises(ValueError):
                zotero._assert_outside_data_dir(dd / "corpus", dd)

    def test_an_out_beside_the_data_dir_is_allowed(self):
        with TemporaryDirectory() as tmp:
            dd = Path(tmp) / "Zotero"
            dd.mkdir()
            zotero._assert_outside_data_dir(Path(tmp) / "corpus", dd)  # no raise


# ---------------------------------------------------------------------------
# sync and reconcile
# ---------------------------------------------------------------------------


class SyncAndReconcile(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.dd = self.tmp / "Zotero"
        # a real stored PDF on disk for the item PAPER1 / attachment ATT1
        self.pdf = self.dd / "storage" / "ATT1" / "paper.pdf"
        self.pdf.parent.mkdir(parents=True)
        self.pdf.write_bytes(b"%PDF stored paper")
        self._saved = zotero._api_get

    def tearDown(self):
        zotero._api_get = self._saved

    def _install(self, fake):
        zotero._api_get = fake

    def test_a_stored_pdf_is_symlinked_under_the_item_key(self):
        fake = FakeZotero(
            items=[_item("PAPER1")],
            children={"PAPER1": [_pdf_attachment("ATT1")]},
        )
        self._install(fake)
        out = self.tmp / "corpus"
        summary = zotero.sync("mytag", out, data_dir=self.dd)
        link = out / "PAPER1.pdf"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), self.pdf.resolve())
        self.assertEqual(summary["symlinked"], 1)
        self.assertEqual(summary["items"], 1)
        state = json.loads((out / zotero.STATE_FILE).read_text())
        self.assertIn("PAPER1", state["entries"])

    def test_a_dropped_tag_removes_our_link_but_not_a_foreign_file(self):
        # first sync creates PAPER1.pdf
        self._install(FakeZotero(items=[_item("PAPER1")],
                                 children={"PAPER1": [_pdf_attachment("ATT1")]}))
        out = self.tmp / "corpus"
        zotero.sync("mytag", out, data_dir=self.dd)
        foreign = out / "USER_FILE.pdf"
        foreign.write_bytes(b"not ours")
        # second sync: the tag now matches nothing
        self._install(FakeZotero(items=[]))
        summary = zotero.sync("mytag", out, data_dir=self.dd)
        self.assertEqual(summary["removed"], 1)
        self.assertFalse((out / "PAPER1.pdf").exists())
        self.assertTrue(foreign.exists(), "a file the tool did not create must survive")

    def test_a_dry_run_writes_nothing(self):
        self._install(FakeZotero(items=[_item("PAPER1")],
                                 children={"PAPER1": [_pdf_attachment("ATT1")]}))
        out = self.tmp / "corpus"
        summary = zotero.sync("mytag", out, data_dir=self.dd, dry_run=True)
        self.assertTrue(summary["dry_run"])
        self.assertFalse((out / "PAPER1.pdf").exists())
        self.assertFalse((out / zotero.STATE_FILE).exists())

    def test_two_pdfs_on_one_item_get_distinct_stems(self):
        (self.dd / "storage" / "ATT2").mkdir(parents=True)
        (self.dd / "storage" / "ATT2" / "paper.pdf").write_bytes(b"%PDF two")
        self._install(FakeZotero(
            items=[_item("PAPER1")],
            children={"PAPER1": [_pdf_attachment("ATT1"), _pdf_attachment("ATT2")]},
        ))
        out = self.tmp / "corpus"
        zotero.sync("mytag", out, data_dir=self.dd)
        self.assertTrue((out / "PAPER1-ATT1.pdf").is_symlink())
        self.assertTrue((out / "PAPER1-ATT2.pdf").is_symlink())


# ---------------------------------------------------------------------------
# annotations as candidate quotes
# ---------------------------------------------------------------------------


class Annotations(unittest.TestCase):
    def setUp(self):
        self._saved = zotero._api_get
        self.addCleanup(lambda: setattr(zotero, "_api_get", self._saved))

    def test_highlights_are_collected_and_grouped_by_paper(self):
        zotero._api_get = FakeZotero(
            items=[_item("PAPER1")],
            children={"PAPER1": [_pdf_attachment("ATT1")]},
            annotations=[
                _annotation("AN1", "ATT1", "a real highlighted sentence"),
                _annotation("AN2", "ATT1", None, atype="image"),  # no text, must be dropped
                _annotation("AN3", "OTHER", "belongs to another paper"),
            ],
        )
        rows = zotero.collect_annotations("mytag")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["item_key"], "PAPER1")
        self.assertEqual(rows[0]["text"], "a real highlighted sentence")


if __name__ == "__main__":
    unittest.main()
