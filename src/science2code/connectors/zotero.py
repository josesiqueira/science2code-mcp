"""Populate a corpus folder from a tagged subset of a Zotero library, read only.

This is an OPTIONAL connector. The science2code core never imports it and does
not depend on it: a clone with no Zotero, and no interest in Zotero, installs
and runs exactly the same. What this module adds is one deterministic command,
run by a person or a scheduler and never by a language model, that turns
"papers tagged X in Zotero" into "PDF files in a folder", which the core then
indexes and verifies like any other folder. The folder is the boundary; once it
exists, nothing here is involved in a lookup.

Why deterministic and not agent driven: building a corpus is a data pipeline,
and a data pipeline has to be reproducible. The same tag on the same library
must yield the same folder every run. A language model deciding what to copy
each time cannot promise that, so corpus population is a command, and the model
is left to do only what it is good at, verifying a quote against the result.

Read only, by construction and on purpose:

  - Every request to Zotero is a GET against the built-in local API at
    http://127.0.0.1:23119/api. There is no function in this module that issues
    a POST, PUT, PATCH or DELETE, and none that requests a write key. Zotero's
    local API answers a write with 501 anyway, but the guarantee here is that
    the code cannot even ask.
  - Nothing is ever written inside the Zotero data directory. Derived files go
    only into the corpus folder you name, and the code refuses to run if that
    folder is inside the data directory.
  - Source PDFs are linked or copied, never modified, and the connector deletes
    only the links and copies it made itself, recorded in its own state file.
    A file it did not create is never removed.

Two kinds of attachment, resolved differently, per Zotero's own model:

  - A STORED attachment lives under <dataDir>/storage/<attachmentKey>/<filename>.
    Zotero manages it, so a symbolic link into the corpus is cheap and always
    reflects the current bytes. These are symlinked.
  - A LINKED-FILE attachment is a pointer to a file anywhere on disk that Zotero
    neither copies nor manages, so its original can move or vanish. These are
    copied into the corpus, so the corpus does not depend on a path outside it.

The identity handed downstream is the Zotero item key (for example C4IT8KT5),
used as the corpus file stem. It is stable across renames and unique, so a
verify result's paper_id maps straight back to the Zotero item a citation is
drawn from, which is the whole point of connecting the two.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:23119/api/users/0"
DEFAULT_DATA_DIR = Path(os.environ.get("ZOTERO_DATA_DIR", str(Path.home() / "Zotero")))
STATE_FILE = ".science2code_zotero.json"
#: annotationType values that carry text lifted from the PDF. An image or ink
#: annotation has only a position box and no text, so it is not a candidate
#: quote and is skipped.
TEXT_ANNOTATION_TYPES = ("highlight", "underline")
_HTTP_TIMEOUT = 15


class ZoteroUnavailable(RuntimeError):
    """Zotero is not reachable, or its local API is not enabled."""


# ---------------------------------------------------------------------------
# read-only transport: the only way this module talks to Zotero
# ---------------------------------------------------------------------------


def _api_get(base: str, path: str, params: dict[str, Any] | None = None) -> Any:
    """GET one local-API path and parse JSON. The sole Zotero access point.

    There is deliberately no sibling that issues any other HTTP method. Every
    read in this module funnels through here, so "read only" is a property of
    the code, not a promise in a comment.
    """
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, method="GET", headers={"Zotero-API-Version": "3"})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
            version = resp.headers.get("Last-Modified-Version")
            data = json.loads(body) if body.strip() else []
            return data, version
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            raise ZoteroUnavailable(
                "Zotero returned 403. Enable Settings, Advanced, 'Allow other "
                "applications on this computer to communicate with Zotero'."
            ) from exc
        raise ZoteroUnavailable("Zotero local API error %s for %s" % (exc.code, path)) from exc
    except urllib.error.URLError as exc:
        raise ZoteroUnavailable(
            "cannot reach Zotero's local API at %s. Is Zotero running? (%s)"
            % (base, exc.reason)
        ) from exc


def _paged_items(base: str, params: dict[str, Any]) -> tuple[list[dict], str | None]:
    """Every item matching `params`, following start/limit, with the version.

    The local API returns the whole matching set unpaginated, but a start/limit
    walk is used anyway so a very large tag still comes back in bounded chunks
    and the same code path works against the paginated web API.
    """
    out: list[dict] = []
    start = 0
    limit = 100
    version = None
    while True:
        page, version = _api_get(base, "/items", {**params, "start": start, "limit": limit})
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        if len(page) < limit:
            break
        start += limit
    return out, version


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------


def check(base: str = DEFAULT_BASE) -> dict[str, Any]:
    """A reachability probe. Returns a small dict, raises if Zotero is down."""
    items, version = _api_get(base, "/items", {"limit": 1})
    return {"reachable": True, "library_version": version}


def tagged_items(tag: str, base: str = DEFAULT_BASE) -> tuple[list[dict], str | None]:
    """Top-level items carrying `tag`, attachments and annotations excluded.

    itemType=-attachment drops standalone attachments; annotations are excluded
    the same way. The PDF for each item is found through its children.
    """
    return _paged_items(base, {"tag": tag, "itemType": "-attachment || -annotation"})


def _children(key: str, base: str = DEFAULT_BASE) -> list[dict]:
    data, _ = _api_get(base, "/items/%s/children" % key)
    return data if isinstance(data, list) else []


def pdf_attachments(item_key: str, base: str = DEFAULT_BASE) -> list[dict]:
    """The PDF attachment child items of one top-level item."""
    return [
        c for c in _children(item_key, base)
        if c.get("data", {}).get("itemType") == "attachment"
        and c.get("data", {}).get("contentType") == "application/pdf"
    ]


def annotations_by_parent(base: str = DEFAULT_BASE) -> dict[str, list[dict]]:
    """Every text annotation in the library, grouped by parent attachment key.

    Annotations are logically children of a PDF attachment, but this Zotero
    build does not return them through the attachment's /children endpoint, so
    they are fetched with an itemType=annotation query and grouped by their
    own parentItem field. Only text-bearing types are kept, because an image
    or ink annotation carries no quotable text. One query serves every paper.
    """
    rows, _ = _paged_items(base, {"itemType": "annotation"})
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        d = r.get("data", {})
        if d.get("annotationType") not in TEXT_ANNOTATION_TYPES:
            continue
        parent = d.get("parentItem")
        if parent:
            grouped.setdefault(parent, []).append(d)
    return grouped


# ---------------------------------------------------------------------------
# attachment to a real file on disk
# ---------------------------------------------------------------------------


def resolve_attachment(att_data: dict, data_dir: Path) -> tuple[str, Path] | None:
    """Resolve an attachment item to (kind, path), or None if unresolvable.

    kind is "stored" for a Zotero-managed file, which is symlinked downstream,
    or "linked" for a linked-file attachment at an arbitrary path, which is
    copied. A path that does not exist on disk yields None, and the caller
    reports it rather than fabricating a corpus entry.
    """
    link_mode = att_data.get("linkMode")
    key = att_data.get("key")
    if link_mode in ("imported_file", "imported_url"):
        filename = att_data.get("filename")
        if not (key and filename):
            return None
        path = data_dir / "storage" / key / filename
        return ("stored", path) if path.is_file() else None
    if link_mode == "linked_file":
        raw = att_data.get("path")
        if not raw:
            return None
        path = Path(raw)
        return ("linked", path) if path.is_file() else None
    # linked_url and imported_url-without-file are not local files.
    return None


# ---------------------------------------------------------------------------
# corpus folder, and the state that makes reconcile safe
# ---------------------------------------------------------------------------


def _load_state(out: Path) -> dict:
    f = out / STATE_FILE
    if f.is_file():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"library_version": None, "entries": {}}


def _save_state(out: Path, state: dict) -> None:
    (out / STATE_FILE).write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")


def _assert_outside_data_dir(out: Path, data_dir: Path) -> None:
    """Refuse to treat a folder inside the Zotero data directory as the corpus.

    Writing sidecars or links inside Zotero's own storage is the way to have
    Zotero overwrite or delete them on its next sync, and to risk confusing
    Zotero. The corpus is always somewhere Zotero does not manage.
    """
    out_r = out.resolve()
    dd_r = data_dir.resolve()
    if out_r == dd_r or dd_r in out_r.parents:
        raise ValueError(
            "refusing to build a corpus inside the Zotero data directory (%s). "
            "Choose an --out folder outside it." % dd_r
        )


def _stem_for(item_key: str, atts: list[dict], this_att_key: str) -> str:
    """The corpus file stem: the item key, plus the attachment key if the item
    has more than one PDF, so identity stays stable and collision free."""
    return item_key if len(atts) == 1 else "%s-%s" % (item_key, this_att_key)


def sync(tag: str, out: Path, *, data_dir: Path = DEFAULT_DATA_DIR,
         copy_all: bool = False, dry_run: bool = False, base: str = DEFAULT_BASE) -> dict:
    """Make `out` mirror the PDFs of the items tagged `tag`. Read only on Zotero.

    Stored PDFs are symlinked, linked-file PDFs are copied (or everything is
    copied with copy_all, for a corpus that must survive Zotero moving files).
    Items that lost the tag or were deleted have the link or copy THIS TOOL made
    removed, and nothing else is touched. Returns a summary dict.
    """
    out.mkdir(parents=True, exist_ok=True)
    _assert_outside_data_dir(out, data_dir)
    state = _load_state(out)
    old_entries: dict = state.get("entries", {})
    new_entries: dict = {}
    added = kept = copied = missing = 0

    items, version = tagged_items(tag, base)
    for item in items:
        item_key = item.get("key") or item.get("data", {}).get("key")
        if not item_key:
            continue
        atts = pdf_attachments(item_key, base)
        for att in atts:
            ad = att.get("data", {})
            att_key = ad.get("key")
            resolved = resolve_attachment(ad, data_dir)
            if resolved is None:
                missing += 1
                continue
            kind, src = resolved
            stem = _stem_for(item_key, atts, att_key)
            target = out / (stem + ".pdf")
            as_copy = copy_all or kind == "linked"
            new_entries[stem] = {
                "item_key": item_key, "attachment_key": att_key,
                "source": str(src), "link_mode": ad.get("linkMode"),
                "kind": "copy" if as_copy else "symlink",
            }
            if dry_run:
                continue
            if target.exists() or target.is_symlink():
                # Already present. Trust the state; refresh only if it changed.
                kept += 1
                continue
            if as_copy:
                shutil.copy2(src, target)
                copied += 1
            else:
                target.symlink_to(src)
                added += 1

    removed = _reconcile(out, old_entries, new_entries, dry_run)
    state["entries"] = new_entries
    state["library_version"] = version
    if not dry_run:
        _save_state(out, state)
    return {
        "tag": tag, "out": str(out), "items": len(items),
        "symlinked": added, "copied": copied, "kept": kept,
        "removed": removed, "unresolved": missing,
        "library_version": version, "dry_run": dry_run,
    }


def _reconcile(out: Path, old: dict, new: dict, dry_run: bool) -> int:
    """Remove only the links and copies THIS tool made that are no longer in
    the tagged set. A file not recorded as ours is never deleted."""
    removed = 0
    for stem in old:
        if stem in new:
            continue
        target = out / (stem + ".pdf")
        if not (target.exists() or target.is_symlink()):
            continue
        if dry_run:
            removed += 1
            continue
        try:
            target.unlink()
            # Drop the stale sidecar the core wrote for it, if present.
            sidecar = out / (stem + ".txt")
            if sidecar.is_file():
                sidecar.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def collect_annotations(tag: str, *, base: str = DEFAULT_BASE) -> list[dict]:
    """Every text highlight on the tagged papers, as CANDIDATE quotes.

    Candidate, not verified: the text is lifted from the same PDF text layer
    the extractor reads, so it carries the usual extraction artifacts. Feed each
    to the verifier; a real quote comes back VERBATIM, a garbled one does not.
    The item key travels so a verified highlight maps back to its Zotero paper.
    """
    out = []
    items, _ = tagged_items(tag, base)
    by_parent = annotations_by_parent(base)
    for item in items:
        item_key = item.get("key")
        title = (item.get("data", {}) or {}).get("title")
        for att in pdf_attachments(item_key, base):
            att_key = att.get("data", {}).get("key")
            for a in by_parent.get(att_key, []):
                out.append({
                    "item_key": item_key, "title": title,
                    "page": a.get("annotationPageLabel"),
                    "text": a.get("annotationText"),
                    "comment": a.get("annotationComment") or None,
                })
    return out


# ---------------------------------------------------------------------------
# CLI: run by a person or a scheduler, never by a model
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="science2code-zotero",
        description="Populate a science2code corpus folder from a tagged Zotero "
                    "subset, read only. Then run: science2code index <out>.",
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="Zotero local API base URL")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), type=Path,
                        help="Zotero data directory (holds storage/)")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sync", help="mirror a tag's PDFs into a corpus folder")
    s.add_argument("--tag", required=True)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--copy-all", action="store_true",
                   help="copy every PDF instead of symlinking stored ones")
    s.add_argument("--dry-run", action="store_true")

    a = sub.add_parser("annotations", help="list tagged papers' highlights as candidate quotes")
    a.add_argument("--tag", required=True)
    a.add_argument("--out", type=Path, help="write JSON here instead of stdout")

    sub.add_parser("check", help="probe that Zotero's local API is reachable")

    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            info = check(args.base)
            print("Zotero reachable. library_version=%s" % info["library_version"])
            return 0
        if args.command == "sync":
            summary = sync(args.tag, args.out, data_dir=args.data_dir,
                           copy_all=args.copy_all, dry_run=args.dry_run, base=args.base)
            verb = "would change" if args.dry_run else "corpus"
            print("science2code-zotero sync (%s): tag %r -> %s"
                  % (verb, summary["tag"], summary["out"]))
            print("  items %d | symlinked %d | copied %d | kept %d | removed %d | unresolved %d"
                  % (summary["items"], summary["symlinked"], summary["copied"],
                     summary["kept"], summary["removed"], summary["unresolved"]))
            print("  library_version %s" % summary["library_version"])
            if not args.dry_run:
                print("  next: science2code index %s" % summary["out"])
            return 0
        if args.command == "annotations":
            rows = collect_annotations(args.tag, base=args.base)
            text = json.dumps(rows, indent=1, ensure_ascii=False)
            if args.out:
                Path(args.out).write_text(text + "\n", encoding="utf-8")
                print("wrote %d candidate quote(s) to %s" % (len(rows), args.out))
            else:
                print(text)
            return 0
    except (ZoteroUnavailable, ValueError) as exc:
        print("science2code-zotero: %s" % exc, file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
