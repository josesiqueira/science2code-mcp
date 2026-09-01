"""Tests for the MCP layer as a server, not as a library.

Everything in `tests/test_server.py` calls `verify_quote` and `find_passage` as
Python functions. That leaves the whole protocol surface untested, and the
protocol surface is the only part of this project a client ever touches. The
gap was not theoretical. Driving a real server over stdio with hand written
JSON-RPC frames found that `fastmcp>=2.0`, the version floor this project
declared, does not start: every release from 2.0.0 to 2.2.6 raises
`TypeError: FastMCP.tool() got an unexpected keyword argument 'annotations'`,
so the advertised install produced a server that could not be built.

Two tiers of test live here.

  * In process, against the object `build_server()` returns. Cheap, and enough
    to pin the tool list, the descriptions, the annotations and the ordering.
  * Out of process, spawning `python -m science2code.server` and speaking
    JSON-RPC to its standard input. Slower, and the only way to prove that a
    value survives serialisation. `char_interval: null` is the case that
    matters: the design depends on the key being present and explicitly null
    rather than absent, following Google LangExtract's convention, and a
    serialiser that drops null keys would break that silently while every
    in-process test kept passing.

Every test here skips cleanly when fastmcp is absent, because the package's
whole point is that `science2code.normalise` and `science2code.anchor` import
on a bare Python. `pip install -e .` with no extra must still run the suite.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from science2code import __version__, server
from science2code.corpus import MANIFEST_VERSION
from science2code.extract import HEADER_SENTINEL, PAGE_BREAK

try:  # pragma: no cover - depends on the environment
    import fastmcp  # noqa: F401
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_FASTMCP = False
else:
    HAVE_FASTMCP = True

needs_fastmcp = unittest.skipUnless(
    HAVE_FASTMCP, "fastmcp is not installed, so there is no MCP layer to drive"
)

PAPER_ID = "held-paper"
QUOTE = "A perfect search system should be like a human assistant."
BODY = (
    "Introduction to the matter at hand, at some length. "
    + QUOTE
    + " The remainder of the page continues in the same vein."
)
HEADER = (
    "Paper ID: %s\n"
    "title: A title written by this toolchain and not by the paper\n"
    "%s" % (PAPER_ID, HEADER_SENTINEL)
)
DOCUMENT = HEADER + PAGE_BREAK + BODY + PAGE_BREAK

#: Long enough to clear MIN_LOCATABLE_CHARS, and absent from the document.
ABSENT = "This sentence occurs in no document that is held anywhere."


def _write_corpus(root: Path) -> None:
    """A whole corpus on disk, with no PDF extractor and no PDF reader.

    `Corpus.load` reads `manifest.json` and the text files beside it. The PDF
    is written as a few real bytes whose digest is recorded, so `is_stale()`
    finds nothing to complain about and the freshness absence stays out of the
    responses these tests assert on.
    """
    pdf = root / "held-paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 not a real document, only a digest anchor\n")
    text = root / "held-paper.txt"
    text.write_text(DOCUMENT, encoding="utf-8")
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "extractor": "test/0 reading-order",
        "papers": [
            {
                "paper_id": PAPER_ID,
                "title": None,
                "authors": [],
                "year": None,
                "venue": None,
                "doi": None,
                "pdf_path": "held-paper.pdf",
                "text_path": "held-paper.txt",
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "text_sha256": hashlib.sha256(text.read_bytes()).hexdigest(),
                "pages": 1,
                "held": True,
                "note": None,
            }
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


class StdioClient:
    """A hand written MCP client. No client library, so nothing is smoothed over.

    Deliberately raw. A library client parses the frames into objects and would
    hide the two properties these tests exist to check: that a key is present
    and explicitly null on the wire, and that nothing but JSON-RPC is written to
    standard output.
    """

    def __init__(self, corpus_root: str | None, cwd: str) -> None:
        environment = dict(os.environ)
        environment.pop(server.CORPUS_ROOT_ENV, None)
        if corpus_root is not None:
            environment[server.CORPUS_ROOT_ENV] = corpus_root
        self.process = subprocess.Popen(
            [sys.executable, "-m", "science2code.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            cwd=cwd,
            text=True,
            bufsize=1,
        )
        self._next_id = 0

    def _send(self, message: dict) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def request(self, method: str, params: dict | None = None, timeout: float = 60.0) -> dict:
        self._next_id += 1
        wanted = self._next_id
        message = {"jsonrpc": "2.0", "id": wanted, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)
        deadline = time.monotonic() + timeout
        assert self.process.stdout is not None
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                raise AssertionError("the server closed its output stream")
            # Anything on standard output that is not JSON-RPC breaks every
            # client there is, so parsing every line is itself the assertion.
            frame = json.loads(line)
            if frame.get("id") == wanted:
                return frame
        raise AssertionError("no reply to %s within %.0f seconds" % (method, timeout))

    def initialize(self) -> dict:
        reply = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "science2code-tests", "version": "0"},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return reply

    def call(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.wait(timeout=10)
        except Exception:  # pragma: no cover - only on a wedged child
            self.process.kill()


@needs_fastmcp
class TheBuiltServer(unittest.TestCase):
    """What `build_server()` hands to fastmcp, checked in process."""

    def test_it_declares_this_package_version_and_not_fastmcps(self):
        # Before this was passed explicitly, the MCP handshake reported
        # fastmcp's own version as this server's, so a client asking which
        # science2code it was talking to was told "3.4.7".
        built = server.build_server()
        self.assertEqual(getattr(built, "version", None), __version__)
        self.assertIn(__version__, server.env.SERVER_VERSION)

    def test_it_masks_the_details_of_an_unhandled_exception(self):
        # Not style. An unhandled exception has its own message forwarded to
        # the client, and an OSError's message carries an absolute local path.
        built = server.build_server()
        # fastmcp has kept this under both names across its releases, so both
        # are tried before the test concludes anything.
        for attribute in ("mask_error_details", "_mask_error_details"):
            if hasattr(built, attribute):
                self.assertTrue(getattr(built, attribute), attribute)
                return
        self.fail("this fastmcp does not record mask_error_details under either name")

    def test_building_it_twice_registers_the_same_tools_in_the_same_order(self):
        first = [spec.name for spec in server.TOOL_SPECS]
        server.build_server()
        second = [spec.name for spec in server.TOOL_SPECS]
        self.assertEqual(first, second)
        self.assertEqual(first, ["verify_quote", "find_passage"])


@needs_fastmcp
class OverStdio(unittest.TestCase):
    """One server process, driven with raw JSON-RPC frames."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.corpus = cls.root / "papers"
        cls.corpus.mkdir()
        _write_corpus(cls.corpus)
        cls.cwd = cls.root / "cwd"
        cls.cwd.mkdir()
        cls.client = StdioClient(str(cls.corpus), str(cls.cwd))
        cls.handshake = cls.client.initialize()
        cls.listing = cls.client.request("tools/list")

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls._tmp.cleanup()

    def tools(self) -> list:
        return self.listing["result"]["tools"]

    def by_name(self, name: str) -> dict:
        for tool in self.tools():
            if tool["name"] == name:
                return tool
        raise AssertionError("%s is not advertised" % name)

    def envelope(self, reply: dict) -> dict:
        result = reply["result"]
        self.assertFalse(
            result.get("isError"),
            "the call failed: %s" % json.dumps(result)[:400],
        )
        return json.loads(result["content"][0]["text"])

    # -- the handshake

    def test_it_initialises_and_names_itself(self):
        info = self.handshake["result"]["serverInfo"]
        self.assertEqual(info["name"], "science2code")
        self.assertEqual(info["version"], __version__)

    # -- the tool list

    def test_both_tools_are_advertised_in_a_fixed_order(self):
        self.assertEqual([t["name"] for t in self.tools()], ["verify_quote", "find_passage"])

    def test_the_descriptions_arrive_whole(self):
        # The tool descriptions are this project's primary reliability surface,
        # so a client that received a truncated one would be reading a
        # different contract from the one this repository reviews.
        self.assertEqual(
            self.by_name("verify_quote")["description"], server.VERIFY_QUOTE_DESCRIPTION
        )
        self.assertEqual(
            self.by_name("find_passage")["description"], server.FIND_PASSAGE_DESCRIPTION
        )

    def test_the_read_only_annotations_reach_the_client(self):
        for name in ("verify_quote", "find_passage"):
            annotations = self.by_name(name).get("annotations")
            self.assertIsNotNone(annotations, "%s carries no annotations" % name)
            for key, value in server.READ_ONLY_ANNOTATIONS.items():
                self.assertEqual(annotations.get(key), value, "%s.%s" % (name, key))

    def test_the_schemas_name_the_arguments_the_tools_actually_take(self):
        # This is the check that would have caught the README, which documented
        # `your_text` for verify_quote and `paper_id` for find_passage. Both are
        # rejected by the real server: the first as a missing required argument
        # and the second as an unexpected keyword.
        verify = self.by_name("verify_quote")["inputSchema"]
        self.assertEqual(sorted(verify["properties"]), ["paper_id", "t_locate", "text"])
        self.assertEqual(verify["required"], ["text"])
        find = self.by_name("find_passage")["inputSchema"]
        self.assertEqual(sorted(find["properties"]), ["max_hits", "paper_ids", "query"])
        self.assertEqual(find["required"], ["query"])

    # -- the responses

    def test_a_located_quote_carries_its_interval(self):
        payload = self.envelope(self.client.call("verify_quote", {"text": QUOTE}))
        self.assertEqual(payload["outcome"], "VERBATIM_EXACT")
        self.assertEqual(payload["paper_id"], PAPER_ID)
        self.assertEqual(payload["document_text"], QUOTE)
        self.assertEqual(payload["char_interval"]["basis"], "normalised_document")

    def test_a_refusal_carries_char_interval_as_an_explicit_null(self):
        # The property the design rests on, checked where it can actually fail:
        # after serialisation, in the bytes the client reads. `assertIsNone` on
        # a parsed dict would pass just as happily if the key had been dropped,
        # so the raw text is searched for the key as well.
        reply = self.client.call("verify_quote", {"text": ABSENT, "paper_id": PAPER_ID})
        raw = reply["result"]["content"][0]["text"]
        payload = json.loads(raw)
        self.assertEqual(payload["outcome"], "NOT_LOCATABLE")
        self.assertIn("char_interval", payload)
        self.assertIsNone(payload["char_interval"])
        self.assertIn('"char_interval":null', raw.replace(", ", ",").replace(": ", ":"))

    def test_the_same_null_survives_the_structured_content_too(self):
        # A client may read either the text block or structuredContent, and a
        # key that is explicit in one and missing from the other would let the
        # choice of client decide what the refusal means.
        reply = self.client.call("verify_quote", {"text": ABSENT, "paper_id": PAPER_ID})
        structured = reply["result"].get("structuredContent")
        if structured is None:  # pragma: no cover - depends on the fastmcp version
            self.skipTest("this fastmcp does not send structuredContent")
        self.assertIn("char_interval", structured)
        self.assertIsNone(structured["char_interval"])

    def test_a_zero_hit_search_is_a_result_and_not_an_error(self):
        payload = self.envelope(self.client.call("find_passage", {"query": ABSENT}))
        self.assertEqual(payload["outcome"], "OK")
        self.assertEqual(payload["hit_count"], 0)
        self.assertEqual(payload["hits"], [])

    def test_no_envelope_carries_a_boolean(self):
        # The boolean ban is enforced by `envelope.validate` in process. This
        # checks it held all the way through JSON serialisation, where a
        # serialiser that rendered a null or a number as true or false would
        # reintroduce exactly the merge the ban exists to prevent.
        for name, arguments in (
            ("verify_quote", {"text": QUOTE}),
            ("verify_quote", {"text": ABSENT}),
            ("verify_quote", {"text": QUOTE, "paper_id": "no-such-paper"}),
            ("find_passage", {"query": QUOTE}),
        ):
            with self.subTest(tool=name, arguments=arguments):
                raw = self.client.call(name, arguments)["result"]["content"][0]["text"]
                self.assertNotIn("true", raw)
                self.assertNotIn("false", raw)

    # -- misuse

    def test_a_missing_required_argument_is_named(self):
        result = self.client.call("verify_quote", {"paper_id": PAPER_ID})["result"]
        self.assertTrue(result.get("isError"))
        self.assertIn("text", result["content"][0]["text"])

    def test_an_unknown_tool_is_refused_without_stopping_the_server(self):
        result = self.client.call("science2code_delete_everything", {})["result"]
        self.assertTrue(result.get("isError"))
        self.assertEqual(
            [t["name"] for t in self.client.request("tools/list")["result"]["tools"]],
            ["verify_quote", "find_passage"],
        )

    def test_a_huge_query_is_refused_rather_than_hanging_the_server(self):
        payload = self.envelope(
            self.client.call("find_passage", {"query": "z" * 200000})
        )
        # A query longer than any real quote is refused as a typed outcome, not
        # answered after seconds of fuzzy scoring on a single stateless call.
        self.assertEqual(payload["outcome"], "NOT_LOCATABLE")
        fields = [a["field"] for a in payload.get("not_available", [])]
        self.assertIn("hits", fields)

    def test_an_identifier_that_looks_like_a_path_is_just_an_unknown_identifier(self):
        payload = self.envelope(
            self.client.call("verify_quote", {"text": QUOTE, "paper_id": "../../etc/passwd"})
        )
        self.assertEqual(payload["outcome"], "SOURCE_UNKNOWN")

    # -- statelessness

    def test_repeated_calls_in_one_process_give_the_same_bytes(self):
        first = self.client.call("verify_quote", {"text": QUOTE})["result"]
        self.client.call("find_passage", {"query": QUOTE})
        self.client.call("verify_quote", {"text": ABSENT})
        second = self.client.call("verify_quote", {"text": QUOTE})["result"]
        self.assertEqual(first, second)

    def test_the_server_writes_no_file_while_it_serves(self):
        # The one invariant, checked against the running process rather than
        # against the source. The working directory is the server's own, and
        # the corpus directory is the only thing it reads.
        def listing(root: Path) -> dict:
            return {
                str(path.relative_to(root)): path.stat().st_size
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }

        before = (listing(self.cwd), listing(self.corpus))
        for _ in range(3):
            self.client.call("verify_quote", {"text": QUOTE})
            self.client.call("find_passage", {"query": QUOTE})
        self.assertEqual(before, (listing(self.cwd), listing(self.corpus)))


@needs_fastmcp
class AcrossRestarts(unittest.TestCase):
    """Two processes, one question. The statelessness claim, checked."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.corpus = self.root / "papers"
        self.corpus.mkdir()
        _write_corpus(self.corpus)

    def answers(self) -> list:
        client = StdioClient(str(self.corpus), str(self.root))
        try:
            client.initialize()
            listing = client.request("tools/list")["result"]
            replies = [
                client.call("verify_quote", {"text": QUOTE})["result"],
                client.call("verify_quote", {"text": ABSENT})["result"],
                client.call("find_passage", {"query": QUOTE})["result"],
            ]
        finally:
            client.close()
        return [listing] + replies

    def test_a_restarted_server_answers_identically(self):
        self.assertEqual(self.answers(), self.answers())


@needs_fastmcp
class WithoutACorpus(unittest.TestCase):
    """A corpus that is not there is a refusal, not a crash."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def refuse(self, corpus_root: str | None) -> dict:
        client = StdioClient(corpus_root, str(self.root))
        try:
            client.initialize()
            reply = client.call("verify_quote", {"text": QUOTE})
        finally:
            client.close()
        result = reply["result"]
        self.assertFalse(
            result.get("isError"),
            "a missing corpus must be an outcome, not a protocol error: %s"
            % json.dumps(result)[:400],
        )
        return json.loads(result["content"][0]["text"])

    def test_a_missing_directory_is_corpus_unavailable(self):
        payload = self.refuse(str(self.root / "nowhere"))
        self.assertEqual(payload["outcome"], "CORPUS_UNAVAILABLE")
        self.assertIsNone(payload["char_interval"])

    def test_a_directory_with_no_manifest_is_corpus_unavailable(self):
        empty = self.root / "empty"
        empty.mkdir()
        self.assertEqual(self.refuse(str(empty))["outcome"], "CORPUS_UNAVAILABLE")

    def test_a_manifest_that_is_not_json_names_the_file_and_not_a_traceback(self):
        broken = self.root / "broken"
        broken.mkdir()
        (broken / "manifest.json").write_text("{ this is not json", encoding="utf-8")
        payload = self.refuse(str(broken))
        self.assertEqual(payload["outcome"], "CORPUS_UNAVAILABLE")
        rendered = json.dumps(payload)
        self.assertNotIn("Traceback", rendered)
        self.assertNotIn(".py\", line", rendered)

    def test_a_manifest_with_a_bad_field_names_that_field(self):
        bad = self.root / "bad"
        bad.mkdir()
        (bad / "manifest.json").write_text(
            json.dumps({"manifest_version": 7, "papers": []}), encoding="utf-8"
        )
        payload = self.refuse(str(bad))
        self.assertEqual(payload["outcome"], "CORPUS_UNAVAILABLE")
        rendered = json.dumps(payload)
        self.assertIn("manifest_version", rendered)
        self.assertNotIn("Traceback", rendered)

    def test_with_no_corpus_configured_at_all_it_still_answers(self):
        payload = self.refuse(None)
        self.assertEqual(payload["outcome"], "CORPUS_UNAVAILABLE")

    def test_a_root_naming_a_user_who_does_not_exist_is_still_an_outcome(self):
        # `Path("~nobody/papers").expanduser()` raises RuntimeError. Resolving
        # the root used to happen outside the guard in `_open_corpus`, so this
        # plain misconfiguration escaped the tool and reached the client as a
        # protocol-level error carrying a bare string, which is not one of the
        # eight outcomes a caller is promised it can enumerate. `refuse` asserts
        # isError is false, so this fails loudly if that ever returns.
        payload = self.refuse("~no-such-user-93f2a1/papers")
        self.assertEqual(payload["outcome"], "CORPUS_UNAVAILABLE")
        self.assertIsNone(payload["char_interval"])


class WithoutFastmcp(unittest.TestCase):
    """The half of the contract that holds when there is no MCP client at all.

    This one does NOT skip. The package promises that the tools import and run
    on a bare Python, so the promise is checked on every machine, including the
    ones where fastmcp is installed.
    """

    def test_the_tools_are_importable_and_callable_as_functions(self):
        self.assertTrue(callable(server.verify_quote))
        self.assertTrue(callable(server.find_passage))

    def test_build_server_names_the_extra_when_fastmcp_is_missing(self):
        source = Path(server.__file__).read_text(encoding="utf-8")
        # The message a user without the extra actually sees. Pinned as text
        # because it is the only instruction they get.
        self.assertIn("fastmcp is not installed", source)
        self.assertIn("science2code.server", source)

    def test_nothing_outside_build_server_imports_fastmcp(self):
        source = Path(server.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            if "import fastmcp" in line or "from fastmcp" in line:
                self.assertTrue(
                    line.startswith(" "),
                    "fastmcp is imported at module level: %r" % line,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
