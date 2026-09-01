"""Tests for the two tools and their descriptions.

Inline fixtures only. No test reads a real corpus, and no test needs an MCP
client: `build_server()` is the only thing in the package that imports fastmcp,
and nothing here calls it.

Two kinds of test live side by side on purpose. Tier mapping is pinned by
substituting `server._locate`, so the mapping from a rung of the ladder to an
outcome is tested independently of the anchoring algorithm. The end-to-end
tests then run the real normaliser and the real ladder over an inline document,
so the two are known to fit together.
"""

from __future__ import annotations

import ast
import inspect
import types
import unittest

from science2code import corpus as real_corpus
from science2code import envelope as env
from science2code import markers as markers_module
from science2code import server
from science2code.anchor import Anchor, Tier
from science2code.envelope import EnvelopeViolation
from science2code.extract import HEADER_SENTINEL, PAGE_BREAK, split_document

DOC_A = (
    "Regulatory sandboxes are established under Article 57 of the Regulation. "
    "They provide a controlled environment for development and testing. "
    "The provider retains responsibility throughout."
)
DOC_B = (
    "A riskbased approach is applied across the Regulation. "
    "Regulatory sandboxes are established under Article 57 of the Regulation."
)
DOC_C = "Human Oversight is required for every high risk system placed on the market."

# The corpus writes a metadata header ahead of the body and separates pages with
# a form feed. The header holds a title and an author list written by the
# toolchain, so a string found there is not something the paper said.
HEADER_H = (
    "title: Regulatory sandboxes and the question of the wrong author\n"
    "authors: A Person\n"
    "Extracted text follows. Pages are separated by a form feed."
)
BODY_H = (
    "This body never contains that title phrase. It discusses an entirely "
    "different matter at some length, page after page."
)
DOC_H = HEADER_H + "\f" + BODY_H


class FakePaper:
    """Mirrors the real Paper in the one respect the server relies on."""

    def __init__(self, paper_id: str, held: bool = True) -> None:
        self.paper_id = paper_id
        self.held = held


class FakeCorpus:
    """An inline corpus. Never touches the filesystem.

    Shapes follow the real `science2code.corpus.Corpus`: `text()` and `pages()`
    return None for a paper that is known but not held, and `is_stale()`
    returns a list of problem messages rather than a flag.
    """

    def __init__(self, docs, pages=None, stale=(), unheld=(), unreadable=(), no_pages=()):
        self._docs = dict(docs)
        self._pages = dict(pages or {})
        self._stale = list(stale)
        self._unheld = set(unheld)
        self._unreadable = set(unreadable)
        self._no_pages = set(no_pages)

    def papers(self):
        return [FakePaper(pid, pid not in self._unheld) for pid in sorted(self._docs)]

    def get(self, paper_id):
        if paper_id in self._docs:
            return FakePaper(paper_id, paper_id not in self._unheld)
        return None

    def text(self, paper_id):
        if paper_id in self._unreadable:
            raise OSError("the text file could not be read")
        if paper_id in self._unheld:
            return None
        return self._docs.get(paper_id)

    def pages(self, paper_id):
        if paper_id in self._no_pages:
            raise OSError("no page index")
        if paper_id in self._unheld:
            return None
        if paper_id in self._pages:
            return self._pages[paper_id]
        # split_document, not a bare split, because that is what the real
        # Corpus.pages does. A fake that splits more naively than the real
        # thing hides exactly the bugs it was built to catch: it did, once.
        return split_document(self._docs.get(paper_id, ""))[1]

    def is_stale(self):
        return list(self._stale)

    def stale_paper_ids(self):
        # The real corpus begins every is_stale message with the paper id, so
        # the fixture derives the id set the same way rather than tracking it
        # twice and letting the two drift.
        ids = set()
        for message in self._stale:
            head = str(message).split(":", 1)[0].strip()
            if head:
                ids.add(head)
        return sorted(ids)


class TestTheFixtureMatchesTheRealCorpus(unittest.TestCase):
    """The fake is only useful while it has the same shape as the real thing."""

    def test_the_server_only_calls_documented_corpus_methods(self):
        for name in ("papers", "get", "text", "pages", "is_stale",
                     "stale_paper_ids"):
            self.assertTrue(hasattr(real_corpus.Corpus, name), name)
            self.assertTrue(hasattr(FakeCorpus, name), name)

    def test_the_identifier_attribute_is_paper_id(self):
        self.assertIn("paper_id", real_corpus.Paper.__annotations__)
        self.assertTrue(hasattr(FakePaper("x"), "paper_id"))

    def test_is_stale_returns_a_list_in_both(self):
        self.assertIsInstance(FakeCorpus({}).is_stale(), list)
        self.assertIn(
            "list",
            str(inspect.signature(real_corpus.Corpus.is_stale).return_annotation),
        )


def anchor_at(tier: Tier, offset=0, length=10, score=1.0, diff=None) -> Anchor:
    return Anchor(tier, offset, length, score, diff, "difflib")


class ServerTestCase(unittest.TestCase):
    """Substitutes the corpus loader, and restores everything afterwards."""

    def setUp(self):
        self._saved = (server._load_corpus, server._locate)
        server._PREPARED_CACHE.clear()
        self.use_corpus(FakeCorpus({"paper-a": DOC_A, "paper-b": DOC_B, "paper-c": DOC_C}))

    def tearDown(self):
        server._load_corpus, server._locate = self._saved
        server._PREPARED_CACHE.clear()

    def use_corpus(self, corpus):
        self.corpus = corpus
        server._load_corpus = lambda root, _c=corpus: _c

    def broken_corpus(self, exc=None):
        def _raise(root):
            raise exc or FileNotFoundError("no corpus at %s" % root)

        server._load_corpus = _raise

    def pin_tier(self, tier, **kwargs):
        anchor = anchor_at(tier, **kwargs)
        server._locate = lambda haystack, quote, t_locate, _a=anchor: _a


# ---------------------------------------------------------------------------
# the descriptions, which are the reliability surface
# ---------------------------------------------------------------------------


class TestDescriptions(unittest.TestCase):
    MIN_LENGTH = 900

    def test_descriptions(self):
        """Every description is substantial, names its refusals, states the ceiling."""
        required = {
            "verify_quote": (
                "VERBATIM_EXACT",
                "VERBATIM_RELAXED_EXTRACTOR_DAMAGE",
                "PASSAGE_RELOCATED_QUOTE_DIFFERS",
                "NOT_LOCATABLE",
                "SOURCE_NOT_HELD",
                "SOURCE_UNKNOWN",
                "CORPUS_UNAVAILABLE",
            ),
            "find_passage": (
                "VERBATIM_EXACT",
                "VERBATIM_RELAXED_EXTRACTOR_DAMAGE",
                "NOT_LOCATABLE",
                "SOURCE_UNKNOWN",
                "CORPUS_UNAVAILABLE",
                "OK",
            ),
        }
        self.assertEqual({"verify_quote", "find_passage"}, {s.name for s in server.TOOL_SPECS})
        for spec in server.TOOL_SPECS:
            text = spec.description
            self.assertTrue(text and text.strip(), "%s has an empty description" % spec.name)
            self.assertGreater(
                len(text),
                self.MIN_LENGTH,
                "%s description is %d characters, under the %d minimum. It is the "
                "primary reliability surface of this server."
                % (spec.name, len(text), self.MIN_LENGTH),
            )
            for outcome in required[spec.name]:
                self.assertIn(outcome, text, "%s never names %s" % (spec.name, outcome))
                self.assertIn(outcome, env.OUTCOME_NAMES)
            # Compared with whitespace collapsed: the sentence is wrapped for
            # display, and wrapping is presentation, not content.
            self.assertIn(
                " ".join(env.CEILING_SENTENCE.split()),
                " ".join(text.split()),
                "%s does not state the ceiling sentence" % spec.name,
            )

    def test_descriptions_say_when_to_call_the_tool(self):
        # The recorded failure was reasoning over summaries and snippets. The
        # description has to name that, or it is documentation and not a
        # mechanism.
        for spec in server.TOOL_SPECS:
            lowered = spec.description.lower()
            self.assertIn("instead of", lowered, spec.name)
            self.assertIn("snippet", lowered, spec.name)
            self.assertIn("when to call it", lowered, spec.name)

    def test_descriptions_obey_the_same_language_rules_as_a_response(self):
        for spec in server.TOOL_SPECS:
            self.assertTrue(
                env._verbatim_uses_are_outcome_names(spec.description),
                "%s uses the word outside the two outcome names" % spec.name,
            )
            lowered = spec.description.lower()
            for term in env.BANNED_TERMS:
                self.assertNotIn(term, lowered, "%s used %r" % (spec.name, term))

    def test_no_dashes_in_any_shipped_string(self):
        for text in (
            server.VERIFY_QUOTE_DESCRIPTION,
            server.FIND_PASSAGE_DESCRIPTION,
            env.INTERPRETATION_NOTICE,
            env.CEILING_SENTENCE,
        ):
            # Written as escapes so this file does not itself violate the
            # rule it enforces. A literal here would make the repo-wide
            # dash check fail on the test that exists to prevent dashes.
            self.assertNotIn("\u2014", text)  # em dash
            self.assertNotIn("\u2013", text)  # en dash


class TestRegistration(unittest.TestCase):
    def test_exactly_two_tools_in_a_fixed_order(self):
        self.assertEqual(2, len(server.TOOL_SPECS))
        self.assertEqual(["verify_quote", "find_passage"], [s.name for s in server.TOOL_SPECS])

    def test_both_tools_are_annotated_read_only(self):
        for spec in server.TOOL_SPECS:
            self.assertIs(True, spec.annotations["readOnlyHint"], spec.name)
            self.assertIs(False, spec.annotations["destructiveHint"], spec.name)
            self.assertIs(False, spec.annotations["openWorldHint"], spec.name)

    def test_the_mcp_import_is_confined_to_build_server(self):
        source = inspect.getsource(server)
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                module = getattr(node, "module", "") or ""
                if "fastmcp" in module or any("fastmcp" in n for n in names):
                    offenders.append(node.col_offset)
        self.assertTrue(offenders, "fastmcp is never imported")
        for col in offenders:
            self.assertGreater(col, 0, "fastmcp is imported at module level")


class TestThisServerNeverWritesAFile(unittest.TestCase):
    FORBIDDEN = {
        "open", "write", "writelines", "write_text", "write_bytes", "mkdir",
        "makedirs", "remove", "unlink", "rmtree", "rename", "replace", "touch",
        "chmod", "copyfile", "copytree", "move", "dump",
    }

    def test_no_write_call_in_the_runtime_modules(self):
        for module in (server, env):
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                self.assertNotIn(
                    name,
                    self.FORBIDDEN,
                    "%s calls %s() at line %d" % (module.__name__, name, node.lineno),
                )


# ---------------------------------------------------------------------------
# verify_quote: the ladder mapped onto the vocabulary
# ---------------------------------------------------------------------------


class TestTierMapping(ServerTestCase):
    QUOTE = "Regulatory sandboxes are established under Article 57"

    def test_t1_becomes_verbatim_exact(self):
        self.pin_tier(Tier.T1_EXACT, offset=0, length=52, score=1.0)
        result = server.verify_quote(self.QUOTE, paper_id="paper-a")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])

    def test_t2_becomes_relaxed_extractor_damage(self):
        self.pin_tier(Tier.T2_RELAXED, offset=0, length=52, score=1.0)
        result = server.verify_quote(self.QUOTE, paper_id="paper-a")
        self.assertEqual("VERBATIM_RELAXED_EXTRACTOR_DAMAGE", result["outcome"])

    def test_t3_becomes_relocated_and_never_verbatim(self):
        self.pin_tier(Tier.T3_LOCATED, offset=0, length=52, score=0.83, diff="a diff")
        result = server.verify_quote(self.QUOTE, paper_id="paper-a")
        self.assertEqual("PASSAGE_RELOCATED_QUOTE_DIFFERS", result["outcome"])

    def test_t4_becomes_not_locatable(self):
        self.pin_tier(Tier.T4_NOT_LOCATABLE, offset=None, length=None, score=0.21)
        result = server.verify_quote(self.QUOTE, paper_id="paper-a")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        self.assertIsNone(result["char_interval"])
        self.assertEqual(0.21, result["best_score"])

    def test_every_tier_in_the_ladder_is_mapped(self):
        for tier in Tier:
            self.assertIn(tier.name, server._TIER_TO_OUTCOME, "%s has no outcome" % tier.name)

    def test_an_unmapped_tier_raises_rather_than_defaulting(self):
        fake = types.SimpleNamespace(tier=types.SimpleNamespace(name="T5_PROBABLY_FINE"))
        with self.assertRaises(EnvelopeViolation):
            server._outcome_for(fake)

    def test_an_unresolvable_interval_refuses_rather_than_reporting_without_words(self):
        # An outcome that asserts character identity but cannot show the
        # characters would be the exact blur this server exists to prevent.
        self.pin_tier(Tier.T1_EXACT, offset=10_000_000, length=10, score=1.0)
        result = server.verify_quote(self.QUOTE, paper_id="paper-a")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        self.assertNotIn("document_text", result)


class TestVerifyQuoteEndToEnd(ServerTestCase):
    """The real normaliser and the real ladder over an inline document."""

    def test_an_exact_string_is_located_with_the_document_own_characters(self):
        quote = "Regulatory sandboxes are established under Article 57"
        result = server.verify_quote(quote, paper_id="paper-a")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        self.assertIn("Regulatory sandboxes", result["document_text"])
        self.assertEqual("normalised_document", result["char_interval"]["basis"])
        self.assertEqual(1, result["page"])

    def test_a_case_difference_is_extractor_damage_not_a_wrong_quote(self):
        result = server.verify_quote(
            "human oversight is required for every high risk system", paper_id="paper-c")
        self.assertEqual("VERBATIM_RELAXED_EXTRACTOR_DAMAGE", result["outcome"])

    def test_a_hyphen_the_extractor_ate_is_extractor_damage(self):
        result = server.verify_quote(
            "A risk-based approach is applied across the Regulation", paper_id="paper-b")
        self.assertEqual("VERBATIM_RELAXED_EXTRACTOR_DAMAGE", result["outcome"])

    def test_a_mistyped_quote_is_relocated_and_carries_a_diff(self):
        result = server.verify_quote(
            "Regulatory sandboxs are establshed under Artcle 57 of the Regulaton",
            paper_id="paper-a",
        )
        self.assertEqual("PASSAGE_RELOCATED_QUOTE_DIFFERS", result["outcome"])
        self.assertIn("your_text", result)
        self.assertIn("document_text", result)
        self.assertIn("char_diff", result)

    def test_a_fabricated_passage_is_not_locatable(self):
        result = server.verify_quote(
            "The authors report a ninety two percent reduction in fabricated citations",
            paper_id="paper-a",
        )
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        self.assertIsNone(result["char_interval"])
        self.assertIn("best_score", result)
        self.assertIn("scorer", result)

    def test_a_string_in_two_papers_names_both(self):
        result = server.verify_quote("Regulatory sandboxes are established under Article 57")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        self.assertEqual("paper-a", result["paper_id"])
        self.assertEqual(["paper-b"], result["also_occurs_in"])

    def test_a_short_string_is_not_locatable_with_the_reason_given(self):
        result = server.verify_quote("Article 57", paper_id="paper-a")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        reasons = " ".join(a["reason"] for a in result["not_available"])
        self.assertIn("shorter than", reasons)

    def test_an_unknown_paper_is_source_unknown(self):
        result = server.verify_quote("Regulatory sandboxes are established", paper_id="paper-zz")
        self.assertEqual("SOURCE_UNKNOWN", result["outcome"])

    def test_a_paper_with_no_text_layer_is_source_not_held(self):
        self.use_corpus(FakeCorpus({"scan-1": ""}, unheld={"scan-1"}))
        result = server.verify_quote("Regulatory sandboxes are established", paper_id="scan-1")
        self.assertEqual("SOURCE_NOT_HELD", result["outcome"])

    def test_an_empty_corpus_is_source_not_held(self):
        self.use_corpus(FakeCorpus({}))
        result = server.verify_quote("Regulatory sandboxes are established")
        self.assertEqual("SOURCE_NOT_HELD", result["outcome"])

    def test_an_unreadable_corpus_is_corpus_unavailable(self):
        self.broken_corpus()
        result = server.verify_quote("Regulatory sandboxes are established")
        self.assertEqual("CORPUS_UNAVAILABLE", result["outcome"])

    def test_a_stale_corpus_is_named_in_not_available(self):
        self.use_corpus(FakeCorpus({"paper-a": DOC_A}, stale=["paper-a: pdf_sha256 mismatch"]))
        result = server.verify_quote("Regulatory sandboxes are established under Article 57")
        fields = [a["field"] for a in result["not_available"]]
        self.assertIn("corpus_freshness", fields)

    def test_a_missing_page_index_is_named_rather_than_guessed(self):
        self.use_corpus(FakeCorpus({"paper-a": DOC_A}, no_pages={"paper-a"}))
        result = server.verify_quote("Regulatory sandboxes are established under Article 57")
        self.assertIsNone(result["page"])
        self.assertIn("page", [a["field"] for a in result["not_available"]])

    def test_an_out_of_range_threshold_falls_back_and_says_so(self):
        result = server.verify_quote(
            "Regulatory sandboxes are established under Article 57",
            paper_id="paper-a",
            t_locate=9.9,
        )
        self.assertIn("t_locate", [a["field"] for a in result["not_available"]])


# ---------------------------------------------------------------------------
# find_passage
# ---------------------------------------------------------------------------


class TestFindPassage(ServerTestCase):
    PHRASE = "Regulatory sandboxes are established under Article 57"

    def test_it_finds_the_phrase_in_every_document_holding_it(self):
        result = server.find_passage(self.PHRASE)
        self.assertEqual("OK", result["outcome"])
        self.assertEqual(2, result["hit_count"])
        self.assertEqual(["paper-a", "paper-b"], [h["paper_id"] for h in result["hits"]])
        for hit in result["hits"]:
            self.assertIn(hit["outcome"], {o.value for o in env.VERBATIM_OUTCOMES})
            self.assertIn("Regulatory sandboxes", hit["document_text"])

    def test_no_occurrence_is_ok_with_zero_hits_not_an_error(self):
        result = server.find_passage("a phrase that is nowhere in this corpus at all")
        self.assertEqual("OK", result["outcome"])
        self.assertEqual(0, result["hit_count"])
        reasons = " ".join(a["reason"] for a in result["not_available"])
        self.assertIn("result and not a failure", reasons)

    def test_a_paraphrase_returns_nothing_because_this_is_not_a_ranker(self):
        # A fuzzy window would make this a ranking engine over the corpus.
        self.pin_tier(Tier.T3_LOCATED, offset=0, length=30, score=0.9, diff="a diff")
        result = server.find_passage(self.PHRASE)
        self.assertEqual("OK", result["outcome"])
        self.assertEqual(0, result["hit_count"])

    def test_paper_ids_restrict_the_search(self):
        result = server.find_passage(self.PHRASE, paper_ids=["paper-b"])
        self.assertEqual(["paper-b"], [h["paper_id"] for h in result["hits"]])

    def test_an_unknown_id_is_named_and_the_rest_still_runs(self):
        result = server.find_passage(self.PHRASE, paper_ids=["paper-a", "paper-zz"])
        self.assertEqual("OK", result["outcome"])
        self.assertIn("document:paper-zz", [a["field"] for a in result["not_available"]])

    def test_all_ids_unknown_is_source_unknown(self):
        result = server.find_passage(self.PHRASE, paper_ids=["paper-yy", "paper-zz"])
        self.assertEqual("SOURCE_UNKNOWN", result["outcome"])

    def test_max_hits_caps_and_says_how_many_were_left_out(self):
        result = server.find_passage(self.PHRASE, max_hits=1)
        self.assertEqual(1, result["hit_count"])
        reasons = " ".join(a["reason"] for a in result["not_available"])
        self.assertIn("not listed", reasons)

    def test_max_hits_is_clamped(self):
        result = server.find_passage(self.PHRASE, max_hits=0)
        self.assertIn("max_hits", [a["field"] for a in result["not_available"]])

    def test_a_short_query_is_not_locatable_rather_than_zero_hits(self):
        result = server.find_passage("Article 57")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        self.assertIsNone(result["char_interval"])

    def test_an_unreadable_corpus_is_corpus_unavailable(self):
        self.broken_corpus()
        result = server.find_passage(self.PHRASE)
        self.assertEqual("CORPUS_UNAVAILABLE", result["outcome"])

    def test_an_empty_corpus_is_source_unknown(self):
        self.use_corpus(FakeCorpus({}))
        result = server.find_passage(self.PHRASE)
        self.assertEqual("SOURCE_UNKNOWN", result["outcome"])

    def test_hits_are_ordered_by_identifier_not_by_score(self):
        result = server.find_passage(self.PHRASE)
        ids = [h["paper_id"] for h in result["hits"]]
        self.assertEqual(sorted(ids), ids)


# ---------------------------------------------------------------------------
# the four anti-blur rules, checked against real tool output
# ---------------------------------------------------------------------------


def every_scenario(case: ServerTestCase) -> dict[str, dict]:
    """One response per reachable outcome, from the tools themselves."""
    phrase = "Regulatory sandboxes are established under Article 57"
    out: dict[str, dict] = {}

    def record(response):
        out.setdefault(response["outcome"], response)

    record(server.verify_quote(phrase, paper_id="paper-a"))
    record(server.verify_quote(
        "human oversight is required for every high risk system", paper_id="paper-c"))
    record(server.verify_quote(
        "Regulatory sandboxs are establshed under Artcle 57 of the Regulaton",
        paper_id="paper-a"))
    record(server.verify_quote(
        "The authors report a ninety two percent reduction in citations",
        paper_id="paper-a"))
    record(server.verify_quote("Article 57", paper_id="paper-a"))
    record(server.verify_quote(phrase, paper_id="paper-zz"))
    record(server.find_passage(phrase))
    record(server.find_passage("a phrase that is nowhere in this corpus at all"))
    record(server.find_passage(phrase, paper_ids=["paper-yy"]))

    case.use_corpus(FakeCorpus({"scan-1": ""}, unheld={"scan-1"}))
    record(server.verify_quote(phrase, paper_id="scan-1"))
    case.broken_corpus()
    record(server.verify_quote(phrase))
    record(server.find_passage(phrase))
    return out


class TestAntiBlurRulesOnRealResponses(ServerTestCase):
    def scenarios(self):
        return every_scenario(self)

    def test_no_code_path_emits_a_string_outside_the_vocabulary(self):
        seen = set()
        for response in self.scenarios().values():
            for _path, key, value in env._walk(response):
                if key == "outcome":
                    seen.add(value)
        self.assertTrue(seen)
        self.assertTrue(
            seen <= env.OUTCOME_NAMES,
            "outcomes outside the vocabulary: %s" % sorted(seen - env.OUTCOME_NAMES),
        )

    def test_every_reachable_outcome_is_exercised(self):
        seen = set(self.scenarios())
        expected = env.OUTCOME_NAMES
        self.assertEqual(
            set(),
            expected - seen,
            "these outcomes were never produced by a tool: %s" % sorted(expected - seen),
        )

    def test_rule_one_a_refusal_has_nothing_to_copy(self):
        refusals = {o.value for o in env.REFUSAL_OUTCOMES}
        checked = 0
        for name, response in self.scenarios().items():
            if name not in refusals:
                continue
            checked += 1
            for path, key, _value in env._walk(response):
                self.assertNotIn(
                    key, env.QUOTE_FIELD_NAMES, "%s carried %r at %s" % (name, key, path)
                )
        self.assertGreaterEqual(checked, 4)

    def test_rule_one_not_locatable_carries_no_document_words(self):
        response = self.scenarios()["NOT_LOCATABLE"]
        self.assertNotIn("document_text", response)
        self.assertIn("your_text", response)

    def test_rule_one_relocated_carries_the_three_named_fields(self):
        response = self.scenarios()["PASSAGE_RELOCATED_QUOTE_DIFFERS"]
        for field in ("document_text", "your_text", "char_diff"):
            self.assertIn(field, response)

    def test_rule_two_the_word_only_appears_in_the_two_outcome_names(self):
        for name, response in self.scenarios().items():
            for path, text in env.iter_strings(response, system_only=True):
                self.assertTrue(
                    env._verbatim_uses_are_outcome_names(text),
                    "%s used the word in prose at %s" % (name, path),
                )

    def test_rule_three_not_locatable_is_explicit_about_having_nothing(self):
        response = self.scenarios()["NOT_LOCATABLE"]
        self.assertIn("char_interval", response)
        self.assertIsNone(response["char_interval"])
        self.assertIn("best_score", response)
        self.assertIn("scorer", response)

    def test_rule_four_no_response_asserts_what_a_passage_does(self):
        for name, response in self.scenarios().items():
            for path, text in env.iter_strings(response, system_only=True):
                lowered = text.lower()
                for term in env.BANNED_TERMS:
                    self.assertNotIn(term, lowered, "%s used %r at %s" % (name, term, path))

    def test_no_response_carries_a_boolean(self):
        for name, response in self.scenarios().items():
            for path, _key, value in env._walk(response):
                self.assertNotIsInstance(value, bool, "%s at %s" % (name, path))

    def test_every_response_carries_the_notice_provenance_and_absences(self):
        for name, response in self.scenarios().items():
            self.assertEqual(env.INTERPRETATION_NOTICE, response["interpretation_notice"], name)
            self.assertEqual(
                {"normaliser", "verifier", "scorer", "server"}, set(response["provenance"]), name
            )
            self.assertIsInstance(response["not_available"], list, name)


class TestTheExtractionHeaderIsNotTheDocument(ServerTestCase):
    """A header carries this toolchain's words, not the paper's."""

    TITLE = "Regulatory sandboxes and the question of the wrong author"

    def setUp(self):
        super().setUp()
        self.use_corpus(FakeCorpus({"paper-h": DOC_H}, pages={"paper-h": [BODY_H]}))

    def test_a_string_only_in_the_header_is_not_reported_as_located(self):
        result = server.verify_quote(self.TITLE, paper_id="paper-h")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        self.assertNotIn("document_text", result)
        reasons = " ".join(a["reason"] for a in result["not_available"])
        self.assertIn("extraction header", reasons)

    def test_find_passage_does_not_list_a_header_only_match(self):
        result = server.find_passage(self.TITLE, paper_ids=["paper-h"])
        self.assertEqual("OK", result["outcome"])
        self.assertEqual(0, result["hit_count"])

    def test_the_body_is_still_searched_normally(self):
        result = server.verify_quote(
            "This body never contains that title phrase", paper_id="paper-h")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        self.assertEqual(1, result["page"])

    def test_a_document_with_no_header_is_not_truncated(self):
        # No header sentinel, so there is no header and nothing is skipped.
        two_pages = ("First page mentions regulatory sandboxes.\f"
                     "Second page mentions oversight duties.")
        self.use_corpus(FakeCorpus({"plain": two_pages}))
        result = server.verify_quote("First page mentions regulatory sandboxes", paper_id="plain")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        self.assertEqual(1, result["page"])

    def test_a_hand_made_pdftotext_dump_keeps_its_first_page(self):
        # pdftotext writes a form feed after the LAST page too, so a plain
        # dump holds exactly as many form feeds as it has pages, which is also
        # what a document with a header holds. Counting them therefore cannot
        # tell the two apart, and an earlier version of _header_end that
        # counted them threw away the whole of page one of every hand made
        # dump. The sentinel is the test.
        dump = ("First page mentions regulatory sandboxes." + PAGE_BREAK
                + "Second page mentions oversight duties." + PAGE_BREAK)
        self.use_corpus(FakeCorpus({"dump": dump}))
        self.assertEqual(0, server._header_end(dump))
        result = server.verify_quote("First page mentions regulatory sandboxes", paper_id="dump")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        self.assertEqual(1, result["page"])

    def test_the_header_boundary_agrees_with_the_document_format(self):
        # server._header_end and extract.split_document must never disagree
        # about where the header ends, so both are asked the same question.
        for name, raw in (("with header", DOC_H),
                          ("no header", "Body only." + PAGE_BREAK),
                          ("empty", "")):
            with self.subTest(name):
                header, _pages = split_document(raw)
                expected = len(header) + len(PAGE_BREAK) if header else 0
                self.assertEqual(expected, server._header_end(raw))

    def test_the_sentinel_is_what_marks_a_header(self):
        self.assertIn(HEADER_SENTINEL, DOC_H)
        self.assertGreater(server._header_end(DOC_H), 0)


class TestReadableButNothingFound(ServerTestCase):
    def test_it_is_not_locatable_rather_than_source_not_held(self):
        # "nothing matched" and "there was nothing to match against" are
        # different answers and need different actions from the human.
        self.pin_tier(Tier.T1_EXACT, offset=10_000_000, length=10)
        result = server.verify_quote("Regulatory sandboxes are established under Article 57")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])

    def test_an_unreadable_text_file_is_named_per_document(self):
        self.use_corpus(FakeCorpus({"paper-a": DOC_A, "bad": ""}, unreadable={"bad"}))
        result = server.verify_quote("Regulatory sandboxes are established under Article 57")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        self.assertIn("document:bad", [a["field"] for a in result["not_available"]])


class TestStatelessness(ServerTestCase):
    def test_the_only_cache_is_keyed_by_content_hash(self):
        server._PREPARED_CACHE.clear()
        phrase = "Regulatory sandboxes are established under Article 57"
        server.verify_quote(phrase, paper_id="paper-a")
        first = dict(server._PREPARED_CACHE)
        server.verify_quote(phrase, paper_id="paper-a")
        self.assertEqual(set(first), set(server._PREPARED_CACHE))
        # Different characters, different key. Nothing is reused across bytes.
        self.use_corpus(FakeCorpus({"paper-a": DOC_A + " An added sentence."}))
        server.verify_quote(phrase, paper_id="paper-a")
        self.assertEqual(len(first) + 1, len(server._PREPARED_CACHE))

    def test_repeated_calls_return_identical_envelopes(self):
        phrase = "Regulatory sandboxes are established under Article 57"
        self.assertEqual(
            server.verify_quote(phrase, paper_id="paper-a"),
            server.verify_quote(phrase, paper_id="paper-a"),
        )

    def test_a_changed_document_changes_the_answer(self):
        phrase = "Regulatory sandboxes are established under Article 57"
        self.assertEqual(
            "VERBATIM_EXACT", server.verify_quote(phrase, paper_id="paper-a")["outcome"])
        self.use_corpus(FakeCorpus(
            {"paper-a": "An entirely different document about something else."}))
        self.assertEqual(
            "NOT_LOCATABLE", server.verify_quote(phrase, paper_id="paper-a")["outcome"])


class TestALineBreakHyphenTheCallerKept(ServerTestCase):
    """The caller's copy of a quote carries damage this corpus does not.

    The corpus holds a reading-order extraction, which rejoined a word broken
    across a line: it says "actionable". A caller who ran their own extractor,
    or copied out of a viewer, hands over "action- able" with the break still
    in it. That is the same extractor damage T2 exists for, seen from the
    caller's side, and it must return the DOCUMENT's characters rather than
    the caller's, or the response would show the caller their own text back.

    Measured on 240 sentences lifted from an independent extraction of 12
    corpus PDFs: 31 of them, 12.9%, differed from the document by nothing but
    this.
    """

    DOC = ("ECCOLA is intended to provide developers an actionable tool for "
           "implementing AI ethics in practice.")

    def setUp(self):
        super().setUp()
        self.use_corpus(FakeCorpus({"paper-e": self.DOC}))

    def test_it_reaches_the_relaxed_identity_tier(self):
        result = server.verify_quote(
            "provide developers an action- able tool", paper_id="paper-e")
        self.assertEqual("VERBATIM_RELAXED_EXTRACTOR_DAMAGE", result["outcome"])

    def test_the_characters_returned_are_the_documents_own(self):
        result = server.verify_quote(
            "provide developers an action- able tool", paper_id="paper-e")
        self.assertEqual("provide developers an actionable tool",
                         result["document_text"])
        self.assertNotIn("action- able", result["document_text"])

    def test_the_interval_points_at_those_characters(self):
        result = server.verify_quote(
            "provide developers an action- able tool", paper_id="paper-e")
        interval = result["char_interval"]
        self.assertEqual(len(result["document_text"]),
                         interval["end_pos"] - interval["start_pos"])

    def test_a_word_the_document_does_not_hold_is_still_refused(self):
        # The fold deletes a hyphen. It never joins two words that had none,
        # so this must not become an identity outcome.
        result = server.verify_quote(
            "provide developers an action able tool", paper_id="paper-e")
        self.assertNotIn(result["outcome"],
                         ("VERBATIM_EXACT", "VERBATIM_RELAXED_EXTRACTOR_DAMAGE"))


class TestTheAttributionTrap(ServerTestCase):
    """A located passage whose claim belongs to another work.

    The deepest hole in a locator that only decides character identity. Paper A
    writes "eating oranges is good [12]". Every field of the response is true,
    the outcome asserts character identity, and a caller can still write "A
    says eating oranges is good" when A was citing B. This server does not
    resolve the marker, which is deferred, but it does report that one is
    there, and it reports the case that used to be invisible: a quote that
    stops one character before the marker.

    Drawn from a real corpus, restated inline. A paper defined warranted trust
    and ended the sentence with a reference number; a quote of the definition
    that stopped before the number came back with nothing to notice at all.
    """

    CITING = (
        "The trust is warranted if it is the result of trustworthiness [53]. "
        "A long stretch of the paper's own prose follows, carrying no marker "
        "of any kind, so that the next sentence sits well clear of the one "
        "above it and the window either side of it holds nothing. "
        "Warranted trust is what this section is about, and it needs no "
        "further attribution here."
    )

    def setUp(self):
        super().setUp()
        self.use_corpus(FakeCorpus({"citing": self.CITING}))

    def test_a_marker_inside_the_located_characters_is_reported(self):
        result = server.verify_quote(
            "The trust is warranted if it is the result of trustworthiness [53]",
            paper_id="citing")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        self.assertEqual(1, result["citation_markers"]["in_located_characters"])

    def test_a_marker_just_outside_the_span_is_reported_separately(self):
        # The case that used to be invisible: the caller's quote stops one
        # character short of the marker, so document_text looks clean.
        result = server.verify_quote(
            "The trust is warranted if it is the result of trustworthiness",
            paper_id="citing")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        self.assertNotIn("[53]", result["document_text"])
        self.assertEqual(0, result["citation_markers"]["in_located_characters"])
        self.assertEqual(1, result["citation_markers"]["in_surrounding_characters"])

    def test_a_passage_with_no_marker_anywhere_near_reports_zero(self):
        result = server.verify_quote(
            "Warranted trust is what this section is about", paper_id="citing")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        markers = result["citation_markers"]
        self.assertEqual(0, markers["in_located_characters"])
        self.assertEqual(0, markers["in_surrounding_characters"])

    def test_find_passage_reports_the_same_counts_on_every_hit(self):
        result = server.find_passage(
            "The trust is warranted if it is the result of trustworthiness")
        self.assertEqual("OK", result["outcome"])
        self.assertEqual(1, result["hit_count"])
        self.assertEqual(
            1, result["hits"][0]["citation_markers"]["in_surrounding_characters"])

    def test_a_relocated_passage_carries_the_counts_too(self):
        # A quote that differs from the document still names a location, and a
        # caller can act on that location, so it needs the same warning.
        result = server.verify_quote(
            "The trust is warranted when it is the result of trustworthiness",
            paper_id="citing")
        self.assertEqual("PASSAGE_RELOCATED_QUOTE_DIFFERS", result["outcome"])
        self.assertIn("citation_markers", result)

    def test_a_refusal_carries_no_counts_because_nothing_was_located(self):
        result = server.verify_quote(
            "A sentence that appears in no document held here at all",
            paper_id="citing")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        self.assertNotIn("citation_markers", result)

    def test_the_counts_are_never_reported_as_a_boolean(self):
        # env.validate bans a boolean anywhere in an envelope. A count of one
        # marker must not degrade into "there is a marker": the whole point is
        # that the caller sees how many and where.
        result = server.verify_quote(
            "The trust is warranted if it is the result of trustworthiness [53]",
            paper_id="citing")
        for value in result["citation_markers"].values():
            self.assertNotIsInstance(value, bool)


# ---------------------------------------------------------------------------
# reference-list contamination, and the two signals that report it
# ---------------------------------------------------------------------------

#: A whole small paper: body, a heading, and a bibliography holding OTHER
#: papers' titles. Every string here is inline; nothing reads the real corpus.
PAPER_WITH_A_BIBLIOGRAPHY = (
    "Trust is warranted if it is the result of trustworthiness [53]. "
    "A long stretch of the paper's own prose follows here, carrying no marker "
    "of any kind, running on far enough that the window either side of the "
    "next sentence holds nothing whatever. "
    "Oranges are good for you (Smith et al., 2019). "
    "This closing sentence of the body carries nothing at all.\n"
    "\n"
    "References\n"
    "\n"
    "[1] J. A. Smith and K. Jones. A study of oranges in winter. Journal of "
    "Fruit, 40(7):936 to 948, 2019. doi:10.1000/abc\n"
    "[2] L. M. Brown and N. White. Warranted trust in machines. Proceedings "
    "of the Conference on Trust, pp. 12 to 30, 2020.\n"
)

#: A paper that cites 92 times and whose every citation was destroyed by the
#: extractor. `pdftotext` glued each superscript onto the word before it, so
#: nothing in the body looks like a marker to any character test.
PAPER_CITING_WITH_SUPERSCRIPTS = (
    "Semantic Units are introduced here as a Principles4 based construct. "
    "The hallucination26,27 problem and FAIRness5,9 are treated at length, "
    "and the status25,26 of every entity36 in the model is recorded so that "
    "a reader can follow the argument without any further help.\n"
    "\n"
    "References\n"
    "\n"
    "1. Smith, J. A. A study of units. Journal of Units 13, 936 (2019). "
    "https://doi.org/10.1000/x\n"
    "2. Brown, L. M. and Green, N. More units. Nature Units 4, 12 (2020). "
    "https://doi.org/10.1000/y\n"
)


class TestTheReferenceRegionRidesOnEveryLocation(ServerTestCase):
    """Problem A: a bibliography holds OTHER papers' titles.

    Search a title, find it in the paper that CITES it, and a caller concludes
    the citing author wrote it. That is the wrong-author error, arriving by a
    route no character test can see, because the characters really are in that
    document.

    Two independent signals report it and are never merged into one number. A
    caller told "reference like: 0.83" cannot tell which test fired, and the
    two need different actions: region membership is precise and silent when
    the model did not apply, while the local-block test always answers and is
    noisy. Measured, region membership plus the local-block test catch 36 of 36
    known reference-list hits where the local-block test alone catches 31.
    """

    def setUp(self):
        super().setUp()
        self.use_corpus(FakeCorpus({"paper-a": PAPER_WITH_A_BIBLIOGRAPHY}))

    def locate(self, text):
        return server.verify_quote(text, paper_id="paper-a")

    def test_a_body_hit_is_labelled_outside_the_reference_region(self):
        result = self.locate("Oranges are good for you")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        self.assertEqual(
            "OUTSIDE_REFERENCE_REGION", result["reference_region"]["status"]
        )

    def test_a_title_located_in_the_bibliography_is_labelled_as_such(self):
        result = self.locate("A study of oranges in winter")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        self.assertEqual(
            "IN_REFERENCE_REGION", result["reference_region"]["status"]
        )

    def test_a_hit_in_the_reference_region_is_still_returned(self):
        """Never suppressed. Labelling a location is a fact; dropping one is a
        judgement, and this server does not make judgements. A caller who
        wanted the bibliography searched would otherwise be told the phrase is
        not in the document, which is false."""
        result = self.locate("A study of oranges in winter")
        self.assertIn("document_text", result)
        self.assertIn("A study of oranges in winter", result["document_text"])
        self.assertIsNotNone(result["char_interval"])

    def test_the_local_block_test_warns_on_the_same_hit(self):
        result = self.locate("A study of oranges in winter")
        local = result["reference_region"]["local_block"]
        self.assertEqual("REFERENCE_LIKE_BLOCK_WARNING", local["signal"])
        self.assertGreaterEqual(local["signals_found"], 3)

    def test_the_local_block_test_stays_quiet_on_the_body(self):
        result = self.locate("Oranges are good for you")
        self.assertEqual(
            "NO_REFERENCE_LIKE_BLOCK",
            result["reference_region"]["local_block"]["signal"],
        )

    def test_the_status_is_a_label_from_a_closed_set(self):
        result = self.locate("Oranges are good for you")
        block = result["reference_region"]
        self.assertIn(block["status"], block["status_vocabulary"])
        self.assertEqual(6, len(block["status_vocabulary"]))

    def test_the_region_version_travels_with_the_status(self):
        # A status means something only while the model that produced it is
        # unchanged, exactly as an offset means something only while the
        # normaliser is unchanged.
        result = self.locate("Oranges are good for you")
        self.assertIsInstance(result["reference_region"]["region_version"], str)

    def test_a_relocated_passage_carries_the_region_block_too(self):
        result = server.verify_quote(
            "Oranges are excellent for you", paper_id="paper-a")
        self.assertEqual("PASSAGE_RELOCATED_QUOTE_DIFFERS", result["outcome"])
        self.assertIn("reference_region", result)

    def test_every_find_passage_hit_carries_the_region_block(self):
        result = server.find_passage("A study of oranges in winter")
        self.assertEqual("OK", result["outcome"])
        self.assertEqual(1, result["hit_count"])
        self.assertEqual(
            "IN_REFERENCE_REGION",
            result["hits"][0]["reference_region"]["status"],
        )

    def test_a_refusal_carries_no_region_block_because_nothing_was_located(self):
        result = server.verify_quote(
            "A sentence that appears in no document held here at all",
            paper_id="paper-a")
        self.assertEqual("NOT_LOCATABLE", result["outcome"])
        self.assertNotIn("reference_region", result)

    def test_no_value_in_the_region_block_is_a_boolean(self):
        result = self.locate("A study of oranges in winter")
        pending = [result["reference_region"]]
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                pending.extend(node.values())
            elif isinstance(node, (list, tuple)):
                pending.extend(node)
            else:
                self.assertNotIsInstance(node, bool)


class TestTheCitationStyleThatCannotBeSeen(ServerTestCase):
    """The most dangerous state in the system, and the label that names it.

    2 of 47 papers cite with superscripts. `pdftotext` has no layout
    information, so it glues each one onto the word before it: Principles4,
    hallucination26,27, FAIRness5,9. One paper in the corpus has 92 such
    occurrences and not one bracketed marker anywhere. Those strings cannot be
    told apart from GPT4, COVID19 or Section3 without the layout the extractor
    threw away, so this server does not try to detect them.

    It detects their ABSENCE. Without the label, such a paper answered "no
    citation marker near this span", which is character-for-character
    indistinguishable from "this is the author's own uncited claim". That is a
    false negative wearing the shape of a positive finding, which is worse than
    a refusal and worse than silence.
    """

    def setUp(self):
        super().setUp()
        self.use_corpus(
            FakeCorpus({
                "superscript": PAPER_CITING_WITH_SUPERSCRIPTS,
                "bracketed": PAPER_WITH_A_BIBLIOGRAPHY,
            })
        )

    def locate(self, text, paper_id):
        return server.verify_quote(text, paper_id=paper_id)

    def test_the_superscript_paper_yields_a_marker_count_of_zero(self):
        result = self.locate(
            "The hallucination26,27 problem and FAIRness5,9 are treated",
            "superscript")
        self.assertEqual("VERBATIM_EXACT", result["outcome"])
        self.assertEqual(0, result["citation_markers"]["in_located_characters"])
        self.assertEqual(0, result["citation_markers"]["in_surrounding_characters"])

    def test_but_the_zero_is_labelled_as_carrying_no_information(self):
        result = self.locate(
            "The hallucination26,27 problem and FAIRness5,9 are treated",
            "superscript")
        self.assertEqual(
            "MARKER_STYLE_UNDETECTABLE",
            result["citation_markers"]["marker_style"],
        )

    def test_the_absence_is_named_in_not_available_with_its_reason(self):
        # Nothing is silently omitted. A count that carries no information is
        # an absence, and an absence has a line naming it and saying why.
        result = self.locate(
            "The hallucination26,27 problem and FAIRness5,9 are treated",
            "superscript")
        fields = [a["field"] for a in result["not_available"]]
        self.assertIn("citation_markers", fields)
        reason = next(
            a["reason"] for a in result["not_available"]
            if a["field"] == "citation_markers"
        )
        self.assertIn("superscript", reason)

    def test_a_paper_whose_markers_are_visible_is_not_labelled_that_way(self):
        result = self.locate("Oranges are good for you", "bracketed")
        self.assertNotEqual(
            "MARKER_STYLE_UNDETECTABLE",
            result["citation_markers"]["marker_style"],
        )
        self.assertNotIn(
            "citation_markers", [a["field"] for a in result["not_available"]]
        )

    def test_find_passage_names_the_document_the_absence_belongs_to(self):
        result = server.find_passage(
            "The hallucination26,27 problem and FAIRness5,9 are treated")
        self.assertEqual("OK", result["outcome"])
        self.assertIn(
            "citation_markers:superscript",
            [a["field"] for a in result["not_available"]],
        )

    def test_the_label_is_in_the_closed_set_on_every_document(self):
        for paper_id, text in (
            ("superscript", "The hallucination26,27 problem"),
            ("bracketed", "Oranges are good for you"),
        ):
            with self.subTest(paper_id=paper_id):
                style = self.locate(text, paper_id)["citation_markers"]["marker_style"]
                self.assertIn(style, markers_module.MARKER_STYLES)


class TestTheSentenceRidesOnEveryLocation(ServerTestCase):
    """The one primitive with a published precision of 1.00 behind it.

    Sarol et al., Bioinformatics 2024;40(7):btae420: the trivial "return the
    sentence containing the marker" baseline scored P 1.00 / R 0.90 / F1 0.94,
    beating a fine-tuned PubMedBERT. It is strong because it decides nothing.
    Returning the sentence hands the human the unit the marker attaches to and
    leaves the judgement where it belongs.
    """

    def setUp(self):
        super().setUp()
        self.use_corpus(FakeCorpus({"paper-a": PAPER_WITH_A_BIBLIOGRAPHY}))

    def test_the_sentence_is_returned_under_document_text(self):
        result = server.verify_quote(
            "Oranges are good for you", paper_id="paper-a")
        sentence = result["citation_markers"]["sentence"]["document_text"]
        self.assertEqual("Oranges are good for you (Smith et al., 2019).", sentence)

    def test_the_sentence_shows_the_marker_a_short_quote_stopped_before(self):
        # The case that used to be invisible: document_text looks clean, and
        # the marker that owns the claim sits just outside it.
        result = server.verify_quote(
            "Oranges are good for you", paper_id="paper-a")
        self.assertNotIn("Smith", result["document_text"])
        self.assertIn("Smith", result["citation_markers"]["sentence"]["document_text"])

    def test_the_sentence_interval_indexes_the_normalised_document(self):
        result = server.verify_quote(
            "Oranges are good for you", paper_id="paper-a")
        interval = result["citation_markers"]["sentence"]["char_interval"]
        self.assertEqual("normalised_document", interval["basis"])
        self.assertLessEqual(
            interval["start_pos"], result["char_interval"]["start_pos"]
        )
        self.assertGreaterEqual(
            interval["end_pos"], result["char_interval"]["end_pos"]
        )

    def test_every_find_passage_hit_carries_its_sentence(self):
        result = server.find_passage("Oranges are good for you")
        self.assertIsNotNone(
            result["hits"][0]["citation_markers"]["sentence"]["document_text"]
        )


class TestTheTwoKindsOfMarkerAreCountedApart(ServerTestCase):
    """A bracketed number and an author-year parenthetical are different facts.

    They have different precision behind them, 72/72 for the bracketed pattern
    and 95.6% for the strict author-year one, so merging the counts would hide
    which of the two fired and with it how much the count is worth.
    """

    def setUp(self):
        super().setUp()
        self.use_corpus(FakeCorpus({"paper-a": PAPER_WITH_A_BIBLIOGRAPHY}))

    def test_a_bracketed_marker_is_counted_as_bracketed(self):
        result = server.verify_quote(
            "Trust is warranted if it is the result of trustworthiness [53]",
            paper_id="paper-a")
        block = result["citation_markers"]
        self.assertEqual(1, block["bracketed_in_located_characters"])
        self.assertEqual(0, block["author_year_in_located_characters"])

    def test_an_author_year_marker_is_counted_as_author_year(self):
        result = server.verify_quote(
            "Oranges are good for you (Smith et al., 2019)", paper_id="paper-a")
        block = result["citation_markers"]
        self.assertEqual(0, block["bracketed_in_located_characters"])
        self.assertEqual(1, block["author_year_in_located_characters"])

    def test_the_totals_are_the_sum_of_the_two_kinds(self):
        result = server.verify_quote(
            "Oranges are good for you (Smith et al., 2019)", paper_id="paper-a")
        block = result["citation_markers"]
        self.assertEqual(
            block["in_located_characters"],
            block["bracketed_in_located_characters"]
            + block["author_year_in_located_characters"],
        )
        self.assertEqual(
            block["in_surrounding_characters"],
            block["bracketed_in_surrounding_characters"]
            + block["author_year_in_surrounding_characters"],
        )


class TestABuildWithoutTheRegionModel(ServerTestCase):
    """The region model is optional, and its absence is reported, not assumed.

    A missing signal must never read as a signal that fired negative. Without
    the model every location comes back REGION_UNKNOWN, which is a different
    answer from OUTSIDE_REFERENCE_REGION, and the absence is named in
    not_available so a caller can see that a whole test did not run.
    """

    def setUp(self):
        super().setUp()
        self.use_corpus(FakeCorpus({"paper-a": PAPER_WITH_A_BIBLIOGRAPHY}))
        self._saved_region = markers_module._region_module
        markers_module._region_module = lambda: None

    def tearDown(self):
        markers_module._region_module = self._saved_region
        super().tearDown()

    def test_the_status_is_region_unknown_and_not_outside(self):
        result = server.verify_quote(
            "A study of oranges in winter", paper_id="paper-a")
        self.assertEqual("REGION_UNKNOWN", result["reference_region"]["status"])

    def test_the_missing_model_is_named_in_not_available(self):
        result = server.verify_quote(
            "A study of oranges in winter", paper_id="paper-a")
        self.assertIn(
            "reference_region", [a["field"] for a in result["not_available"]]
        )

    def test_the_local_block_test_still_answers_on_its_own(self):
        # The second signal is independent by construction, so it must survive
        # the first one being unavailable.
        result = server.verify_quote(
            "A study of oranges in winter", paper_id="paper-a")
        self.assertEqual(
            "REFERENCE_LIKE_BLOCK_WARNING",
            result["reference_region"]["local_block"]["signal"],
        )

    def test_the_style_falls_back_to_not_assessed_rather_than_undetectable(self):
        # Without a region there is no evidence the paper cites anything, so
        # UNDETECTABLE cannot be claimed and the label says so instead.
        self.use_corpus(
            FakeCorpus({"superscript": PAPER_CITING_WITH_SUPERSCRIPTS}))
        result = server.verify_quote(
            "The hallucination26,27 problem and FAIRness5,9 are treated",
            paper_id="superscript")
        self.assertEqual(
            "MARKER_STYLE_NOT_ASSESSED",
            result["citation_markers"]["marker_style"],
        )


class TestTheDescriptionsTellTheModelAboutTheNewSignals(unittest.TestCase):
    """The descriptions are the reliability surface, so a new signal that the
    model is not told about is a signal that will not be read."""

    def test_both_descriptions_name_the_undetectable_style(self):
        for spec in server.TOOL_SPECS:
            with self.subTest(spec.name):
                self.assertIn("MARKER_STYLE_UNDETECTABLE", spec.description)
                self.assertIn("superscript", spec.description.lower())

    def test_both_descriptions_name_every_region_status(self):
        for spec in server.TOOL_SPECS:
            for status in markers_module.REGION_STATUS_NAMES:
                with self.subTest(spec=spec.name, status=status):
                    self.assertIn(status, spec.description)

    def test_both_descriptions_name_the_local_block_warning(self):
        for spec in server.TOOL_SPECS:
            with self.subTest(spec.name):
                self.assertIn("REFERENCE_LIKE_BLOCK_WARNING", spec.description)

    def test_both_descriptions_say_a_reference_region_hit_is_still_returned(self):
        for spec in server.TOOL_SPECS:
            with self.subTest(spec.name):
                self.assertIn("never dropped", spec.description)

    def test_both_descriptions_state_the_attribution_ceiling_with_its_numbers(self):
        for spec in server.TOOL_SPECS:
            with self.subTest(spec.name):
                self.assertIn("0.18 to 0.31", spec.description)

    def test_verify_quote_names_the_sentence_and_its_precision(self):
        text = server.VERIFY_QUOTE_DESCRIPTION
        self.assertIn("sentence", text.lower())
        self.assertIn("1.00", text)

    def test_the_new_prose_obeys_the_same_language_rules(self):
        for spec in server.TOOL_SPECS:
            lowered = spec.description.lower()
            for term in env.BANNED_TERMS:
                self.assertNotIn(term, lowered, "%s used %r" % (spec.name, term))
            self.assertTrue(
                env._verbatim_uses_are_outcome_names(spec.description), spec.name
            )


if __name__ == "__main__":
    unittest.main()
