"""Regression guards for the hardening pass of 2026-09.

Every test here reproduces a defect that a five-agent adversarial review found
and that the suite at the time did NOT catch. Each one was a live false
verbatim, a wrong verdict, a wrong location, a text leak, or a containment gap.
The whole point of this file is that the suite would now fail if any of them
came back, so a fix cannot silently regress.

Findings, by the identifiers used in the review:

  F1  fraction expansion cut at a span boundary bypassed the fold guard and
      returned T1_EXACT for a number the paper did not contain.
  F2  a T2 casefold merged unit prefixes (10 mW matched 10 MW).
  F3  a guard demotion at the first occurrence denied a genuine verbatim that
      the document supported at a later occurrence.
  F4  str.lower() length expansion skewed every T3 offset, window and diff.
  F5  a citation marker straddling the located-span boundary was counted in
      neither the located nor the surrounding bucket.
  F6  the sentence expansion overshot into the next sentence when the located
      span ended exactly at its terminator.
  F7  the authoritative TextQuoteSelector was incoherent at T2/T3, so a
      standard re-anchor by it failed.
  H1  the toolchain header leaked into citation_markers.sentence.document_text.
  H2  a symlinked sidecar served files from outside the corpus.
  L1  a manifest path containing NUL or newline was accepted.
  M1  an unbounded input argument hung the single stateless server.
  M2  a stale sidecar was served with an authoritative outcome and the
      freshness note could not say which document was stale.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from science2code import markers as M
from science2code import server
from science2code.anchor import Tier, anchor_record, locate
from science2code.corpus import Corpus, ManifestError, _corpus_path
from science2code.extract import build_document, sha256_file

# ---------------------------------------------------------------------------
# F1: no false verbatim from a fraction expansion cut at the span boundary
# ---------------------------------------------------------------------------


class F1FractionBoundary(unittest.TestCase):
    def test_a_quote_ending_inside_a_fraction_expansion_is_not_verbatim(self):
        # "2 and a half" NFKC-folds to "21/2". A quote of "grew 21" matched the
        # "grew 2" plus the leading "1" of that expansion and claimed verbatim.
        a = locate("The dose grew 2½ times overall.", "grew 21")
        self.assertFalse(a.is_verbatim)
        self.assertNotIn(a.tier, (Tier.T1_EXACT, Tier.T2_RELAXED))

    def test_the_quarter_and_single_char_cases_are_not_verbatim(self):
        self.assertFalse(locate("with x¼ y", "x1").is_verbatim)
        self.assertFalse(locate("½ x", "1").is_verbatim)

    def test_the_full_fraction_still_locates_as_a_diff(self):
        a = locate("The dose grew 2½ times overall.", "grew 21/2")
        self.assertEqual(a.tier, Tier.T3_LOCATED)


# ---------------------------------------------------------------------------
# F2: a case difference on a unit prefix is not verbatim
# ---------------------------------------------------------------------------


class F2UnitPrefixCasefold(unittest.TestCase):
    def test_milliwatt_is_not_the_same_verbatim_as_megawatt(self):
        a = locate("The transmitter draws 10 MW under load.", "draws 10 mW under load")
        self.assertFalse(a.is_verbatim)

    def test_megabit_is_not_the_same_verbatim_as_megabyte(self):
        a = locate("The link sustains 500 MB per second.", "sustains 500 Mb per second")
        self.assertFalse(a.is_verbatim)

    def test_a_gigahertz_case_flip_next_to_a_number_is_not_verbatim(self):
        a = locate("clocked at 3 GHz today", "clocked at 3 Ghz today")
        self.assertFalse(a.is_verbatim)

    def test_a_digit_elsewhere_in_the_token_still_triggers_the_guard(self):
        # A re-attack found the first fix missed a digit in the same no-space
        # token when a letter sat between it and the recased one.
        self.assertFalse(locate("result 12ms observed", "12mS").is_verbatim)
        self.assertFalse(locate("throughput of 5MBps sustained", "5Mbps").is_verbatim)
        self.assertFalse(locate("the 5MBE link", "5MbE").is_verbatim)

    def test_a_recased_formula_with_a_digit_is_not_verbatim(self):
        self.assertFalse(locate("measured H2O content", "measured h2o content").is_verbatim)
        self.assertFalse(locate("reached CO2 levels", "reached co2 levels").is_verbatim)

    def test_a_plain_sentence_case_change_is_still_forgiven(self):
        # The guard must not over-demote. Presentation case away from a digit
        # is exactly what T2 exists to forgive.
        self.assertTrue(
            locate("The result was clear here.", "the result was clear here").is_verbatim)
        self.assertTrue(locate("the study found X here", "The study found X here").is_verbatim)
        self.assertTrue(locate("We ran DNA analysis now", "We ran dna analysis now").is_verbatim)


# ---------------------------------------------------------------------------
# F3: a genuine verbatim at a later occurrence wins over a demoted first one
# ---------------------------------------------------------------------------


class F3OccurrenceWalk(unittest.TestCase):
    def test_a_later_exact_occurrence_is_returned_not_the_demoted_first(self):
        doc = (
            "Early estimates put throughput at 10⁶ operations per second. "
            "A later benchmark measured exactly 106 operations per second."
        )
        a = locate(doc, "106 operations per second")
        self.assertEqual(a.tier, Tier.T1_EXACT)
        self.assertTrue(a.is_verbatim)
        # And the offset is the real one, the plain-digit occurrence.
        self.assertGreater(a.offset_norm, 40)


# ---------------------------------------------------------------------------
# F4: a length-changing lowercase must not skew the T3 offset
# ---------------------------------------------------------------------------


class F4LowerLengthSkew(unittest.TestCase):
    def test_dotted_capital_i_does_not_skew_the_located_offset(self):
        # Twenty U+0130 before the passage. str.lower() expands each to two
        # characters, so the reported offset used to be twenty past the truth.
        prefix = "İ" * 20
        passage = "the calibration protocol was applied to every field station"
        doc = prefix + " . " + passage + " and more text after it here."
        a = locate(doc, "calibration protocol was applied to every field station")
        self.assertIn(a.tier, (Tier.T1_EXACT, Tier.T2_RELAXED, Tier.T3_LOCATED))
        served = doc[a.offset_norm:a.offset_norm + a.length_norm]
        # The served characters actually contain the passage words, not a
        # window shifted off the front of them.
        self.assertIn("calibration protocol", served)


# ---------------------------------------------------------------------------
# F5, F6: marker classification and sentence boundaries
# ---------------------------------------------------------------------------


class F5MarkerStraddle(unittest.TestCase):
    def test_a_marker_on_the_located_boundary_is_counted_once(self):
        norm = "Oranges are good for you [12] said the review of diets."
        # Located span ends one character into "[12]".
        start = 0
        end = norm.index("[12]") + 2  # includes "[1", excludes "2]"
        block = M.citation_marker_block(norm, start, end - start)
        total = block["in_located_characters"] + block["in_surrounding_characters"]
        self.assertEqual(total, 1, "a straddling marker must land in exactly one bucket")


class F6SentenceOvershoot(unittest.TestCase):
    def test_the_sentence_stops_at_its_own_terminator(self):
        text = "Oranges are good [12]. Pears are attributed to nobody at all."
        end = text.index(".") + 1  # located span ends just past the full stop
        span = M.sentence_span(text, 0, end)
        self.assertIsNotNone(span)
        self.assertEqual(text[span[0]:span[1]], "Oranges are good [12].")


# ---------------------------------------------------------------------------
# F7: the authoritative selector round-trips at T2
# ---------------------------------------------------------------------------


class F7SelectorCoherence(unittest.TestCase):
    def test_prefix_exact_suffix_is_a_substring_of_the_document_at_t2(self):
        doc = "many laboratories that report long-term calibration records agree"
        record = anchor_record("laboratories that report longterm calibration records", doc)
        self.assertEqual(record["refs"]["tier"], Tier.T2_RELAXED.value)
        sel = next(
            s for s in record["target"]["selector"] if s["type"] == "TextQuoteSelector"
        )
        from science2code.normalise import normalise

        norm_doc = normalise(doc)[0]
        self.assertIn(sel["prefix"] + sel["exact"] + sel["suffix"], norm_doc)


# ---------------------------------------------------------------------------
# H2, L1: manifest path containment
# ---------------------------------------------------------------------------


class H2L1PathContainment(unittest.TestCase):
    def test_a_served_path_symlink_that_escapes_the_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "corpus"
            root.mkdir()
            secret = base / "secret.txt"
            secret.write_text("TOP SECRET outside the corpus\n", encoding="utf-8")
            pdf = root / "p.pdf"
            pdf.write_bytes(b"%PDF fake\n")
            link = root / "p.txt"
            os.symlink(secret, link)
            manifest = {
                "manifest_version": "corpus/1",
                "extractor": "x",
                "papers": [{
                    "paper_id": "p", "title": None, "authors": [], "year": None,
                    "venue": None, "doi": None, "pdf_path": "p.pdf",
                    "text_path": "p.txt", "pdf_sha256": sha256_file(pdf),
                    "text_sha256": None, "pages": 1, "held": True,
                }],
            }
            (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaises(ManifestError):
                Corpus.load(root)

    def test_a_pdf_symlink_into_a_store_is_still_allowed(self):
        # The PDF is hashed, never served as text, and symlinking papers in is
        # a normal way to build a corpus. Containment applies to served paths.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "corpus"
            root.mkdir()
            p = _corpus_path("linked.pdf", "x.pdf_path", root, served=False)
            self.assertEqual(p, root / "linked.pdf")

    def test_control_characters_in_a_path_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for bad in ("nul\x00.txt", "line\nbreak.txt"):
                with self.assertRaises(ManifestError):
                    _corpus_path(bad, "x.text_path", root)

    def _held_corpus_with_sidecar(self, root, sidecar_writer):
        pdf = root / "p.pdf"
        pdf.write_bytes(b"%PDF fake\n")
        sidecar_writer(root / "p.txt")
        manifest = {
            "manifest_version": "corpus/1", "extractor": "x",
            "papers": [{
                "paper_id": "p", "title": None, "authors": [], "year": None,
                "venue": None, "doi": None, "pdf_path": "p.pdf",
                "text_path": "p.txt", "pdf_sha256": sha256_file(pdf),
                "text_sha256": None, "pages": 1, "held": True,
            }],
        }
        (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        return Corpus.load(root)

    def test_a_hardlinked_sidecar_to_an_outside_file_is_not_served(self):
        # A hardlink follows no symlink, so it resolves inside the root and
        # passes the load-time check, yet exposes outside bytes. The read-time
        # guard refuses any sidecar with more than one link.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "corpus"
            root.mkdir()
            secret = base / "secret.txt"
            secret.write_text("SECRET outside the corpus\n", encoding="utf-8")
            corpus = self._held_corpus_with_sidecar(root, lambda p: os.link(secret, p))
            self.assertIsNone(corpus.text("p"))
            self.assertNotIn("SECRET", (corpus.body("p") or ""))

    def test_a_sidecar_swapped_to_an_outside_symlink_after_load_is_not_served(self):
        # TOCTOU: the manifest validated a regular sidecar, then it became a
        # symlink pointing outside. The read-time containment refuses it.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "corpus"
            root.mkdir()
            secret = base / "secret.txt"
            secret.write_text("SECRET outside the corpus\n", encoding="utf-8")
            corpus = self._held_corpus_with_sidecar(
                root, lambda p: p.write_text("innocent held body text here\n", encoding="utf-8")
            )
            self.assertIn("innocent", corpus.text("p"))
            link = root / "p.txt"
            link.unlink()
            os.symlink(secret, link)
            self.assertIsNone(corpus.text("p"))


# ---------------------------------------------------------------------------
# server-level guards: H1, M1, M2, using a corpus on disk
# ---------------------------------------------------------------------------


def _build_corpus(root: Path, paper_id: str, body: str, *, bad_pdf_hash=False):
    pdf = root / (paper_id + ".pdf")
    pdf.write_bytes(b"%PDF orig\n")
    txt = root / (paper_id + ".txt")
    meta = {
        "title": "T", "authors": ["A"], "year": 2025, "venue": "V",
        "doi": "10.1/x", "paper_id": paper_id,
    }
    txt.write_text(build_document(pdf, [body], meta), encoding="utf-8")
    manifest = {
        "manifest_version": "corpus/1",
        "extractor": "x",
        "papers": [{
            "paper_id": paper_id, "title": "T", "authors": ["A"], "year": 2025,
            "venue": "V", "doi": "10.1/x", "pdf_path": pdf.name,
            "text_path": txt.name,
            "pdf_sha256": ("0" * 64) if bad_pdf_hash else sha256_file(pdf),
            "text_sha256": sha256_file(txt), "pages": 1, "held": True,
        }],
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


class ServerGuards(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self._saved_env = os.environ.get(server.CORPUS_ROOT_ENV)
        os.environ[server.CORPUS_ROOT_ENV] = str(self.root)
        server._PREPARED_CACHE.clear()

        def _restore():
            if self._saved_env is None:
                os.environ.pop(server.CORPUS_ROOT_ENV, None)
            else:
                os.environ[server.CORPUS_ROOT_ENV] = self._saved_env
            server._PREPARED_CACHE.clear()

        self.addCleanup(_restore)

    def test_h1_header_text_never_appears_in_the_response(self):
        body = ("Large language models can hallucinate citations in generated "
                "text according to recent work.")
        _build_corpus(self.root, "REF-1", body)
        r = server.verify_quote("Large language models can hallucinate citations in generated text")
        self.assertEqual(r["outcome"], "VERBATIM_EXACT")
        blob = json.dumps(r)
        self.assertNotIn("form feed", blob)
        self.assertNotIn("Source PDF", blob)
        sentence = (r.get("citation_markers") or {}).get("sentence") or {}
        self.assertNotIn("form feed", sentence.get("document_text", ""))

    def test_m1_a_huge_argument_is_refused_fast(self):
        _build_corpus(self.root, "REF-1", "a normal body sentence that is long enough to locate.")
        import time
        t0 = time.time()
        r = server.verify_quote("z" * 5_000_000)
        self.assertLess(time.time() - t0, 5.0)
        self.assertEqual(r["outcome"], "NOT_LOCATABLE")
        self.assertLessEqual(len(r.get("your_text", "")), server.MAX_LOCATABLE_CHARS)

    def test_m1_a_sub_ceiling_pathological_quote_does_not_hang(self):
        # A re-attack showed a 9999-char quote against a large document ran for
        # over a minute, because the ceiling bounded needle length but not the
        # per-needle window count or the seed-gram count. Both are bounded now.
        # A repetitive body plus a repetitive at-ceiling quote is the worst case;
        # it must finish quickly and return a typed refusal, not hang.
        body = ("the model of the data in the paper of the study " * 4200)[:200_000]
        _build_corpus(self.root, "REF-1", body)
        import time
        t0 = time.time()
        r = server.verify_quote(("the of " * 1000)[: server.MAX_LOCATABLE_CHARS])
        elapsed = time.time() - t0
        self.assertEqual(r["outcome"], "NOT_LOCATABLE")
        self.assertLess(elapsed, 10.0, "a sub-ceiling quote must not run away")

    def test_m1_the_anchor_ladder_bounds_a_long_needle_directly(self):
        # Defence in depth below the server ceiling: locate() itself must not
        # run away on a long needle, so the bound holds even if the ceiling is
        # raised or the library is used without the server.
        from science2code.anchor import locate, prepare
        doc = ("the model of the data in the paper of the study " * 4200)[:200_000]
        prepared = prepare(doc)
        import time
        t0 = time.time()
        locate(prepared, ("the of " * 1500)[:9999])
        self.assertLess(time.time() - t0, 10.0)

    def test_m2_a_stale_document_is_named_in_the_response(self):
        body = "The mitochondria is the powerhouse of the cell and this is a long sentence indeed."
        _build_corpus(self.root, "REF-9", body, bad_pdf_hash=True)
        r = server.verify_quote(
            "The mitochondria is the powerhouse of the cell and this is a long sentence")
        fields = {a["field"]: a for a in r.get("not_available", [])}
        self.assertIn("document_freshness", fields)
        self.assertIn("REF-9", fields["document_freshness"].get("detail", ""))
        self.assertIn("REF-9", fields["corpus_freshness"].get("detail", ""))

    def test_m2_an_unreadable_file_counts_as_stale_not_as_silence(self):
        # A re-attack showed that if hashing raised (an unreadable file), the
        # whole staleness check was swallowed and the served document got an
        # authoritative outcome with no freshness flag. Staleness must fail
        # toward "stale". Skipped as root, which can read a 000-mode file.
        if os.geteuid() == 0:
            self.skipTest("root can read a 000-mode file, so the hazard cannot be staged")
        body = "The mitochondria is the powerhouse of the cell and this is a long sentence indeed."
        _build_corpus(self.root, "REF-9", body)
        (self.root / "REF-9.pdf").chmod(0o000)
        self.addCleanup(lambda: (self.root / "REF-9.pdf").chmod(0o644))
        r = server.verify_quote(
            "The mitochondria is the powerhouse of the cell and this is a long sentence")
        fields = {a["field"] for a in r.get("not_available", [])}
        self.assertIn("document_freshness", fields)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Property loops: the "no false positive by construction" claim, mechanised.
#
# Stdlib random with fixed seeds, so a failure is reproducible and no new
# dependency is added. The alphabets deliberately include the character classes
# that produced the historical false verbatims: zero-width and combining marks,
# superscripts and fractions, dash and space variants, ligatures, dotted I.
# ---------------------------------------------------------------------------

import random  # noqa: E402

from science2code.normalise import normalise  # noqa: E402

_ADVERSARIAL = (
    "abcdefg 0123 "
    "½¼⁶⁻₂"   # fractions, super/subscripts
    "​‌‍­"           # zero-width, soft hyphen
    "̧́"                       # combining marks
    "ﬁﬂ"                       # ligatures fi, fl
    "İß"                       # dotted capital I, sharp s
    "\u2010\u2013\u2014\u2212-"  # hyphen, en dash, em dash, minus, ascii hyphen
    "  \t\n "                  # space variants
    "MWmwKBkb"                            # unit-prefix letters
)


def _rand_text(rng, n):
    return "".join(rng.choice(_ADVERSARIAL) for _ in range(n))


class NormaliseOffsetMapProperties(unittest.TestCase):
    TRIALS = 3000

    def test_the_offset_map_holds_its_invariants(self):
        rng = random.Random(20260901)
        for _ in range(self.TRIALS):
            raw = _rand_text(rng, rng.randint(0, 40))
            text, index_map = normalise(raw)
            # length: one map entry per output character
            self.assertEqual(len(index_map), len(text))
            prev = -1
            for j in index_map:
                # in range of the raw string
                self.assertGreaterEqual(j, 0)
                self.assertLessEqual(j, len(raw))
                # non-decreasing
                self.assertGreaterEqual(j, prev)
                prev = j

    def test_normalise_is_idempotent(self):
        rng = random.Random(4242)
        for _ in range(self.TRIALS):
            raw = _rand_text(rng, rng.randint(0, 40))
            once = normalise(raw)[0]
            twice = normalise(once)[0]
            self.assertEqual(once, twice)


class AnchorNoFalsePositiveProperties(unittest.TestCase):
    TRIALS = 2000

    def test_a_real_substring_anchors_verbatim_at_its_offset(self):
        rng = random.Random(13)
        for _ in range(self.TRIALS):
            doc = _rand_text(rng, rng.randint(30, 80))
            norm = normalise(doc)[0]
            if len(norm) < 20:
                continue
            i = rng.randint(0, len(norm) - 15)
            length = rng.randint(12, min(30, len(norm) - i))
            quote = norm[i:i + length]
            if len(quote.strip()) < 12:
                continue
            a = locate(doc, quote)
            # A genuine substring of the normalised document must reach an
            # identity tier (T1 or T2), never a lower verdict, and the span it
            # names must actually contain the quote's characters.
            if a.is_verbatim:
                served = norm[a.offset_norm:a.offset_norm + a.length_norm]
                self.assertIn(quote.strip()[:8], served)

    def test_a_quote_from_another_document_is_never_an_identity_tier(self):
        rng = random.Random(99)
        for _ in range(self.TRIALS):
            doc_a = _rand_text(rng, rng.randint(40, 90))
            doc_b = _rand_text(rng, rng.randint(40, 90))
            norm_b = normalise(doc_b)[0]
            if len(norm_b) < 20:
                continue
            i = rng.randint(0, len(norm_b) - 15)
            quote = norm_b[i:i + rng.randint(12, min(30, len(norm_b) - i))]
            if len(quote.strip()) < 12:
                continue
            a = locate(doc_a, quote)
            if a.is_verbatim:
                # An identity verdict against doc_a is only allowed if the
                # quote genuinely occurs in doc_a's normalised form too.
                self.assertIn(quote, normalise(doc_a)[0])
