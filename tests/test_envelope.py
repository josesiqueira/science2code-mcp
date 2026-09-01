"""Tests for the refusal contract.

Every rule in `envelope.py` is enforced by `validate()`, so each test here does
two things: it checks that the builders satisfy the rule, and it checks that
`validate()` actually rejects a violation. A rule that only ever sees compliant
input has not been tested, it has been assumed.

Stdlib unittest, inline fixtures, no corpus, no MCP client.
"""

from __future__ import annotations

import unittest

from science2code import envelope as env
from science2code.envelope import EnvelopeViolation, Outcome

PROV = env.provenance("norm/1.0.0", "verify/1.0.0", "difflib")

EXPECTED_VOCABULARY = {
    "VERBATIM_EXACT",
    "VERBATIM_RELAXED_EXTRACTOR_DAMAGE",
    "PASSAGE_RELOCATED_QUOTE_DIFFERS",
    "NOT_LOCATABLE",
    "SOURCE_NOT_HELD",
    "SOURCE_UNKNOWN",
    "CORPUS_UNAVAILABLE",
    "OK",
}


def every_builder_output() -> dict[str, dict]:
    """One response from every builder in the module, keyed by outcome."""
    return {
        "VERBATIM_EXACT": env.verbatim(
            Outcome.VERBATIM_EXACT,
            paper_id="paper-a",
            document_text="a passage the document really holds",
            start_pos=10,
            end_pos=45,
            score=1.0,
            scorer="difflib",
            provenance_block=PROV,
            page=3,
        ),
        "VERBATIM_RELAXED_EXTRACTOR_DAMAGE": env.verbatim(
            Outcome.VERBATIM_RELAXED_EXTRACTOR_DAMAGE,
            paper_id="paper-a",
            document_text="a hyphen-ated passage the document holds",
            start_pos=10,
            end_pos=50,
            score=1.0,
            scorer="difflib",
            provenance_block=PROV,
        ),
        "PASSAGE_RELOCATED_QUOTE_DIFFERS": env.relocated(
            paper_id="paper-a",
            document_text="what the document actually says here",
            your_text="what you thought the document said here",
            char_diff="what {+the +}document [-actually -]says here",
            start_pos=10,
            end_pos=46,
            best_score=0.81,
            scorer="difflib",
            provenance_block=PROV,
        ),
        "NOT_LOCATABLE": env.not_locatable(
            your_text="a string that is nowhere in the corpus",
            best_score=0.31,
            scorer="difflib",
            provenance_block=PROV,
            paper_id="paper-a",
        ),
        "SOURCE_NOT_HELD": env.source_not_held(
            paper_id="paper-b",
            reason="no characters are held for this document",
            provenance_block=PROV,
            your_text="a string",
        ),
        "SOURCE_UNKNOWN": env.source_unknown(
            paper_id="paper-z",
            reason="no document with this identifier is held locally",
            provenance_block=PROV,
            your_text="a string",
        ),
        "CORPUS_UNAVAILABLE": env.corpus_unavailable(
            reason="the corpus root could not be read",
            provenance_block=PROV,
            your_text="a string",
        ),
        "OK": env.ok(
            query="a phrase to locate",
            hits=[
                {
                    "outcome": "VERBATIM_EXACT",
                    "paper_id": "paper-a",
                    "document_text": "a phrase to locate",
                    "char_interval": env.char_interval(10, 28),
                    "score": 1.0,
                    "scorer": "difflib",
                    "page": 2,
                }
            ],
            searched=["paper-a", "paper-b"],
            provenance_block=PROV,
        ),
    }


class TestClosedVocabulary(unittest.TestCase):
    def test_enum_is_exactly_the_eight_names(self):
        self.assertEqual(EXPECTED_VOCABULARY, set(env.OUTCOME_NAMES))
        self.assertEqual(EXPECTED_VOCABULARY, {o.value for o in Outcome})

    def test_every_builder_emits_a_vocabulary_string(self):
        built = every_builder_output()
        self.assertEqual(EXPECTED_VOCABULARY, set(built))
        for name, response in built.items():
            self.assertEqual(name, response["outcome"])
            self.assertIn(response["outcome"], env.OUTCOME_NAMES)

    def test_validate_rejects_a_string_outside_the_vocabulary(self):
        response = every_builder_output()["NOT_LOCATABLE"]
        response["outcome"] = "VERIFIED"
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_validate_rejects_an_unknown_nested_outcome(self):
        response = every_builder_output()["OK"]
        response["hits"][0]["outcome"] = "PROBABLY_FINE"
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)


class TestBooleanIsBanned(unittest.TestCase):
    """The ladder has four outcomes and a boolean has two values."""

    def test_no_builder_emits_a_boolean_anywhere(self):
        for name, response in every_builder_output().items():
            for path, _key, value in env._walk(response):
                self.assertNotIsInstance(value, bool, "%s carried a boolean at %s" % (name, path))

    def test_validate_rejects_an_injected_boolean(self):
        response = every_builder_output()["VERBATIM_EXACT"]
        response["verified"] = True
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_validate_rejects_a_nested_boolean(self):
        response = every_builder_output()["OK"]
        response["hits"][0]["is_match"] = False
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)


class TestRuleOneNothingToCopyInARefusal(unittest.TestCase):
    """A refusal carries no field a caller can lift and paste as a quote."""

    def test_refusals_have_no_quotable_field_name(self):
        built = every_builder_output()
        refusals = [o.value for o in env.REFUSAL_OUTCOMES]
        self.assertIn("PASSAGE_RELOCATED_QUOTE_DIFFERS", refusals)
        self.assertIn("NOT_LOCATABLE", refusals)
        for name in refusals:
            for path, key, _value in env._walk(built[name]):
                self.assertNotIn(
                    key,
                    env.QUOTE_FIELD_NAMES,
                    "%s carried a quotable field name at %s" % (name, path),
                )

    def test_relocated_carries_the_three_honest_fields_instead(self):
        response = every_builder_output()["PASSAGE_RELOCATED_QUOTE_DIFFERS"]
        for field in ("document_text", "your_text", "char_diff"):
            self.assertIn(field, response)

    def test_not_locatable_carries_no_document_words_at_all(self):
        response = every_builder_output()["NOT_LOCATABLE"]
        self.assertNotIn("document_text", response)

    def test_no_envelope_at_all_uses_a_quotable_field_name(self):
        for name, response in every_builder_output().items():
            for path, key, _value in env._walk(response):
                self.assertNotIn(key, env.QUOTE_FIELD_NAMES, "%s at %s" % (name, path))

    def test_validate_rejects_each_quotable_field_name(self):
        for field in sorted(env.QUOTE_FIELD_NAMES):
            response = every_builder_output()["NOT_LOCATABLE"]
            response[field] = "something a caller would paste"
            with self.assertRaises(EnvelopeViolation, msg="field %r was allowed" % field):
                env.validate(response)

    def test_validate_rejects_a_quotable_field_nested_in_a_hit(self):
        response = every_builder_output()["OK"]
        response["hits"][0]["quote"] = "something a caller would paste"
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)


class TestRuleTwoTheWordOnlyAppearsInOutcomeNames(unittest.TestCase):
    def test_no_system_string_uses_the_word_outside_an_outcome_name(self):
        for name, response in every_builder_output().items():
            for path, text in env.iter_strings(response, system_only=True):
                self.assertTrue(
                    env._verbatim_uses_are_outcome_names(text),
                    "%s used the word in prose at %s" % (name, path),
                )

    def test_the_notice_names_the_two_outcomes_rather_than_describing_them(self):
        notice = env.INTERPRETATION_NOTICE
        self.assertIn("VERBATIM_EXACT", notice)
        self.assertIn("VERBATIM_RELAXED_EXTRACTOR_DAMAGE", notice)
        self.assertTrue(env._verbatim_uses_are_outcome_names(notice))

    def test_validate_rejects_the_word_in_prose(self):
        response = every_builder_output()["VERBATIM_EXACT"]
        response["note"] = "this is a verbatim match"
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_validate_rejects_the_word_in_a_field_name(self):
        response = every_builder_output()["VERBATIM_EXACT"]
        response["is_verbatim_ish"] = "yes"
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_the_document_own_words_are_exempt(self):
        # A real paper may contain the word. Censoring the document would be a
        # worse fault than the one the rule prevents.
        response = env.verbatim(
            Outcome.VERBATIM_EXACT,
            paper_id="paper-a",
            document_text="we reproduce the instructions verbatim below",
            start_pos=0,
            end_pos=44,
            score=1.0,
            scorer="difflib",
            provenance_block=PROV,
        )
        self.assertEqual("VERBATIM_EXACT", response["outcome"])


class TestRuleThreeNotLocatableIsExplicit(unittest.TestCase):
    def test_char_interval_is_present_and_null(self):
        response = every_builder_output()["NOT_LOCATABLE"]
        self.assertIn("char_interval", response)
        self.assertIsNone(response["char_interval"])

    def test_best_score_and_scorer_are_reported(self):
        response = every_builder_output()["NOT_LOCATABLE"]
        self.assertIn("best_score", response)
        self.assertIn("scorer", response)
        self.assertEqual(0.31, response["best_score"])
        self.assertEqual("difflib", response["scorer"])

    def test_a_best_score_of_none_is_still_reported_as_a_key(self):
        response = env.not_locatable(
            your_text="short", best_score=None, scorer="difflib", provenance_block=PROV
        )
        self.assertIn("best_score", response)
        self.assertIsNone(response["best_score"])

    def test_validate_rejects_an_omitted_char_interval(self):
        response = every_builder_output()["NOT_LOCATABLE"]
        del response["char_interval"]
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_validate_rejects_a_non_null_char_interval(self):
        response = every_builder_output()["NOT_LOCATABLE"]
        response["char_interval"] = env.char_interval(0, 10)
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_validate_rejects_a_missing_best_score(self):
        response = every_builder_output()["NOT_LOCATABLE"]
        del response["best_score"]
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)


class TestRuleFourNoAssertionVerbs(unittest.TestCase):
    def test_no_system_string_contains_a_banned_term(self):
        for name, response in every_builder_output().items():
            for path, text in env.iter_strings(response, system_only=True):
                lowered = text.lower()
                for term in env.BANNED_TERMS:
                    self.assertNotIn(term, lowered, "%s used %r at %s" % (name, term, path))

    def test_the_notice_and_ceiling_carry_no_banned_term(self):
        for text in (env.INTERPRETATION_NOTICE, env.CEILING_SENTENCE):
            for term in env.BANNED_TERMS:
                self.assertNotIn(term, text.lower())

    def test_validate_rejects_each_banned_term(self):
        for term in env.BANNED_TERMS:
            response = every_builder_output()["VERBATIM_EXACT"]
            response["note"] = "this passage %s the claim" % term
            with self.assertRaises(EnvelopeViolation, msg="term %r was allowed" % term):
                env.validate(response)

    def test_validate_rejects_a_banned_term_in_a_reason(self):
        response = every_builder_output()["NOT_LOCATABLE"]
        response["not_available"].append(env.absence("page", "nothing here proves it"))
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_document_and_caller_words_are_exempt(self):
        # These four fields hold words this server did not write.
        self.assertEqual(
            {"document_text", "your_text", "query", "char_diff"},
            set(env.CALLER_OR_DOCUMENT_FIELDS),
        )
        response = env.relocated(
            paper_id="paper-a",
            document_text="the experiment supports the hypothesis",
            your_text="the experiment proves the hypothesis",
            char_diff="the experiment [-proves-]{+supports+} the hypothesis",
            start_pos=0,
            end_pos=38,
            best_score=0.9,
            scorer="difflib",
            provenance_block=PROV,
        )
        self.assertEqual("PASSAGE_RELOCATED_QUOTE_DIFFERS", response["outcome"])


class TestConstantsOnEveryResponse(unittest.TestCase):
    def test_interpretation_notice_is_present_and_constant(self):
        for name, response in every_builder_output().items():
            self.assertEqual(env.INTERPRETATION_NOTICE, response["interpretation_notice"], name)

    def test_the_notice_states_the_ceiling(self):
        self.assertIn(env.CEILING_SENTENCE, env.INTERPRETATION_NOTICE)

    def test_provenance_names_four_components(self):
        for name, response in every_builder_output().items():
            self.assertEqual(
                {"normaliser", "verifier", "scorer", "server"},
                set(response["provenance"]),
                name,
            )

    def test_not_available_is_always_a_list(self):
        for name, response in every_builder_output().items():
            self.assertIsInstance(response["not_available"], list, name)

    def test_absences_name_a_field_and_a_reason(self):
        entry = env.absence("page", "the corpus does not expose page text")
        self.assertEqual({"field", "reason"}, set(entry))

    def test_validate_rejects_a_missing_notice(self):
        response = every_builder_output()["OK"]
        del response["interpretation_notice"]
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_validate_rejects_an_altered_notice(self):
        response = every_builder_output()["OK"]
        response["interpretation_notice"] = "trust me"
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_validate_rejects_incomplete_provenance(self):
        response = every_builder_output()["OK"]
        del response["provenance"]["normaliser"]
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)


class TestCharInterval(unittest.TestCase):
    def test_an_interval_names_its_basis(self):
        interval = env.char_interval(10, 20)
        self.assertEqual({"start_pos", "end_pos", "basis"}, set(interval))
        self.assertEqual("normalised_document", interval["basis"])

    def test_a_missing_bound_gives_none_rather_than_a_guess(self):
        self.assertIsNone(env.char_interval(None, 20))
        self.assertIsNone(env.char_interval(10, None))


class TestVerbatimBuilderIsNarrow(unittest.TestCase):
    def test_it_refuses_a_non_identity_outcome(self):
        for outcome in (Outcome.PASSAGE_RELOCATED_QUOTE_DIFFERS, Outcome.NOT_LOCATABLE, Outcome.OK):
            with self.assertRaises(EnvelopeViolation, msg=str(outcome)):
                env.verbatim(
                    outcome,
                    paper_id="paper-a",
                    document_text="words",
                    start_pos=0,
                    end_pos=5,
                    score=1.0,
                    scorer="difflib",
                    provenance_block=PROV,
                )

    def test_envelope_imports_nothing_from_the_package(self):
        # envelope.py must stay stdlib only, so the refusal contract can be
        # tested with no corpus, no extractor and no MCP client.
        import inspect

        source = inspect.getsource(env)
        self.assertNotIn("science2code.", source.replace('"""', ""))
        self.assertNotIn("import fastmcp", source)
        self.assertNotIn("from fastmcp", source)


class TestCitationMarkerCounts(unittest.TestCase):
    """The attribution trap, and the character test that reports it.

    A paper writes "eating oranges is good [12]". The whole sentence is in the
    paper, character for character, so it reaches a character-identity outcome
    and every field of the response is true, while the claim belongs to
    reference 12. Worse, a caller whose quote stops one character short of the
    marker gets a located span with nothing in it to notice, and the marker
    that owns the claim sits just outside. Both counts are reported, and they
    are reported separately, because a marker beside a span is a different fact
    from a marker inside it.
    """

    def test_a_bracketed_number_is_counted(self):
        self.assertEqual(env.count_citation_markers("oranges are good [12]"), 1)

    def test_a_bracketed_list_is_one_marker(self):
        self.assertEqual(env.count_citation_markers("as reported [3, 4]"), 1)

    def test_a_bracketed_range_is_one_marker(self):
        self.assertEqual(env.count_citation_markers("as reported [7-9]"), 1)

    def test_two_markers_in_one_sentence_are_two(self):
        self.assertEqual(env.count_citation_markers("this [1] and that [2]"), 2)

    def test_an_author_year_marker_is_counted(self):
        self.assertEqual(
            env.count_citation_markers("as shown (Smith et al., 2019)"), 1)

    def test_a_bare_parenthesised_year_is_counted(self):
        self.assertEqual(env.count_citation_markers("the study (2019) found"), 1)

    def test_prose_with_no_marker_counts_zero(self):
        self.assertEqual(
            env.count_citation_markers("a sentence with no citation at all"), 0)

    def test_an_array_index_is_not_mistaken_for_a_marker(self):
        self.assertEqual(env.count_citation_markers("the value of x[i] rises"), 0)

    def test_the_block_separates_inside_from_beside(self):
        block = env.citation_markers(
            "oranges are good", before="", after=" [12]. The next sentence")
        self.assertEqual(block["in_located_characters"], 0)
        self.assertEqual(block["in_surrounding_characters"], 1)

    def test_a_marker_inside_the_span_is_counted_inside(self):
        block = env.citation_markers("oranges are good [12]")
        self.assertEqual(block["in_located_characters"], 1)
        self.assertEqual(block["in_surrounding_characters"], 0)

    def test_the_block_names_the_window_it_looked_at(self):
        block = env.citation_markers("words")
        self.assertEqual(block["context_chars"], env.CITATION_CONTEXT_CHARS)

    def test_the_rule_is_carried_so_the_counts_cannot_be_read_as_a_judgement(self):
        self.assertEqual(env.citation_markers("words")["rule"],
                         env.CITATION_MARKER_RULE)

    def test_the_block_survives_validate_on_both_located_outcomes(self):
        block = env.citation_markers("oranges are good [12]")
        located = env.verbatim(
            Outcome.VERBATIM_EXACT,
            paper_id="paper-a",
            document_text="oranges are good [12]",
            start_pos=0,
            end_pos=21,
            score=1.0,
            scorer="difflib",
            provenance_block=PROV,
            citation_marker_block=block,
        )
        self.assertEqual(located["citation_markers"], block)
        relocated = env.relocated(
            paper_id="paper-a",
            document_text="oranges are good [12]",
            your_text="oranges are excellent",
            char_diff="char diff: - quote, + document",
            start_pos=0,
            end_pos=21,
            best_score=0.8,
            scorer="difflib",
            provenance_block=PROV,
            citation_marker_block=block,
        )
        self.assertEqual(relocated["citation_markers"], block)

    def test_the_counts_are_whole_numbers_and_never_a_boolean(self):
        # validate() bans a boolean anywhere in an envelope, so a count of one
        # marker must not be reported as "there is a marker".
        block = env.citation_markers("oranges are good [12]")
        for key in ("in_located_characters", "in_surrounding_characters"):
            with self.subTest(key=key):
                self.assertIsInstance(block[key], int)
                self.assertNotIsInstance(block[key], bool)

    def test_the_rule_names_no_assertion_verb(self):
        lowered = env.CITATION_MARKER_RULE.lower()
        for term in env.BANNED_TERMS:
            with self.subTest(term=term):
                self.assertNotIn(term, lowered)


class TestRuleFiveNoResponseNamesTheOwnerOfAClaim(unittest.TestCase):
    """The fifth anti-blur rule, and the measurement that closes the question.

    A paper's body says "oranges are good [12]". The sentence is that paper's,
    character for character, and the CLAIM is reference 12's. Measured over the
    design corpus, 13.9% of body sentences carry a marker inside the located
    span and 23.6% carry one within 64 characters, so this is roughly one
    located passage in four rather than an edge case.

    The obvious next move is a field saying whose claim it is, and that move is
    closed. Trained human annotators judging citation accuracy agree at Cohen's
    kappa 0.18 to 0.31. The one dedicated attribution study reaches
    Krippendorff's alpha .654 against a human ceiling of .806. A field
    answering the question would be a number nobody can stand behind, printed
    beside fields that are mechanically true and therefore read as one of them.

    Rule 5 makes it impossible to add by accident rather than merely
    discouraged, because a written rule that depends on being remembered is
    what failed on 2026-08-28 and caused this project to exist.
    """

    def response(self):
        return env.verbatim(
            Outcome.VERBATIM_EXACT,
            paper_id="paper-a",
            document_text="oranges are good [12]",
            start_pos=0,
            end_pos=21,
            score=1.0,
            scorer="difflib",
            provenance_block=PROV,
        )

    def test_a_field_naming_the_attributed_work_is_refused(self):
        response = self.response()
        response["attributed_to"] = "reference 12"
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_the_rule_catches_the_whole_family_by_substring(self):
        for name in (
            "attribution",
            "attributed_to",
            "claim_attribution",
            "claim_owner",
            "claimed_by",
            "cited_work",
            "whose_claim",
            "source_of_claim",
            "belongs_to",
        ):
            with self.subTest(name=name):
                response = self.response()
                response[name] = "reference 12"
                with self.assertRaises(EnvelopeViolation):
                    env.validate(response)

    def test_the_rule_reaches_a_nested_field_too(self):
        response = self.response()
        response["citation_markers"] = {"in_located_characters": 1,
                                        "attributed_to": "reference 12"}
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_the_rule_reaches_a_field_inside_a_list(self):
        response = self.response()
        response["not_available"] = [{"field": "page", "reason": "no page",
                                      "claim_owner": "reference 12"}]
        with self.assertRaises(EnvelopeViolation):
            env.validate(response)

    def test_a_marker_count_is_not_refused_by_the_rule(self):
        # The rule must bite on the answer and not on the evidence, or the
        # counts that make the trap visible would themselves be unreportable.
        response = self.response()
        response["citation_markers"] = env.citation_markers("oranges are good [12]")
        self.assertEqual(response, env.validate(response))

    def test_the_ceiling_sentence_states_the_agreement_numbers(self):
        for fragment in ("0.18 to 0.31", ".654", ".806"):
            self.assertIn(fragment, env.ATTRIBUTION_CEILING_SENTENCE, fragment)

    def test_the_ceiling_sentence_obeys_rules_two_and_four(self):
        text = env.ATTRIBUTION_CEILING_SENTENCE
        self.assertTrue(env._verbatim_uses_are_outcome_names(text))
        for term in env.BANNED_TERMS:
            self.assertNotIn(term, text.lower(), term)

    def test_the_module_docstring_records_why_the_rule_exists(self):
        doc = env.__doc__ or ""
        for fragment in ("0.18 to 0.31", ".654", ".806", "13.9%", "23.6%"):
            self.assertIn(fragment, doc, fragment)


class TestTheReferenceRegionBlockRidesOnALocatedResponse(unittest.TestCase):
    """A bibliography holds OTHER papers' titles.

    Search a title, find it in the citing paper, and a caller concludes the
    citing author wrote it. The region block reports where a located offset
    sits relative to the bibliography, and it rides on the two builders that
    name a location, because those are the two answers a caller can act on.
    """

    REGION = {
        "status": "IN_REFERENCE_REGION",
        "status_vocabulary": ["IN_REFERENCE_REGION", "OUTSIDE_REFERENCE_REGION"],
        "region_version": "region/1.0.0",
        "rule": "where the located offset falls relative to the bibliography",
        "local_block": {
            "signal": "REFERENCE_LIKE_BLOCK_WARNING",
            "signals_found": 3,
            "signals_named": ["year", "initials", "entry_marker"],
        },
    }

    def test_a_character_identity_outcome_carries_it(self):
        response = env.verbatim(
            Outcome.VERBATIM_EXACT,
            paper_id="paper-a",
            document_text="A study of oranges",
            start_pos=0,
            end_pos=18,
            score=1.0,
            scorer="difflib",
            provenance_block=PROV,
            reference_region_block=self.REGION,
        )
        self.assertEqual(self.REGION, response["reference_region"])

    def test_a_relocated_passage_carries_it_too(self):
        response = env.relocated(
            paper_id="paper-a",
            document_text="A study of oranges",
            your_text="A study on oranges",
            char_diff="one word differs",
            start_pos=0,
            end_pos=18,
            best_score=0.9,
            scorer="difflib",
            provenance_block=PROV,
            reference_region_block=self.REGION,
        )
        self.assertEqual(self.REGION, response["reference_region"])

    def test_a_builder_called_without_one_omits_the_key_rather_than_guessing(self):
        response = env.verbatim(
            Outcome.VERBATIM_EXACT,
            paper_id="paper-a",
            document_text="A study of oranges",
            start_pos=0,
            end_pos=18,
            score=1.0,
            scorer="difflib",
            provenance_block=PROV,
        )
        self.assertNotIn("reference_region", response)

    def test_the_block_is_copied_so_a_later_edit_cannot_reach_the_response(self):
        block = dict(self.REGION)
        response = env.verbatim(
            Outcome.VERBATIM_EXACT,
            paper_id="paper-a",
            document_text="A study of oranges",
            start_pos=0,
            end_pos=18,
            score=1.0,
            scorer="difflib",
            provenance_block=PROV,
            reference_region_block=block,
        )
        block["status"] = "OUTSIDE_REFERENCE_REGION"
        self.assertEqual("IN_REFERENCE_REGION", response["reference_region"]["status"])

    def test_a_status_label_is_a_string_and_never_a_flag(self):
        response = env.verbatim(
            Outcome.VERBATIM_EXACT,
            paper_id="paper-a",
            document_text="A study of oranges",
            start_pos=0,
            end_pos=18,
            score=1.0,
            scorer="difflib",
            provenance_block=PROV,
            reference_region_block=self.REGION,
        )
        # A boolean anywhere would have raised inside validate() already; this
        # states the reason, which is that "is this in the bibliography" has
        # six answers and two of them mean "the model did not decide".
        self.assertIsInstance(response["reference_region"]["status"], str)


class TestTheSentenceUsesTheExemptionThatAlreadyExists(unittest.TestCase):
    """The document's own sentence rides under `document_text`, and must.

    A real paper contains the word "demonstrates". Rules 2 and 4 police the
    server's own prose and exempt exactly four keys, and
    `test_the_exempt_key_list_has_not_grown` guards that list because widening
    it is a decision. Putting the sentence under a NEW key would have forced
    that widening, and would have made every paper containing a banned verb
    permanently unanswerable in the meantime.
    """

    def test_a_sentence_holding_a_banned_verb_survives_validate(self):
        block = {
            "in_located_characters": 1,
            "sentence": {
                "document_text": "This paper demonstrates the effect [4].",
                "char_interval": {"start_pos": 0, "end_pos": 38,
                                  "basis": "normalised_document"},
            },
        }
        response = env.verbatim(
            Outcome.VERBATIM_EXACT,
            paper_id="paper-a",
            document_text="This paper demonstrates the effect [4].",
            start_pos=0,
            end_pos=38,
            score=1.0,
            scorer="difflib",
            provenance_block=PROV,
            citation_marker_block=block,
        )
        self.assertIn("demonstrates", response["citation_markers"]["sentence"]["document_text"])

    def test_the_same_words_under_another_key_are_still_refused(self):
        block = {
            "in_located_characters": 1,
            "sentence": {"sentence_text_copy": "This paper demonstrates it [4]."},
        }
        with self.assertRaises(EnvelopeViolation):
            env.verbatim(
                Outcome.VERBATIM_EXACT,
                paper_id="paper-a",
                document_text="This paper demonstrates the effect [4].",
                start_pos=0,
                end_pos=38,
                score=1.0,
                scorer="difflib",
                provenance_block=PROV,
                citation_marker_block=block,
            )


if __name__ == "__main__":
    unittest.main()
