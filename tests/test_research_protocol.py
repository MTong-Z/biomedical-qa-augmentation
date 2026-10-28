import json
from pathlib import Path
import tempfile
import unittest

from biomedqa.augment import augment, cached_call, quality_errors, review_pack
from biomedqa.data import build_sft, prepare, qa_prompt
from biomedqa.evaluate import predict, score
from biomedqa.io import fingerprint, read_json, read_jsonl, write_json, write_jsonl
from biomedqa.model import parse_object
from biomedqa.train import encode_example

CONTEXT = "The trial compared treatment A with treatment B. Treatment A reduced marker C in the treated group."
QUOTE = "Treatment A reduced marker C in the treated group."


class FakeModel:
    identity = {"model": "UNIT_TEST_ONLY", "seed": 42}

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, seed):
        self.calls += 1
        if "Identify one biomedical claim" in prompt:
            value = {"claim": QUOTE, "evidence_quote": QUOTE, "supported": True}
        elif "Explain a biomedical relationship" in prompt:
            value = {"explanation": QUOTE, "evidence_quote": QUOTE}
        elif "Based on the supported claim, create" in prompt:
            value = {"question": "Treatment A reduced which marker ___ in the treated group?"}
        elif "Create a new specific biomedical question" in prompt:
            value = {"question": "Which marker decreased after treatment A?"}
        else:
            value = {"answer": "Marker C", "rationale": QUOTE, "evidence_quote": QUOTE}
        return {"text": json.dumps(value), "input_tokens": 10, "output_tokens": 20}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self):
        return {"id": "source1", "source_id": "source1", "question": "Did treatment A alter marker C?", "context": CONTEXT,
                "answer": "yes", "rationale": QUOTE, "method": "original", "split": "train"}

    def prepared(self):
        raw = {str(i): {"QUESTION": f"Research question {i}?", "CONTEXTS": [f"Abstract evidence {i}."],
                       "final_decision": ["yes", "no", "maybe"][i % 3], "LONG_ANSWER": "PRIVATE_GOLD_RATIONALE"} for i in range(30)}
        truth = {str(i): raw[str(i)]["final_decision"] for i in range(15)}
        write_json(self.root / "raw.json", raw)
        write_json(self.root / "truth.json", truth)
        prepare(self.root / "raw.json", self.root / "truth.json", self.root / "splits", dev_size=6, pilot_size=4)
        return self.root / "splits"

    def test_split_preserves_official_ids_and_pilot_is_train_only(self):
        split = self.prepared()
        sets = {name: {r["id"] for r in read_jsonl(split / f"{name}.jsonl")} for name in ("train", "dev", "test", "pilot")}
        self.assertEqual(sets["test"], {str(i) for i in range(15)})
        self.assertEqual((len(sets["train"]), len(sets["dev"])), (9, 6))
        self.assertFalse(sets["train"] & sets["dev"] or sets["train"] & sets["test"] or sets["dev"] & sets["test"])
        self.assertLessEqual(sets["pilot"], sets["train"])

    def test_split_is_reproducible_and_refuses_overwrite(self):
        split = self.prepared()
        prepare(self.root / "raw.json", self.root / "truth.json", self.root / "other", dev_size=6, pilot_size=4)
        self.assertEqual(read_json(split / "manifest.json"), read_json(self.root / "other/manifest.json"))
        with self.assertRaises(FileExistsError):
            prepare(self.root / "raw.json", self.root / "truth.json", split, dev_size=6, pilot_size=4)

    def test_evaluation_prompt_excludes_gold_fields(self):
        row = self.seed()
        row.update(answer="PRIVATE_GOLD_LABEL", rationale="PRIVATE_GOLD_RATIONALE")
        prompt = qa_prompt(row)
        self.assertNotIn("PRIVATE_GOLD", prompt)
        self.assertIn(CONTEXT, prompt)

    def test_json_parser_rejects_partial_or_multiple_objects(self):
        self.assertEqual(parse_object('```json\n{"answer":"yes"}\n```'), {"answer": "yes"})
        for value in ('{"answer":', '{"a":1}{"a":2}', '[]', '```json\n{"a":1}'):
            with self.assertRaises(ValueError):
                parse_object(value)

    def test_cache_reuses_success_and_invalidates_parameter_changes(self):
        model = FakeModel()
        for _ in range(2):
            cached_call(model, "a prompt", self.root / "cache", seed=42)
        self.assertEqual(model.calls, 1)
        cached_call(model, "a prompt", self.root / "cache", seed=43)
        self.assertEqual(model.calls, 2)

    def test_both_routes_keep_context_and_resume_without_model_calls(self):
        path = self.root / "seeds.jsonl"
        write_jsonl(path, [self.seed()])
        for route in ("cr", "bio"):
            model = FakeModel()
            result = augment(path, self.root / route, route, model)
            self.assertEqual(result["automatic_pass_count"], 1)
            self.assertEqual(model.calls, 3)
            candidate = read_jsonl(self.root / route / "candidates.jsonl")[0]
            self.assertEqual(candidate["context"], CONTEXT)
            augment(path, self.root / route, route, model)
            self.assertEqual(model.calls, 3)
            self.assertEqual(candidate, read_jsonl(self.root / route / "candidates.jsonl")[0])

    def test_augmentation_refuses_test_data(self):
        write_jsonl(self.root / "test.jsonl", [dict(self.seed(), split="test")])
        with self.assertRaises(ValueError):
            augment(self.root / "test.jsonl", self.root / "out", "bio", FakeModel())

    def test_changed_generation_config_cannot_reuse_output_directory(self):
        write_jsonl(self.root / "seeds.jsonl", [self.seed()])
        augment(self.root / "seeds.jsonl", self.root / "out", "bio", FakeModel())
        with self.assertRaises(ValueError):
            augment(self.root / "seeds.jsonl", self.root / "out", "bio", FakeModel(), seed=43)

    def test_prediction_never_sends_reference_to_generator(self):
        class PredictionModel:
            identity = {"model": "UNIT_TEST_ONLY", "seed": 42}
            def generate(self, prompt, seed):
                assert "PRIVATE_REFERENCE" not in prompt
                return {"text": '{"answer":"YES","rationale":"test fixture"}', "input_tokens": 1, "output_tokens": 1}
        row = dict(self.seed(), answer="PRIVATE_REFERENCE_LABEL", rationale="PRIVATE_REFERENCE_RATIONALE", split="test")
        write_jsonl(self.root / "test.jsonl", [row])
        result = predict(self.root / "test.jsonl", self.root / "predictions", PredictionModel())
        self.assertEqual(result["invalid"], 0)
        self.assertEqual(read_json(self.root / "predictions/predictions.json"), {"source1": "yes"})

    def test_failed_model_response_is_not_silently_retried(self):
        class IncompleteModel:
            identity = {"model": "UNIT_TEST_ONLY"}
            calls = 0
            def generate(self, prompt, seed):
                self.calls += 1
                return {"text": '{"answer":', "input_tokens": 1, "output_tokens": 1}
        model = IncompleteModel()
        for _ in range(2):
            with self.assertRaises(ValueError):
                cached_call(model, "prompt", self.root / "cache", 42)
        self.assertEqual(model.calls, 1)
        with self.assertRaises(ValueError):
            cached_call(model, "prompt", self.root / "cache", 42, retry_errors=True)
        self.assertEqual(model.calls, 2)
        cached = read_json(next((self.root / "cache").glob("*.json")))
        self.assertEqual(len(cached["attempts"]), 2)

    def test_truncated_generation_is_saved_but_rejected(self):
        class TruncatedModel:
            identity = {"model": "UNIT_TEST_ONLY"}
            def generate(self, prompt, seed):
                return {"text": '{"answer":"yes"}', "input_tokens": 2, "output_tokens": 3, "truncated": True}
        with self.assertRaises(ValueError):
            cached_call(TruncatedModel(), "prompt", self.root / "cache", 42)
        cached = read_json(next((self.root / "cache").glob("*.json")))
        self.assertEqual(cached["response"]["output_tokens"], 3)
        self.assertTrue(cached["error"])

    def test_unsupported_evidence_is_rejected(self):
        row = dict(self.seed(), question="Which marker was changed after treatment?", evidence_quote="This is invented evidence that is not in the abstract.")
        self.assertIn("evidence_quote_not_found", quality_errors(row, self.seed()))

    def test_reviews_require_explicit_acceptance_and_match_current_candidate(self):
        train_path, dev_path = self.root / "train.jsonl", self.root / "dev.jsonl"
        write_jsonl(train_path, [self.seed()])
        write_jsonl(dev_path, [dict(self.seed(), id="dev1", source_id="dev1", split="dev")])
        augment(train_path, self.root / "bio", "bio", FakeModel())
        candidate_path = self.root / "bio/candidates.jsonl"
        review_path = self.root / "reviews.jsonl"
        review_pack(candidate_path, review_path)
        review = read_jsonl(review_path)[0]
        self.assertEqual(review["decision"], "pending")
        build_sft(train_path, dev_path, self.root / "sft_pending", candidate_path, review_path)
        self.assertEqual(len(read_jsonl(self.root / "sft_pending/train.jsonl")), 1)
        review.update(decision="accept", reviewer="unit-test-reviewer", reason="Unit test fixture is explicitly supported.")
        write_jsonl(review_path, [review])
        build_sft(train_path, dev_path, self.root / "sft_accepted", candidate_path, review_path)
        self.assertEqual(len(read_jsonl(self.root / "sft_accepted/train.jsonl")), 2)
        altered = read_jsonl(candidate_path)[0]
        altered["answer"] = "ALTERED"
        write_jsonl(candidate_path, [altered])
        with self.assertRaises(ValueError):
            build_sft(train_path, dev_path, self.root / "sft_altered", candidate_path, review_path)

    def test_scoring_counts_invalid_outputs_as_errors_and_checks_ids(self):
        reference = [dict(self.seed(), id=str(i), answer=label, split="test") for i, label in enumerate(("yes", "no", "maybe"))]
        write_jsonl(self.root / "test.jsonl", reference)
        write_json(self.root / "pred.json", {"0": "yes", "1": "no", "2": "invalid"})
        result = score(self.root / "test.jsonl", self.root / "pred.json", self.root / "score.json")
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(result["invalid_count"], 1)
        write_json(self.root / "pred.json", {"0": "yes"})
        with self.assertRaises(ValueError):
            score(self.root / "test.jsonl", self.root / "pred.json", self.root / "score2.json")

    def test_training_masks_prompt_and_rejects_truncation(self):
        class Tokenizer:
            chat_template = None
            eos_token_id = 0
            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text]
        row = {"id": "1", "prompt": "PROMPT", "response": {"answer": "yes"}}
        encoded = encode_example(row, Tokenizer(), 100)
        self.assertEqual(encoded["labels"][:6], [-100] * 6)
        self.assertEqual(encoded["labels"][-1], 0)
        with self.assertRaises(ValueError):
            encode_example(row, Tokenizer(), 8)


if __name__ == "__main__":
    unittest.main()
