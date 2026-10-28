from pathlib import Path
import tempfile
import unittest

from biomedqa.chat_import import import_chat
from biomedqa.io import fingerprint, read_jsonl, write_json, write_jsonl


class ChatImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = {"id": "source1", "source_id": "source1", "split": "train", "method": "original",
                       "question": "Did treatment A change marker C?", "context": "Treatment A reduced marker C in the treated group.",
                       "answer": "yes", "rationale": "Test fixture"}
        quote = self.source["context"]
        common = {"answer": "Marker C", "rationale": quote, "evidence_quote": quote, "review_reason": "Explicit test fixture support"}
        self.drafts = {"provenance": {"mode": "current_chat_assistant_authored", "generation_process": "Unit test only",
                                     "review_process": "Unit test same-author check"}, "cases": [{"source_id": "source1",
                        "source_record_sha256": fingerprint(self.source),
                        "cr": {**common, "claim": quote, "question": "Treatment A reduced ___ in the treated group."},
                        "bio": {**common, "explanation": quote, "question": "Which marker decreased after treatment A?"}}]}
        self.sources_path, self.drafts_path = self.root/"source.jsonl", self.root/"drafts.json"
        write_jsonl(self.sources_path, [self.source])
        write_json(self.drafts_path, self.drafts)

    def tearDown(self):
        self.tmp.cleanup()

    def test_import_preserves_attribution_and_binds_reviews(self):
        report = import_chat(self.sources_path, self.drafts_path, self.root/"out")
        self.assertEqual(report["candidates"], 2)
        self.assertEqual(report["structural_pass_count"], 2)
        self.assertFalse(report["independent_expert_review"])
        candidates = read_jsonl(self.root/"out/candidates.jsonl")
        reviews = read_jsonl(self.root/"out/reviews.jsonl")
        for candidate, review in zip(candidates, reviews):
            self.assertEqual(candidate["source_id"], "source1")
            self.assertEqual(fingerprint(candidate), review["candidate_sha256"])
            self.assertFalse(review["independent"])

    def test_import_rejects_changed_source_content(self):
        write_jsonl(self.sources_path, [dict(self.source, context="Changed source")])
        with self.assertRaises(ValueError):
            import_chat(self.sources_path, self.drafts_path, self.root/"out")

    def test_import_rejects_held_out_data(self):
        altered = dict(self.source, split="test")
        self.drafts["cases"][0]["source_record_sha256"] = fingerprint(altered)
        write_jsonl(self.sources_path, [altered])
        write_json(self.drafts_path, self.drafts)
        with self.assertRaises(ValueError):
            import_chat(self.sources_path, self.drafts_path, self.root/"out")


if __name__ == "__main__":
    unittest.main()
