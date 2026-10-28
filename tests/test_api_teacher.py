import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from biomedqa.augment import cached_call
from biomedqa.model import CallBudgetExceeded
from biomedqa.openai_teacher import OpenAITeacher


class FakeResponse(io.StringIO):
    def __init__(self, status="completed", value=None):
        super().__init__(json.dumps({"id": "UNIT_TEST_RESPONSE", "model": "UNIT_TEST_MODEL_VERSION", "status": status,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value or {"answer": "yes"})}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5}}))
        self.headers = {"x-request-id": "UNIT_TEST_REQUEST"}


class APITeacherTests(unittest.TestCase):
    @patch.dict("os.environ", {"OPENAI_API_KEY": "unit-test-placeholder"})
    def test_api_payload_and_usage_without_a_live_request(self):
        teacher = OpenAITeacher("UNIT_TEST_MODEL", 2048, 1)
        captured = []
        def fake_open(request, timeout):
            captured.append(json.loads(request.data))
            self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
            return FakeResponse()
        with patch("biomedqa.openai_teacher.urlopen", side_effect=fake_open):
            value = teacher.generate("Return JSON", seed=42)
        self.assertEqual(captured[0]["text"]["format"]["type"], "json_object")
        self.assertFalse(captured[0]["store"])
        self.assertNotIn("seed", captured[0])
        self.assertNotIn("temperature", captured[0])
        self.assertEqual(value["input_tokens"], 10)
        self.assertEqual(value["provider_model"], "UNIT_TEST_MODEL_VERSION")
        self.assertNotIn("unit-test-placeholder", json.dumps(teacher.identity))

    @patch.dict("os.environ", {"OPENAI_API_KEY": "unit-test-placeholder"})
    def test_call_cap_prevents_an_extra_network_request(self):
        teacher = OpenAITeacher("UNIT_TEST_MODEL", 2048, 1)
        with patch("biomedqa.openai_teacher.urlopen", side_effect=lambda *a, **k: FakeResponse()) as opener:
            teacher.generate("Return JSON", seed=1)
            with self.assertRaises(CallBudgetExceeded):
                teacher.generate("Return JSON", seed=2)
        self.assertEqual(opener.call_count, 1)

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_key_fails_without_network_access(self):
        with self.assertRaises(ValueError), patch("biomedqa.openai_teacher.urlopen") as opener:
            OpenAITeacher("UNIT_TEST_MODEL", 2048, 1)
        opener.assert_not_called()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "unit-test-placeholder"})
    def test_call_budget_does_not_poison_the_resume_cache(self):
        teacher = OpenAITeacher("UNIT_TEST_MODEL", 2048, 1)
        with tempfile.TemporaryDirectory() as folder:
            with patch("biomedqa.openai_teacher.urlopen", side_effect=lambda *a, **k: FakeResponse()):
                cached_call(teacher, "first JSON", folder, 1)
                with self.assertRaises(CallBudgetExceeded):
                    cached_call(teacher, "second JSON", folder, 2)
                self.assertEqual(len(list(Path(folder).glob('*.json'))), 1)
                resumed = OpenAITeacher("UNIT_TEST_MODEL", 2048, 1)
                cached_call(resumed, "second JSON", folder, 2)
            self.assertEqual(len(list(Path(folder).glob('*.json'))), 2)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "unit-test-placeholder"})
    def test_api_teacher_runs_the_cli_route_without_gpu_or_live_requests(self):
        from biomedqa.cli import main
        from biomedqa.io import read_jsonl, write_jsonl
        quote = "Treatment A reduced marker C in the treated group."
        def fake_open(request, timeout):
            prompt = json.loads(request.data)["input"]
            if prompt.startswith("Explain a biomedical relationship"):
                value = {"explanation": quote, "evidence_quote": quote}
            elif prompt.startswith("Create a new specific biomedical question"):
                value = {"question": "Which marker decreased after treatment A?"}
            else:
                value = {"answer": "Marker C", "rationale": quote, "evidence_quote": quote}
            return FakeResponse(value=value)
        with tempfile.TemporaryDirectory() as folder:
            source, out = Path(folder)/"input.jsonl", Path(folder)/"out"
            write_jsonl(source, [{"id": "source1", "source_id": "source1", "split": "train", "method": "original",
                                "question": "Did treatment A affect marker C?", "answer": "yes", "context": quote, "rationale": quote}])
            with patch("biomedqa.openai_teacher.urlopen", side_effect=fake_open) as opener:
                main(["augment", "--backend", "openai", "--model", "UNIT_TEST_MODEL", "--input", str(source),
                      "--output", str(out), "--route", "bio", "--limit", "1", "--max-api-calls", "3"])
            self.assertEqual(opener.call_count, 3)
            result = read_jsonl(out/"candidates.jsonl")[0]
            self.assertEqual(result["quality_errors"], [])
            self.assertNotIn("unit-test-placeholder", "".join(p.read_text() for p in out.rglob('*.json')))


if __name__ == "__main__":
    unittest.main()
