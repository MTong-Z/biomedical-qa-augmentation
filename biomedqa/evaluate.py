import collections
import re
from pathlib import Path

from .augment import cached_call
from .data import LABELS, qa_prompt
from .io import file_hash, fingerprint, read_json, read_jsonl, write_json


def predict(test_path, output, generator):
    test = read_jsonl(test_path)
    if not test or any(row.get("split") not in {"dev", "test"} for row in test):
        raise ValueError("Prediction input must be development or test records")
    if len({r["id"] for r in test}) != len(test):
        raise ValueError("Duplicate evaluation IDs")
    output = Path(output)
    signature = fingerprint({"input": file_hash(test_path), "model": generator.identity, "prompt_protocol": "abstract-question-json-v1"})
    meta = output / "manifest.json"
    if meta.exists() and read_json(meta)["signature"] != signature:
        raise ValueError("Prediction directory has another model or dataset configuration")
    write_json(meta, {"signature": signature, "model": generator.identity, "input_sha256": file_hash(test_path), "split": test[0]["split"]})
    predictions, details = {}, {}
    for index, row in enumerate(test):
        # qa_prompt only consumes the abstract and question, never the reference answer.
        seed = int(fingerprint({"seed": generator.identity.get("seed", 42), "id": row["id"]})[:8], 16)
        try:
            response, key = cached_call(generator, qa_prompt(row), output / "cache", seed)
            answer = response.get("answer")
            answer = answer.strip().casefold() if isinstance(answer, str) else "invalid"
            predictions[row["id"]] = answer if answer in LABELS else "invalid"
            details[row["id"]] = {"response": response, "cache_key": key}
        except ValueError as exc:
            predictions[row["id"]] = "invalid"
            details[row["id"]] = {"error": str(exc)}
        write_json(output / "predictions.json", predictions)
        write_json(output / "details.json", details)
        print(f"Predicted {index + 1}/{len(test)}", flush=True)
    return {"predictions": len(predictions), "invalid": sum(v == "invalid" for v in predictions.values())}


def score(test_path, predictions_path, output):
    rows = read_jsonl(test_path)
    truth = {r["id"]: r["answer"] for r in rows}
    predictions = read_json(predictions_path)
    if len(truth) != len(rows) or not truth or set(truth) != set(predictions):
        raise ValueError("Predictions must include all and only the reference evaluation IDs")
    if set(truth.values()) - set(LABELS) or set(predictions.values()) - set(LABELS) - {"invalid"}:
        raise ValueError("Unknown gold or predicted label")
    per_label = {}
    confusion = {label: dict(collections.Counter(predictions[k] for k, value in truth.items() if value == label)) for label in LABELS}
    for label in LABELS:
        tp = sum(truth[k] == label and predictions[k] == label for k in truth)
        fp = sum(truth[k] != label and predictions[k] == label for k in truth)
        fn = sum(truth[k] == label and predictions[k] != label for k in truth)
        denominator = 2 * tp + fp + fn
        per_label[label] = {"f1": 2 * tp / denominator if denominator else 0.0, "support": sum(x == label for x in truth.values())}
    result = {"n": len(truth), "accuracy": sum(truth[k] == predictions[k] for k in truth) / len(truth),
              "macro_f1": sum(x["f1"] for x in per_label.values()) / len(LABELS), "per_label": per_label,
              "invalid_count": sum(x == "invalid" for x in predictions.values()), "confusion": confusion,
              "test_sha256": file_hash(test_path), "predictions_sha256": file_hash(predictions_path),
              "note": "Invalid outputs count as incorrect. Macro F1 uses the three task labels."}
    write_json(output, result)
    return result
