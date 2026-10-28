import collections
import difflib
import json
import time
from pathlib import Path

from .io import fingerprint, read_json, read_jsonl, write_json, write_jsonl
from .model import CallBudgetExceeded, parse_object


def require_text(value, key):
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing nonempty text field: {key}")
    return text.strip()


def quoted_in_context(quote, context):
    normalize = lambda text: " ".join(text.split())
    quote = normalize(quote)
    return len(quote) >= 20 and quote in normalize(context)


def quality_errors(candidate, original):
    errors = []
    question, answer = candidate.get("question", ""), candidate.get("answer", "")
    if len(question.strip()) < 12 or not answer.strip():
        errors.append("missing_or_short_question_or_answer")
    if any(x in question.casefold() for x in ("subject of the question", "answer provided", "original question", "above question")):
        errors.append("underspecified_question")
    if difflib.SequenceMatcher(None, question.casefold(), original["question"].casefold()).ratio() > 0.95:
        errors.append("near_copy_of_original_question")
    if not quoted_in_context(candidate.get("evidence_quote", ""), original["context"]):
        errors.append("evidence_quote_not_found")
    words = answer.casefold().split()
    if len(words) >= 30:
        grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
        if 1 - len(set(grams)) / len(grams) > 0.35:
            errors.append("repetitive_answer")
    return errors


def cached_call(generator, prompt, cache_dir, seed, retry_errors=False):
    request = {"model": generator.identity, "prompt": prompt, "seed": seed}
    key = fingerprint(request)
    path = Path(cache_dir) / f"{key}.json"
    attempts = []
    if path.exists():
        saved = read_json(path)
        attempts = saved.get("attempts", [])
        if not saved.get("error"):
            return parse_object(saved["response"]["text"]), key
        if not retry_errors:
            error_type = RuntimeError if saved.get("runtime_failure") else ValueError
            raise error_type(saved["error"])
    started = time.perf_counter()
    try:
        response = generator.generate(prompt, seed=seed)
        if response.get("truncated"):
            raise ValueError("Generation reached its token limit")
        if response.get("status") not in {None, "completed"} or response.get("refusal"):
            raise ValueError("Provider response did not complete successfully")
        result = parse_object(response["text"])
    except Exception as exc:
        if isinstance(exc, CallBudgetExceeded):
            raise
        attempts.append({"response": locals().get("response"), "error": str(exc), "elapsed_seconds": time.perf_counter() - started})
        write_json(path, {"key": key, "request": request, "error": str(exc), "runtime_failure": isinstance(exc, RuntimeError),
                          "response": locals().get("response"), "attempts": attempts})
        raise
    attempts.append({"response": response, "error": None, "elapsed_seconds": time.perf_counter() - started})
    write_json(path, {"key": key, "request": request, "response": response, "error": None, "attempts": attempts})
    return result, key


def augment(input_path, output_dir, route, generator, limit=100, seed=42, retry_errors=False):
    if route not in {"bio", "cr"} or limit <= 0:
        raise ValueError("Choose bio or cr and a positive limit")
    seeds = read_jsonl(input_path)[:limit]
    if not seeds or any(r.get("split") != "train" or r.get("method") != "original" for r in seeds):
        raise ValueError("Augmentation accepts only original training records")
    if len({r["id"] for r in seeds}) != len(seeds):
        raise ValueError("Duplicate input IDs")
    output_dir = Path(output_dir)
    templates = read_json(Path(__file__).with_name("prompt_templates.json"))
    signature = fingerprint({"seeds": seeds, "route": route, "templates": templates, "model": generator.identity, "seed": seed})
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["run_signature"] != signature:
        raise ValueError("Output belongs to another generation configuration; choose a new directory")
    manifest = {"run_signature": signature, "route": route, "seed_count": len(seeds), "model": generator.identity,
                "source_fingerprint": fingerprint(seeds), "template_fingerprint": fingerprint(templates), "seed": seed}
    write_json(manifest_path, manifest)
    candidates, seen_questions = [], set()
    for index, original in enumerate(seeds):
        row = {"id": f"{original['id']}:{route}:{signature[:12]}", "source_id": original["source_id"],
               "split": "train", "method": route, "context": original["context"], "question": "", "answer": "",
               "rationale": "", "evidence_quote": "", "cache_keys": [], "quality_errors": []}
        state = {"abstract": original["context"], "original_question": original["question"],
                 "original_answer": original["answer"], "reference_rationale": original.get("rationale", "")}
        steps = ("verify", "question", "answer") if route == "cr" else ("explain", "question", "answer")
        try:
            for stage, step in enumerate(steps):
                prompt = templates[f"{route}_{step}"] + "\n\nINPUT JSON\n" + json.dumps(state, ensure_ascii=False)
                stage_seed = int(fingerprint({"seed": seed, "id": original["id"], "route": route, "stage": stage})[:8], 16)
                value, key = cached_call(generator, prompt, output_dir / "cache", stage_seed, retry_errors)
                row["cache_keys"].append(key)
                if step == "verify":
                    if value.get("supported") is not True:
                        raise ValueError("Self check did not identify supported evidence")
                    require_text(value, "claim")
                if step in {"verify", "explain", "answer"}:
                    if not quoted_in_context(require_text(value, "evidence_quote"), original["context"]):
                        raise ValueError("Stage evidence quote is not in the abstract")
                if step == "explain":
                    require_text(value, "explanation")
                if step == "question":
                    question = require_text(value, "question")
                    if route == "cr" and "___" not in question:
                        raise ValueError("CR question is missing its blank")
                if step == "answer":
                    row.update(question=state["question"]["question"], answer=require_text(value, "answer"),
                               rationale=require_text(value, "rationale"), evidence_quote=value["evidence_quote"])
                state[step] = value
            row["quality_errors"] = quality_errors(row, original)
            normalized_question = " ".join(row["question"].casefold().split())
            if normalized_question in seen_questions:
                row["quality_errors"].append("duplicate_generated_question")
            seen_questions.add(normalized_question)
        except (ValueError, KeyError, TypeError) as exc:
            row["quality_errors"] = ["generation_failed"]
            row["error"] = str(exc)
        row["intermediate"] = {k: v for k, v in state.items() if k in steps}
        candidates.append(row)
        write_jsonl(output_dir / "candidates.jsonl", candidates)
        print(json.dumps({"route": route, "completed": index + 1, "total": len(seeds), "automatic_checks_passed": not row["quality_errors"]}), flush=True)
    errors = collections.Counter(error for row in candidates for error in row["quality_errors"])
    report = {"route": route, "total": len(candidates), "automatic_pass_count": sum(not row["quality_errors"] for row in candidates),
              "unique_questions": len({r["question"] for r in candidates if r["question"]}), "error_counts": dict(errors),
              "review_status": "Not yet reviewed for scientific support. Automatic checks are structural screening only."}
    attempts = [attempt for p in (output_dir / "cache").glob("*.json") for attempt in read_json(p).get("attempts", [])]
    report["model_usage"] = {"attempts": len(attempts), "input_tokens_recorded": sum((a.get("response") or {}).get("input_tokens", 0) for a in attempts),
                             "output_tokens_recorded": sum((a.get("response") or {}).get("output_tokens", 0) for a in attempts),
                             "attempts_without_token_counts": sum(a.get("response") is None or a["response"].get("token_counts_available") is False for a in attempts),
                             "note": "Includes recorded retries; cache hits do not add model tokens. Incomplete backend calls may have unavailable counts."}
    write_json(output_dir / "quality_report.json", report)
    return report


def review_pack(candidates_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing review file")
    rows = []
    for candidate in read_jsonl(candidates_path):
        rows.append({"id": candidate["id"], "candidate_sha256": fingerprint(candidate), "decision": "pending", "reviewer": "", "reason": "",
                     "question": candidate["question"], "answer": candidate["answer"], "rationale": candidate.get("rationale", ""),
                     "evidence_quote": candidate.get("evidence_quote", ""), "context": candidate["context"],
                     "quality_errors": candidate["quality_errors"]})
    write_jsonl(output, rows)
    return {"review_items": len(rows), "output": str(output)}
