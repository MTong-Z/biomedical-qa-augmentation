"""Import explicitly attributed chat-authored drafts; never calls a model."""

import collections
from pathlib import Path

from .augment import quality_errors, quoted_in_context, require_text
from .io import file_hash, fingerprint, read_json, read_jsonl, write_json, write_jsonl


def import_chat(sources_path, drafts_path, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory for the chat import")
    sources = read_jsonl(sources_path)
    indexed = {row["source_id"]: row for row in sources}
    if len(indexed) != len(sources):
        raise ValueError("Source records must have unique source IDs")
    drafts = read_json(drafts_path)
    provenance = drafts.get("provenance", {})
    if provenance.get("mode") != "current_chat_assistant_authored":
        raise ValueError("Chat drafts must explicitly identify their generation mode")
    if not provenance.get("generation_process") or not provenance.get("review_process"):
        raise ValueError("Chat generation and review provenance are required")
    cases = drafts.get("cases", [])
    if not cases or len({x["source_id"] for x in cases}) != len(cases):
        raise ValueError("Chat cases must be nonempty with unique source IDs")
    candidates, reviews, seen_questions = [], [], set()
    for case in cases:
        original = indexed.get(case["source_id"])
        if not original or original.get("split") != "train" or original.get("method") != "original":
            raise ValueError("Chat augmentation can use only original training sources")
        if case.get("source_record_sha256") != fingerprint(original):
            raise ValueError("The chat draft source hash does not match the supplied record")
        for route in ("cr", "bio"):
            draft = case[route]
            first_field = "claim" if route == "cr" else "explanation"
            first_text = require_text(draft, first_field)
            quote = require_text(draft, "evidence_quote")
            answer_quote = draft.get("answer_evidence_quote", quote)
            question = require_text(draft, "question")
            if not quoted_in_context(quote, original["context"]) or not quoted_in_context(answer_quote, original["context"]):
                raise ValueError("A chat stage quote is not present in its original abstract")
            if route == "cr" and "___" not in question:
                raise ValueError("CR chat questions must contain a blank")
            first = {first_field: first_text, "evidence_quote": quote}
            if route == "cr":
                first["supported"] = True
            answer_stage = {"answer": require_text(draft, "answer"), "rationale": require_text(draft, "rationale"), "evidence_quote": answer_quote}
            candidate = {"id": f"{case['source_id']}:{route}:chat:{fingerprint(draft)[:12]}", "source_id": case["source_id"],
                         "split": "train", "method": route, "context": original["context"], "question": question,
                         **answer_stage, "intermediate": {"verify" if route == "cr" else "explain": first,
                                                          "question": {"question": question}, "answer": answer_stage},
                         "generation_provenance": provenance, "source_record_sha256": fingerprint(original), "cache_keys": []}
            candidate["quality_errors"] = quality_errors(candidate, original)
            normalized = " ".join(question.casefold().split())
            if normalized in seen_questions:
                candidate["quality_errors"].append("duplicate_generated_question")
            seen_questions.add(normalized)
            candidates.append(candidate)
            # This imports an explicitly attributed author review, not an independent judgment.
            reason = require_text(draft, "review_reason")
            reviews.append({"id": candidate["id"], "candidate_sha256": fingerprint(candidate),
                            "decision": "reject" if candidate["quality_errors"] else "accept",
                            "reviewer": "current_conversation_assistant",
                            "review_type": "same_assistant_source_consistency_review", "independent": False,
                            "reason": reason, "automatic_errors": candidate["quality_errors"]})
    report = {"mode": provenance["mode"], "sources": len(cases), "candidates": len(candidates),
              "by_route": dict(collections.Counter(r["method"] for r in candidates)),
              "source_labels": dict(collections.Counter(indexed[c["source_id"]]["answer"] for c in cases)),
              "structural_pass_count": sum(not r["quality_errors"] for r in candidates),
              "same_assistant_accepted_count": sum(r["decision"] == "accept" for r in reviews),
              "error_counts": dict(collections.Counter(e for c in candidates for e in c["quality_errors"])),
              "independent_expert_review": False, "model_network_calls": 0, "per_stage_token_counts": None,
              "not_a_generation_acceptance_benchmark": True, "student_training_performed": False,
              "source_file_sha256": file_hash(sources_path), "drafts_file_sha256": file_hash(drafts_path),
              "note": "Curated current-chat examples, jointly composed stages and same-assistant source review. Ten source groups, not twenty independent studies. No downstream performance claim."}
    write_jsonl(output / "candidates.jsonl", candidates)
    write_jsonl(output / "reviews.jsonl", reviews)
    write_json(output / "quality_report.json", report)
    return report
