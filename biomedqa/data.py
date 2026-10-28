import collections
import random
from pathlib import Path

from .io import file_hash, fingerprint, read_json, read_jsonl, write_json, write_jsonl

LABELS = ("yes", "no", "maybe")


def normalize(pmid, row):
    question = row["QUESTION"].strip()
    contexts = row["CONTEXTS"]
    if not isinstance(contexts, list) or not contexts or not all(isinstance(x, str) for x in contexts):
        raise ValueError(f"Invalid abstract context for {pmid}")
    context = "\n".join(contexts).strip()
    label = row["final_decision"].strip().lower()
    if label not in LABELS or not question or not context:
        raise ValueError(f"Invalid question or label for {pmid}")
    return {"id": str(pmid), "source_id": str(pmid), "dataset": "PubMedQA-PQA-L", "question": question,
            "context": context, "answer": label, "rationale": row.get("LONG_ANSWER", "").strip(),
            "method": "original"}


def prepare(raw_path, ground_truth_path, output, dev_size=100, pilot_size=100, seed=42):
    output = Path(output)
    if any(output.glob("*.jsonl")) or (output / "manifest.json").exists():
        raise FileExistsError("Dataset output already contains a split; choose a new directory")
    raw, truth = read_json(raw_path), read_json(ground_truth_path)
    if not isinstance(raw, dict) or not isinstance(truth, dict) or not truth or set(truth) - set(raw):
        raise ValueError("Reference test IDs must be present in the raw dataset")
    records = {str(k): normalize(k, v) for k, v in raw.items()}
    if any(records[k]["answer"] != str(v).lower() for k, v in truth.items()):
        raise ValueError("Official test labels do not match the source dataset")
    pools = collections.defaultdict(list)
    for key in sorted(set(records) - set(truth)):
        pools[records[key]["answer"]].append(key)
    remaining = sum(map(len, pools.values()))
    if not 0 < dev_size < remaining or not 0 < pilot_size <= remaining - dev_size:
        raise ValueError("Invalid development or pilot size")
    rng = random.Random(seed)
    quotas = {label: int(len(ids) * dev_size / remaining) for label, ids in pools.items()}
    for label in sorted(pools, key=lambda x: (-(len(pools[x]) * dev_size / remaining - quotas[x]), x))[:dev_size - sum(quotas.values())]:
        quotas[label] += 1
    train_ids, dev_ids = [], []
    for label in sorted(pools):
        ids = pools[label][:]
        rng.shuffle(ids)
        dev_ids.extend(ids[:quotas[label]])
        train_ids.extend(ids[quotas[label]:])
    rng.shuffle(train_ids)
    test_ids = sorted(truth)
    split_ids = {"train": train_ids, "dev": sorted(dev_ids), "test": test_ids, "pilot": train_ids[:pilot_size]}
    # Identical evidence/questions cannot cross the test boundary even under different IDs.
    groups = {}
    for split in ("train", "dev", "test"):
        for key in split_ids[split]:
            group = fingerprint({"question": records[key]["question"].casefold(), "context": records[key]["context"].casefold()})
            if group in groups and groups[group] != split:
                raise ValueError("Duplicate question and abstract cross dataset splits")
            groups[group] = split
    for split, ids in split_ids.items():
        write_jsonl(output / f"{split}.jsonl", [dict(records[k], split="train" if split == "pilot" else split) for k in ids])
    manifest = {"dataset": "PubMedQA-PQA-L", "seed": seed, "raw_sha256": file_hash(raw_path),
                "reference_test_sha256": file_hash(ground_truth_path), "ids": split_ids,
                "counts": {k: len(v) for k, v in split_ids.items()},
                "labels": {k: dict(collections.Counter(records[i]["answer"] for i in ids)) for k, ids in split_ids.items()},
                "note": "Official test IDs preserved. Development split is a new seeded stratified split, not official cross validation."}
    write_json(output / "manifest.json", manifest)
    return manifest


def qa_prompt(row):
    return ("Use only the abstract below to answer the research question. "
            "Return one JSON object with keys answer and rationale. "
            "For a yes/no research question, answer must be yes, no, or maybe. "
            "For other questions, give a concise evidence supported answer.\n\n"
            f"Abstract:\n{row['context']}\n\nQuestion:\n{row['question']}\n\nAnswer:\n")


def sft_record(row):
    return {"id": row["id"], "source_id": row["source_id"], "method": row["method"],
            "prompt": qa_prompt(row), "response": {"answer": row["answer"], "rationale": row.get("rationale", "")}}


def build_sft(train_path, dev_path, output, candidates_path=None, reviews_path=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty SFT output directory")
    train, dev = read_jsonl(train_path), read_jsonl(dev_path)
    train_ids = {r["source_id"] for r in train}
    originals = {r["source_id"]: r for r in train}
    if any(r.get("split") != "train" or r.get("method") != "original" for r in train):
        raise ValueError("Training seeds must be original training records")
    if any(r.get("split") != "dev" for r in dev) or train_ids & {r["source_id"] for r in dev}:
        raise ValueError("Invalid or overlapping development data")
    rows = [sft_record(r) for r in train]
    if bool(candidates_path) != bool(reviews_path):
        raise ValueError("Augmentation requires candidate data and an explicit review file")
    if candidates_path:
        reviews = {}
        for review in read_jsonl(reviews_path):
            if review["id"] in reviews:
                raise ValueError("Duplicate review IDs")
            if review.get("decision") not in {"accept", "reject", "pending"}:
                raise ValueError("Unknown review decision")
            reviews[review["id"]] = review
        for row in read_jsonl(candidates_path):
            if row["source_id"] not in train_ids:
                raise ValueError("An augmented sample is not derived from a training source")
            if row.get("split") != "train" or row.get("method") not in {"cr", "bio"} or row.get("context") != originals[row["source_id"]]["context"]:
                raise ValueError("Augmentation provenance or abstract does not match the training source")
            review = reviews.get(row["id"], {})
            if review.get("decision") == "accept":
                if not review.get("reviewer") or not review.get("reason"):
                    raise ValueError("Accepted reviews require reviewer and rationale")
                if review.get("candidate_sha256") != fingerprint(row):
                    raise ValueError("Review does not match the current candidate content")
                if row.get("quality_errors"):
                    raise ValueError("An accepted sample failed automatic checks")
                rows.append(sft_record(row))
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate training IDs")
    write_jsonl(Path(output) / "train.jsonl", rows)
    write_jsonl(Path(output) / "dev.jsonl", [sft_record(r) for r in dev])
    manifest = {"train_count": len(rows), "dev_count": len(dev), "methods": dict(collections.Counter(r["method"] for r in rows)),
                "train_sha256": file_hash(Path(output) / "train.jsonl"), "dev_sha256": file_hash(Path(output) / "dev.jsonl")}
    write_json(Path(output) / "manifest.json", manifest)
    return manifest
