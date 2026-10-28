#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_ID:?Set the teacher model ID or local directory}"
: "${MODEL_REVISION:?Set the exact model commit hash}"
for route in cr bio; do
    python -m biomedqa.cli augment \
        --input data/processed/pilot.jsonl \
        --output "runs/pilot/$route" \
        --route "$route" --limit 100 \
        --model "$MODEL_ID" --revision "$MODEL_REVISION" --four-bit
done
