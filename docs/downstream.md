# Optional Student Fine Tuning

Readers can use the exported training data with a student and training framework of their choice. Student training and benchmark comparisons are optional extensions, outside the core method and prototype release. No student performance improvement has been measured by this project.

The bundled training command is an unvalidated CUDA reference for causal models supported by the pinned Transformers 4.57.1 and PEFT versions. Newer architectures such as Qwen3.5 require their own compatible package versions and loading code; changing only a model ID is insufficient.

## Suggested comparisons

| Condition | Training data |
| --- | --- |
| Base | No project fine tuning |
| Seed only | Original training examples |
| Seed plus CR | Original examples and reviewed self verification candidates |
| Seed plus BIO | Original examples and reviewed explanation guided candidates |

Fine tuning conditions use the same base model, optimizer steps, batch settings and evaluation prompts. Their dataset token totals are recorded. Equal optimizer steps are not an exactly equal token budget; conclusions must acknowledge that limitation.

## Reference commands


Run training inside a GPU allocation. Fix the same student model and revision for every condition. A teacher and student may be the same model, but record that choice and consider its implications for self generated data.

```bash
python -m biomedqa.cli train \
  --train data/sft/seed_only/train.jsonl --dev data/sft/seed_only/dev.jsonl \
  --output runs/train/seed_only \
  --model "$MODEL_ID" --revision "$MODEL_REVISION" --four-bit --steps 100

python -m biomedqa.cli predict \
  --input data/processed/test.jsonl --output runs/eval/seed_only \
  --model "$MODEL_ID" --revision "$MODEL_REVISION" \
  --adapter runs/train/seed_only/adapter --four-bit

python -m biomedqa.cli score \
  --reference data/processed/test.jsonl \
  --predictions runs/eval/seed_only/predictions.json \
  --output runs/eval/seed_only/metrics.json
```

Repeat with the two augmented training sets. Evaluate the base condition by omitting `--adapter`. Keep base checkpoint, quantization, decoding and prompts consistent. Checkpoint selection uses development loss; do not adjust prompts or hyperparameters against the test set.

Report accuracy, macro F1 over yes/no/maybe, invalid output count, accepted candidate count and per-method quality results. For valid task labels the scoring definitions match the official metrics. This implementation treats malformed model outputs as errors and includes them in the denominator. Small differences in a 500-item test set do not establish a robust improvement; additional seeds and uncertainty estimates are needed for stronger claims.

