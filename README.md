# Biomedical QA Augmentation for Knowledge Distillation

A research prototype for constructing biomedical instruction data through self verification and explanation guided question generation. Existing questions, reference answers and abstracts are transformed into candidate QA pairs, reviewed against source evidence, and exported for downstream language model fine tuning.

The contribution is the biomedical augmentation strategy and its implementation. Teacher and student choices are separate from the method. Student training and biomedical benchmark comparisons are optional reader experiments. No improvement in biomedical knowledge or reasoning is claimed.

## Method

| Route | Steps | Purpose |
| --- | --- | --- |
| CR | Propose a supported claim and source quote → construct a fill in the blank question → generate an answer | Explore another view of a biomedical relationship through a self verification inspired transformation |
| BIO | Generate an explanation bounded by the abstract → construct a specific question → generate an answer | Make supporting biomedical information explicit in additional instruction examples |

Both routes retain the original abstract and source ID. Intermediate outputs, teacher identity and token usage are recorded. Structural checks flag incomplete JSON, unsupported quote spans, underspecified questions, repeated text and duplicate questions. Explicit reviews distinguish accepted, rejected and pending examples before export.

A model's self check and a matching quote are screening signals, not independent proof of biomedical correctness. Review the complete question and answer against the evidence, including its conditions and uncertainty.

## Relationship to MetaMath

[MetaMath](https://arxiv.org/html/2309.12284v4) motivates this work through mathematical question bootstrapping and backward reasoning. Its experiments report gains on GSM8K and MATH after fine tuning on generated mathematical data. Those results support investigating a biomedical adaptation, but do not establish its effectiveness in biomedicine. This project adapts the idea to biomedical source material and adds explanation guided augmentation and evidence traceability. It does not claim to originate general question bootstrapping or knowledge distillation.

The original exploration dates to 2024. The current structured implementation was prepared in October 2026; see [research provenance](docs/provenance.md).

## Setup

Python 3.10 or newer is required. Chat-record validation, data preparation, review exports and protocol tests use the standard library. These operations require neither a GPU nor an API key.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m unittest discover -s tests -v
python -m biomedqa.cli --help
```

## Prepare source data

Obtain `ori_pqal.json` and `test_ground_truth.json` from the [PubMedQA repository](https://github.com/pubmedqa/pubmedqa), and place them in `data/raw/`.

```bash
python -m biomedqa.cli prepare \
  --raw data/raw/ori_pqal.json \
  --test-labels data/raw/test_ground_truth.json \
  --output data/processed \
  --dev-size 100 --pilot-size 100 --seed 42
```

The command reserves the official 500 test IDs, divides the other records into 400 training and 100 development examples, and selects 100 training questions as pilot seeds. This development split is specific to this project. Generation accepts training sources only, and downstream reference answers never enter evaluation prompts. Raw data and run outputs are excluded from Git.

## Use the included current chat demonstration

The [included demonstration](examples/chat_demo/README.md) contains twenty curated QA examples from ten training-source questions: one CR and one BIO example per source. They were composed by the assistant in the project conversation, with stage text, exact source excerpts and same-assistant source-consistency reviews. No external teacher API or local teacher model was used.

```bash
python -m biomedqa.cli import-chat \
  --sources data/processed/train.jsonl \
  --drafts examples/chat_demo/teacher_drafts.json \
  --output runs/chat_demo/replay
```

This reconstructs and validates the recorded examples; it does not generate new text or call a model. Full abstracts are hydrated from the original local dataset and remain outside Git. The public records identify the generation and review process, source hashes and lack of an exposed exact model snapshot. These curated examples are a method demonstration, not an independent quality benchmark or evidence of student learning.

Use the resulting `candidates.jsonl` and `reviews.jsonl` with `build-sft` to export training records. See the example README for the complete command. The local demonstration produced 420 training records: 400 original, ten CR and ten BIO, with the 100-example development split unchanged.

## Optional API teacher

An API is not needed to inspect or validate the included examples. This optional backend lets readers generate additional candidates with their own authorized teacher and budget.

Set `OPENAI_API_KEY` securely in the environment. Choose an API model supporting Responses API JSON mode and record an exact snapshot when available. The standard OpenAI endpoint is used, response IDs and returned model identifiers are saved, and requests set `store=false`. No API model or paid request is selected automatically.

```bash
python -m biomedqa.cli augment \
  --backend openai --model YOUR_API_MODEL_SNAPSHOT \
  --input data/processed/pilot.jsonl \
  --output runs/demo/cr --route cr --limit 10 \
  --max-api-calls 30 --max-new-tokens 2048

python -m biomedqa.cli augment \
  --backend openai --model YOUR_API_MODEL_SNAPSHOT \
  --input data/processed/pilot.jsonl \
  --output runs/demo/bio --route bio --limit 10 \
  --max-api-calls 30 --max-new-tokens 2048
```

Each route makes up to three calls per seed before retries. The call cap is per invocation, not a dollar or account limit. Set token and spending limits appropriate to the model. Output budgets include reasoning tokens where applicable. Input prompts are sent without local tokenization or truncation. JSON mode constrains syntax; the requested fields and evidence are checked separately.

Successful calls are cached. Failed or ambiguous requests are not retried automatically; `--retry-errors` explicitly enables retrying failed calls. API seed control is unavailable: the local seed is recorded but not sent to the provider. Preserve snapshots, request logs and cached responses for reproducibility.

The optional Hugging Face teacher uses `--backend hf`, `--model` and an exact `--revision`. Install `.[gpu]` and use a compatible allocated GPU. The current reference backend uses pinned text model APIs; newer architectures may need adaptation. See `scripts/run_pilot.sh` for that path.

## Review and export instruction data

```bash
python -m biomedqa.cli review-pack \
  --candidates runs/demo/bio/candidates.jsonl \
  --output runs/demo/bio/reviews.jsonl
```

Compare each candidate with its abstract. Set `decision` to `accept` or `reject`, and supply `reviewer` and `reason`. A reviewer can be a human or an explicitly identified model assisted procedure; report which. Pending examples are excluded. Reviews are bound to candidate content hashes and cannot bypass failed structural checks. Repeat for CR.

```bash
python -m biomedqa.cli build-sft \
  --train data/processed/train.jsonl --dev data/processed/dev.jsonl \
  --candidates runs/demo/bio/candidates.jsonl \
  --reviews runs/demo/bio/reviews.jsonl \
  --output data/export/seed_plus_bio
```

Exports contain source IDs, method labels, evidence based prompts and target responses. Readers can convert them to their preferred training format. A particular student model or the bundled reference trainer is not required.

## Scope and evidence

- Implemented: both routes, attributed chat-draft import, optional API and local teacher interfaces, caching, structural screening, review records and instruction data export.
- Demonstrated: twenty current-chat examples from ten source groups, quote and provenance checks, same-assistant reviews, and training-data export. Three intentionally constructed negative controls illustrate structural and semantic failure cases; they are not observed model failures.
- Locally checked: software tests, real source preparation and export of original plus demonstrated examples. See [local validation](docs/local_validation.md).
- Optional reader work: student training and biomedical effectiveness comparisons, described in [the downstream guide](docs/downstream.md).

The release does not provide trained student weights or claim improved biomedical scores. Same-assistant reviews are not independent expert validation, and the curated examples do not estimate an automatic generation success rate. Old template-transformed records are not counted as successful generations. Historical 2024 notes remain distinct from the 2026 implementation and demonstration.

## Documentation

- [Method and scope](docs/protocol.md)
- [Project positioning and resume statement](docs/project_positioning.md)
- [Research provenance](docs/provenance.md)
- [Optional GPU execution](docs/cluster.md)

## References and release terms

- [MetaMath paper](https://arxiv.org/abs/2309.12284) and [reference code](https://github.com/meta-math/MetaMath).
- [PubMedQA data and evaluation](https://github.com/pubmedqa/pubmedqa).
- [OpenAI text generation](https://developers.openai.com/api/docs/guides/text) and [JSON output guidance](https://developers.openai.com/api/docs/guides/structured-outputs).
- [Transformers](https://huggingface.co/docs/transformers/v4.57.1/) and [PEFT](https://huggingface.co/docs/peft/v0.17.0/).

A project code license has not yet been selected. Dataset, API output and model use remain subject to their applicable terms. The prototype is not a clinically validated system.
