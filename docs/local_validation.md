# Local Validation

Validation date: 2026-10-27.

## Completed checks

The standard library test suite passes 23 tests covering official test ID preservation, deterministic disjoint splits, training-only augmentation, reference-free evaluation prompts, strict JSON parsing, three-stage routing, cache reuse and invalidation, explicit retry behavior, preservation of truncated responses, evidence quote checks, review approval and content binding, strict prediction ID matching, accuracy and macro F1, prompt masking during training data encoding, mocked API formatting and budgets, and attributed chat import with source-hash and held-out-data checks.

Model interactions use deterministic fixtures in temporary directories and mocked HTTP responses. No paid API requests are made by the tests. Their outputs are software checks, not research generation or performance results.

The command line package builds as a wheel, includes its prompt template resource, and installs in a separate local environment without GPU dependencies.

## Prepared source data

The real PQA-L source files contain 1,000 records. All 500 official reference test IDs are present and their labels match the source records.

| Split | Total | Yes | No | Maybe |
| --- | --- | --- | --- | --- |
| Training | 400 | 221 | 135 | 44 |
| Development | 100 | 55 | 34 | 11 |
| Official test | 500 | 276 | 169 | 55 |
| Pilot subset of training | 100 | 50 | 37 | 13 |

The pilot is part of the training split, not an additional independent dataset. Original-only supervised inputs contain 400 training and 100 development examples. Complete datasets and operational manifests remain excluded from the source repository.

## Method demonstration

Ten training questions were selected from the existing pilot: four yes, four no and two maybe labels. The current conversation assistant composed one CR and one BIO example for each source, retaining stage text and exact evidence excerpts. All twenty curated examples passed structural checks and received explicit same-assistant source-consistency reviews. These counts are not an independent medical accuracy or automatic generation acceptance benchmark.

The recorded examples were imported without a model call, matched to their original training-source hashes, checked for quote spans and exported alongside the original data. The resulting export contains 420 training records: 400 original, ten CR and ten BIO; the development export remains at 100.

Three intentionally constructed negative controls illustrate an underspecified question, an invented quote and an unsupported causal generalization. The first two are detected by structural checks; the third can pass those checks and needs substantive review. These are teaching controls, not observed failed generations.

No external model API requests or student training were performed. The exact chat model snapshot and per-stage token counts were not exposed in the saved record. See [the demonstration](../examples/chat_demo/README.md).

## Optional GPU validation

No local teacher has been loaded and no LoRA training, GPU memory benchmark or downstream comparison has been run. The Transformers and PEFT paths follow the pinned APIs but still require validation by anyone using these optional GPU paths. The chat demonstration does not establish that those deployment paths have been exercised.
