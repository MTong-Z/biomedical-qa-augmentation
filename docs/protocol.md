# Biomedical Augmentation Method

The method constructs candidate biomedical instruction data for downstream knowledge distillation and supervised fine tuning. The core deliverable ends at reviewed, traceable QA data. Student training and performance evaluation are optional reader experiments.

## Input and output

Input records contain biomedical questions, reference answers, source abstracts and source identifiers. Training reference rationales may be used by the teacher. Outputs retain source IDs, abstracts, route labels, intermediate outputs, generated questions and answers, source quotes and screening results. Reviewed exports add acceptance decisions and reviewer provenance.

## Self verification inspired route

The teacher proposes a claim and an exact evidence quote. Sources without proposed support are marked unsuccessful. The teacher constructs a concrete fill in the blank question about the claim and answers from the same abstract. This check proposes support; it is not an independent factual verifier or a causal identification procedure.

This adapts the transformation idea underlying MetaMath's backward reasoning and self verification questions. The biomedical steps and evidence checks are a domain adaptation, not an invention of the general technique.

## Explanation guided route

The teacher produces a supplementary explanation supported by the abstract, asks a specific new question about the explanatory detail, then answers using the original evidence. All stages retain source context. Explanations are deliberately bounded by supplied evidence; adding outside knowledge would require additional attributable sources and validation.

## Data boundaries

The preparation example uses 1,000 expert labeled PQA-L records. It preserves the official 500 test IDs, divides the remaining records into 400 training and 100 development examples using seed 42 and label stratification, and chooses a 100-question training pilot. Only training sources may generate derived examples; descendants retain the source split.

A first demonstration may use ten sources per route. No particular teacher or student defines the method. Record the actual model version, prompts, output budget and provider settings. The original eight-example prompt files remain historical references; current prompts do not claim to reproduce a historical eight-shot experiment.

## Screening and demonstration

Structural checks flag missing fields, incomplete JSON, unsupported quotes, near copies, underspecified questions, repeated answers and duplicate questions. They do not establish medical correctness. Review question, answer, explanation and source together, preserving uncertainty and study scope. Keep accepted and rejected cases with reasons.

A convincing prototype demonstration shows actual source-to-output examples from both routes, stage outputs, a small quality summary and failure cases. Synthetic software test fixtures must not be presented as actual model generations.

## Prior evidence

[MetaMath](https://arxiv.org/html/2309.12284v4) reports mathematical reasoning results after bootstrapping and fine tuning on MetaMathQA. Its GSM8K and MATH results motivate this biomedical adaptation. They do not show that this pipeline improves biomedical knowledge, factuality or reasoning. Those questions require downstream biomedical experiments.

## Release scope

The method release includes the augmentation design, teacher interfaces, structured prompts, caching, screening, reviews, data export and runnable instructions. The included demonstration contains twenty current-chat examples from ten training sources. Stages were composed together by the same assistant, then checked against the source; they are not independent pipeline calls or independent expert validation. The record importer verifies source hashes and quote spans without an API. Student checkpoints and improvement claims are outside the first release scope. The optional trainer and scorer support readers who wish to investigate them.
