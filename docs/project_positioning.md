# Research Positioning

## Project description

A biomedical data augmentation prototype for constructing instruction data for downstream knowledge distillation. Inspired by question bootstrapping, it explores self verification and explanation guided question generation while retaining source evidence and review provenance. Teacher and student choices can be updated independently of the method.

## Contribution

The contribution is the biomedical adaptation, the design of two augmentation routes and the prototype for generating, screening and exporting candidate QA examples. It does not claim priority for general knowledge distillation, self verification or question bootstrapping. Cite MetaMath and describe the biomedical changes explicitly.

## Resume statement

Proposed and prototyped a biomedical data augmentation strategy using self-verification and explanation-guided question generation for downstream LLM fine-tuning.

The statement describes the design and prototype, with fine tuning as the intended downstream use. It does not assert completed student training or a measured performance improvement.

## Evidence boundary

The [MetaMath paper](https://arxiv.org/html/2309.12284v4) reports mathematical benchmark results after augmentation and fine tuning. That evidence motivates exploring a related biomedical method. Biomedical effectiveness remains an empirical question; the project does not substitute mathematical benchmark results for biomedical validation.

## Reader extensions

Readers may generate instruction data with an authorized teacher, choose a current student and framework, and compare original-only and augmented conditions under a fixed evaluation protocol. Their student training is optional downstream work. New model choices do not change which models were used in the 2024 exploration.
