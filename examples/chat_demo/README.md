# Current Chat Demonstration

This demonstration was authored by the Codex assistant in the project conversation on 2026-10-27. No external model API, local teacher deployment or student training was used. The exact model snapshot and per-stage token counts were not exposed in the exported record.

Ten training-source records were chosen by traversing the fixed pilot and taking four yes, four no and two maybe labels. Each produced one CR and one BIO example. These are twenty curated QA examples from ten source groups, not twenty independent studies or a representative acceptance-rate benchmark.

The three logical stages were composed in the same conversation, not executed as three independent model calls. The same assistant reviewed source consistency; there was no independent expert or separate-model validation.

## Validate the recorded examples

After preparing the original PQA-L source files as described in the main README:

```bash
python -m biomedqa.cli import-chat \
  --sources data/processed/train.jsonl \
  --drafts examples/chat_demo/teacher_drafts.json \
  --output runs/chat_demo/replay
```

This validates source hashes and quote spans, reconstructs the recorded candidates and imports the explicitly attributed same-assistant reviews. It does not call a model or regenerate the text. Full abstracts are hydrated from the local source dataset and remain in ignored run directories. The public draft file contains selected evidence excerpts.

```bash
python -m biomedqa.cli build-sft \
  --train data/processed/train.jsonl --dev data/processed/dev.jsonl \
  --candidates runs/chat_demo/replay/candidates.jsonl \
  --reviews runs/chat_demo/replay/reviews.jsonl \
  --output data/export/chat_demo
```

## Checks and limits

Twenty of twenty curated examples pass the structural checks and have same-assistant source-consistency acceptance records. This count is not medical accuracy, independent validation or evidence of student learning. Read [the quality report](quality_report.json), [review records](reviews.jsonl) and [authored stages](teacher_drafts.json).

[Negative controls](negative_controls.json) are deliberately constructed counterexamples, not observed failed generations. Generic questions and invented quote spans are caught automatically. An unsupported universal causal statement can retain a valid quote and pass structural checks, so substantive review remains necessary.

## Source examples

### Source 12684740

Original question: Cue-induced behavioural activation: a novel model of alcohol craving?

Original label: yes.

**CR question:** After a 10-day training break in this rat model, the increase in alcohol consumption elicited by five-minute alcohol-associated cues could be dose-dependently blocked by ___.

**Answer:** Naltrexone.

**Source consistency review:** The named drug and dose-dependent blockade are explicit in the source; the question specifies the animal model and exposure conditions.

**BIO question:** How did five-minute and ten- or fifteen-minute alcohol-associated cue exposures differ in their effects on alcohol consumption in the rat experiment?

**Answer:** Five-minute cues increased consumption, with blood alcohol levels of 8.9 mM versus 7.5 mM in controls and a 26% increase in active-lever responding. Ten- or fifteen-minute cues did not change alcohol consumption.

**Source consistency review:** All durations, directions and numerical outcomes match the abstract; the explanation is limited to the reported behavioral comparison.

Full stage text and evidence excerpts are retained in the authored JSON.

### Source 21618245

Original question: Does surgery or radiation therapy impact survival for patients with extrapulmonary small cell cancers?

Original label: yes.

**CR question:** In the SEER analysis of extrapulmonary small cell carcinoma, the two treatment variables identified as independent survival predictors were ___ and ___.

**Answer:** Radiation and surgery.

**Source consistency review:** The answer is directly supported and the wording preserves the distinction between a survival predictor and an established causal effect.

**BIO question:** Which non-treatment variables were identified as independent survival predictors in the extrapulmonary small cell carcinoma analysis?

**Answer:** Age greater than 50 years, female sex, and regional or distant stage were listed as independent predictors.

**Source consistency review:** The additional question exposes covariates omitted from the original treatment-focused question without inventing effect directions.

Full stage text and evidence excerpts are retained in the authored JSON.

### Source 19198736

Original question: Is being small for gestational age a risk factor for retinopathy of prematurity?

Original label: no.

**CR question:** After adjusted logistic regression in the preterm-infant cohort, significant ROP risk factors included weight gain from birth to the ___ week of life and the need for ___.

**Answer:** Sixth; blood transfusions.

**Source consistency review:** The time interval and transfusion variable match the source; no causal claim or unsupported direction of weight gain is introduced.

**BIO question:** How did any-stage and severe ROP prevalences compare between SGA and AGA infants, and what statistical evidence accompanied those comparisons?

**Answer:** Any-stage ROP occurred in 24.7% of SGA and 33.2% of AGA infants, with p = 0.111. Severe ROP occurred in 6.2% and 7.5%, respectively, with p = 0.779. The reported comparisons did not establish statistically significant group differences.

**Source consistency review:** Both outcome definitions and the group ordering were checked against the source, including the two non-significant p-values.

Full stage text and evidence excerpts are retained in the authored JSON.

### Source 16097998

Original question: Is coeliac disease screening in risk groups justified?

Original label: yes.

**CR question:** In the long-term follow-up of screen-detected coeliac disease, ___% of screen-detected patients adhered to a strict or fairly strict gluten-free diet.

**Answer:** 96%.

**Source consistency review:** The percentage is attached to the correct patient group and retains the source's strict-or-fairly-strict adherence definition.

**BIO question:** Which outcomes besides dietary adherence were comparable between long-term screen-detected coeliac patients and the reported comparator groups?

**Answer:** Quality of life and gastrointestinal symptoms were similar to those of symptom-detected patients or non-coeliac controls, while bone mineral density was similar to that of the general population.

**Source consistency review:** All three outcomes and their corresponding comparison groups are stated in the quoted source.

Full stage text and evidence excerpts are retained in the authored JSON.

### Source 14978612

Original question: Does positron emission tomography change management in primary rectal cancer?

Original label: yes.

**CR question:** In the 46-patient advanced primary rectal cancer study, PET findings altered management in ___ cases, including cancellation of surgery in ___ cases.

**Answer:** Eight; six.

**Source consistency review:** Both counts are directly reported; management changes are kept distinct from the study's separate staging endpoint.

**BIO question:** What observations in the rectal cancer PET study show that a staging change and a surgical-management change are different endpoints?

**Answer:** Surgical management remained unchanged in 36 of 46 patients, but within those 36 patients PET upstaged three and downstaged five. Thus a reported staging change did not necessarily entail a change in surgery.

**Source consistency review:** The subgroup denominator and upstage/downstage counts are preserved exactly; the inference is limited to endpoint distinction in this study.

Full stage text and evidence excerpts are retained in the authored JSON.

### Source 16216859

Original question: Does a well developed collateral circulation predispose to restenosis after percutaneous coronary intervention?

Original label: no.

**CR question:** In the prospective PCI study, collateral supply was classified as good when the collateral flow index was ___.

**Answer:** At least 0.25.

**Source consistency review:** The inequality direction and threshold match the source's operational definition.

**BIO question:** How did the baseline stenosis association differ from the later restenosis findings for patients with good versus poor collateral circulation in the PCI study?

**Answer:** Good collaterals were associated with more severe baseline stenosis, 90% versus 75% with p<0.001. Later restenosis rates were reported as similar under diameter, area and volumetric definitions, and CFI had r2<0.1 with each restenosis measure.

**Source consistency review:** The source separates the significant baseline association from three non-significant restenosis comparisons; the answer retains that distinction.

Full stage text and evidence excerpts are retained in the authored JSON.

### Source 9444542

Original question: Does hippocampal atrophy on MRI predict cognitive decline?

Original label: no.

**CR question:** In the MRI hippocampal-atrophy follow-up study, cognitive function was assessed using the ___ rather than being inferred from the MRI result alone.

**Answer:** Cambridge Cognitive Examination (CAMCOG).

**Source consistency review:** The instrument is directly named in the source, and the question distinguishes the imaging finding from the measured cognitive outcome.

**BIO question:** Which three outcomes showed no significant differences between subjects with and without hippocampal atrophy in the reported follow-up series?

**Answer:** Rate of cognitive decline, mortality, and progression to dementia.

**Source consistency review:** All three outcomes are quoted directly and the scope remains the studied follow-up series.

Full stage text and evidence excerpts are retained in the authored JSON.

### Source 24450673

Original question: Delayed imaging in routine CT examinations of the abdomen and pelvis: is it worth the additional cost of radiation and time?

Original label: no.

**CR question:** In the retrospective routine abdominopelvic CT study, adding the delayed phase increased effective radiation dose by a mean of ___%.

**Answer:** 59.5%.

**Source consistency review:** The numeric value, mean statistic and dose endpoint match the abstract.

**BIO question:** Which CT examinations were excluded from the delayed-phase study, and how does that exclusion limit interpretation of its low-yield findings?

**Answer:** Examinations with indications requiring multiphasic imaging were excluded. The reported yield therefore concerns the included routine examinations and does not directly evaluate the excluded indication-specific examinations.

**Source consistency review:** The exclusion is explicit and the scope limitation follows from it without introducing a new diagnostic recommendation.

Full stage text and evidence excerpts are retained in the authored JSON.

### Source 17279467

Original question: Can bisphosphonate treatment be stopped in a growing child with skeletal fragility?

Original label: maybe.

**CR question:** After pamidronate discontinuation in the reported child, marked reductions in volumetric BMD occurred at the ___ and ___ sites.

**Answer:** Spinal trabecular and distal radial sites.

**Source consistency review:** The skeletal sites and distinction between vBMD and BMAD are preserved; the answer remains a case-specific observation.

**BIO question:** Why does the reported child's course after stopping pamidronate not establish a universal bisphosphonate stopping rule for growing children?

**Answer:** The evidence is a single-child case report with before-, during- and after-treatment bone densitometry. The observed course can motivate further investigation, but that design does not by itself establish a population-wide stopping rule.

**Source consistency review:** The inference concerns the limits of the stated study design, and no individual treatment advice is supplied.

Full stage text and evidence excerpts are retained in the authored JSON.

### Source 15381614

Original question: Cutaneous melanoma in a multiethnic population: is this a different disease?

Original label: maybe.

**CR question:** In the multiethnic melanoma case series, worse overall survival in nonwhite patients contrasted with ___ after stratifying by stage at initial diagnosis.

**Answer:** No difference in outcome between nonwhite and white patients.

**Source consistency review:** The stratification condition is retained, preventing an unadjusted survival difference from being presented as a stage-independent effect.

**BIO question:** Which four severity-related features at initial diagnosis were more frequent or more advanced among nonwhite melanoma patients in this case series?

**Answer:** Thicker tumors, ulcerated primary tumors, positive nodes, and a more advanced stage at diagnosis.

**Source consistency review:** All four features are directly reported; the wording is limited to the case series and avoids causal attribution to ethnicity.

Full stage text and evidence excerpts are retained in the authored JSON.

