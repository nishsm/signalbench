# Tinker research credits: routes and application draft

Checked October 14, 2026. This is a draft and source summary, not a submitted application.
No credits have been granted, claimed or spent by this repository.

## Verified public routes

[Thinking Machines' research/teaching grant announcement](https://thinkingmachines.ai/news/tinker-research-and-teaching-grants/)
was published October 29, 2025. It lists research grants starting at $5,000 and $250 per student
for academic classes through an instructor's teaching-grant application. Use the research and
teaching application links on that page. Confirm current eligibility, terms and availability
with the program; a still-published announcement does not establish individual approval.

For an independent CMU student project, the research/open-source route may be relevant, but
eligibility is not guaranteed. If this is part of an actual course, a faculty instructor could
consider a teaching application. Do not claim a course, advisor, lab appointment or sponsorship
that has not been agreed. A CMU email alone does not entitle an individual to teaching credits.
This check did not verify a current official automatic $150 signup-credit offer; inspect the
account's console/billing page for any actual account-specific grant and expiration.

## Research application text to adapt

**Applicant:** Nishanth Madthila, MS Information Systems Management student, Carnegie Mellon
University (expected graduation December 2026). Independent student research; institutional
sponsorship and a faculty advisor are not asserted.

**Project:** signalbench — reproducible specialist-model benchmarking on financial text.

**Public code:** https://github.com/nishsm/signalbench

**Question:** Can supervised LoRA adaptation of a small open model match an API classifier on
financial sentiment, and does improved text classification translate into an out-of-sample
return signal? A null financial result is an acceptable outcome and will be reported.

**Why Tinker:** Managed LoRA training would allow controlled base-versus-adapter comparisons
without depending on a student laptop's GPU memory. We can use a currently supported base
model, disclose that choice, retain identical fixed data splits and compare with that model's
untuned baseline. This is a distinct experiment from the existing bitsandbytes QLoRA backend.

**Methods:** Deduplicate FinancialPhraseBank before fixed stratified train/validation/test
splits; retrieve demonstrations only from training; use Twitter Financial News Sentiment as
an out-of-domain holdout. Stop training on validation loss and report confidence intervals,
paired prediction tests, measured usage/cost and failure cases. Any exploratory SEC event
study uses public acceptance timestamps, exchange calendars and post-publication execution.
Disclose survivorship bias, FinBERT's PhraseBank overlap and base-model pretraining contamination.

**Open deliverables:** Reproducible Python pipeline, training and evaluation configuration,
source provenance, tests, results and a license-compliant adapter/model card where permitted.
Raw data and credentials will not be published. Dataset/model licenses limit commercial use;
this proposal is noncommercial research and makes no deployable-alpha claim.

**Compute request:** Small-model supervised training, validation and held-out likelihood-based
scoring. Request the amount allowed by the program and justify it with a tokenizer-based cost
estimate for the selected supported model before submission. No runtime, token count or budget
estimate is asserted here because that model and the account's rates/credits are not yet fixed.
