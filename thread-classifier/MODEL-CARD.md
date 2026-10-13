# thread-classifier model card

Shipped model: **baseline-v1**, TF-IDF (word 1-2 grams + char 2-5 grams) with logistic regression, used with a 0.90 confidence gate.
Reference result that lost: **v1**, fine-tuned multilingual-e5-small.

## Purpose
Predict the `role` of a Gmail thread in the NLS RFQ tracker (customer_rfq, customer_followup,
our_rfq_to_vendor, vendor_pitch, prequal_registration, admin_noise, other) locally, so the daily
Haiku call can be skipped when the model is sure. It predicts role only. Category, company, scope,
items, deadline and the other tracker fields are not covered.

## Data
- 1,152 labelled threads from Supabase `rfq_tracker_threads` (1 thread with a null role dropped), text from `rfq_tracker_messages`. Messages span 2026-08-11 to 2026-10-13, about nine weeks.
- Labels are silver labels: Claude Haiku output, never human reviewed. The model imitates Haiku, mistakes included. Agreement with Haiku is not accuracy against the truth.
- Input per thread: direction and sender domain of the first message, subject (120 chars), first 250 chars of body excerpt or snippet. The brief asked for 1,500 chars; 250 was used because the data had to be exported through a tool channel.
- De-identified: email addresses removed (domain only, addresses in text replaced by `<email>`). Company names and subjects kept. Names and phone numbers inside body text are not scrubbed, so keep `data/` private.
- Export caveat: rows were copied through sub-agents that retyped them. The role sequence was checked against the database with a checksum and one wrong label (row 979) was fixed, so labels match the database. Text fields were not diffed; a few may contain copy slips.
- Counts: customer_rfq 196, customer_followup 56, our_rfq_to_vendor 444, vendor_pitch 97, prequal_registration 16, admin_noise 232, other 111 (`data/README.md`).
- Split: stratified 80/20 by role, seed 42. Test 231 rows, untouched until the end. Train 805, validation 116 (carved from the 80%). The baseline is fit on train+validation (921 rows).

## Method
- Baseline (shipped): TfidfVectorizer word 1-2 grams (min_df 2) and char_wb 2-5 grams (min_df 2, 200k features max), LogisticRegression C=10, balanced class weights. Saved to `models/thread-classifier-baseline-v1/` (`model.joblib`, `labels.json`).
- Reference: `intfloat/multilingual-e5-small` with a classification head, max 256 tokens, batch 8, AdamW (encoder 1e-4, head 2e-3), 5 epochs, sqrt-balanced class weights, best epoch by validation macro-F1, Apple MPS. Seed 42.
- Tuning disclosure: the encoder was trained three times. Run 1 (batch 16, lr 5e-5) underfit: test accuracy 0.827, macro-F1 0.570. Run 2 (batch 8, higher rates) gave 0.836 / 0.630. The final `./run.sh` rerun of the same settings gave 0.818 / 0.613; MPS training is not bit-for-bit repeatable even with fixed seeds. The settings were chosen on validation macro-F1, but test numbers were printed for each run. The baseline had no tuning beyond the first configuration.

## Metrics on the untouched test set (231 rows)
| model | accuracy | macro-F1 |
|---|---|---|
| TF-IDF + logistic regression (shipped) | 0.879 | 0.794 |
| multilingual-e5-small fine-tuned (final run.sh run) | 0.818 | 0.613 |

The encoder lost. With 805 training rows and Haiku label noise, n-grams beat it, mostly on the small classes.

Per-class F1 (baseline / encoder, test support): customer_rfq 0.88 / 0.86 (39); customer_followup 0.63 / 0.47 (11); our_rfq_to_vendor 0.95 / 0.91 (89); vendor_pitch 0.71 / 0.51 (20); prequal_registration 0.67 / 0.00 (3); admin_noise 0.89 / 0.91 (47); other 0.83 / 0.62 (22). With 3 prequal and 11 followup test rows those numbers are noisy.

Confusion matrix, baseline (rows = Haiku label, columns = model; order customer_rfq, customer_followup, our_rfq_to_vendor, vendor_pitch, prequal_registration, admin_noise, other):
```
customer_rfq          35  2  0  0  0  1  1
customer_followup      2  6  1  0  0  2  0
our_rfq_to_vendor      1  0 84  0  0  3  1
vendor_pitch           2  0  1 12  0  1  4
prequal_registration   0  0  0  0  2  1  0
admin_noise            1  0  1  0  1 44  0
other                  0  0  0  2  0  0 20
```
Confusion matrix, encoder (same order):
```
customer_rfq          35  2  2  0  0  0  0
customer_followup      3  4  4  0  0  0  0
our_rfq_to_vendor      1  0 83  1  0  2  2
vendor_pitch           2  0  2 10  0  0  6
prequal_registration   0  0  0  0  0  2  1
admin_noise            1  0  2  1  0 43  0
other                  0  0  1  7  0  0 14
```

## Confidence gate (shipped: 0.90)
Baseline probability >= 0.90 prints `accept`, otherwise `fallback` (send to Haiku). On the test set (`eval/baseline_threshold_curve.json`):

| threshold | coverage | agreement with Haiku on accepted |
|---|---|---|
| none | 100% | 87.9% |
| 0.6 | 79% | 94.0% |
| 0.8 | 62% | 97.2% |
| 0.90 | 48% | 99.1% |

Agreement is with the Haiku label, not truth. The 0.90 gate was picked from this test curve, so the 99.1% is slightly optimistic; it has not been confirmed on fresh data. Encoder curve for reference (`eval/threshold_curve.json`): 0.95 gives 61% coverage at 98.6%.

## Latency (model loaded, single process, 100 test threads, this Mac, `eval/latency.json`)
| model | mean | p50 | p95 |
|---|---|---|---|
| baseline, CPU | 0.56 ms | 0.56 ms | 0.66 ms |
| encoder, CPU | 36.7 ms | 30.2 ms | 73.3 ms |
| encoder, MPS | 24.3 ms | 22.5 ms | 50.5 ms |

Command-line start (loading the model) is separate: about 10 to 15 ms of work for the baseline per call after Python and scikit-learn import (a second or so), and a few seconds for the encoder.

## Size on disk
Baseline 1.7 MB (`models/thread-classifier-baseline-v1/`). Encoder 465 MB (`models/thread-classifier-v1/`, float32).

## Cost
$0 per thread locally. The tracker shells out to the `claude` CLI with Haiku, billed against the subscription rather than per call. At API list price a thread of about 1,500 input and 150 output tokens is roughly $0.002 (estimate, not measured).

## Limits and failure modes
- Silver labels and only nine weeks of mail. Changes in supplier mix or wording will hurt.
- Weak classes: customer_followup (confused with customer_rfq, our_rfq_to_vendor), vendor_pitch vs other, prequal_registration (3 test rows, 16 total).
- Looks only at the first message. Haiku's digest also uses the first reply when the opener is ours.
- Outbound outreach digests and cold intros sit between vendor_pitch, other and admin_noise even for Haiku.
- Few Arabic rows; accuracy on Arabic not measured separately.
- The existing rules layer (`ruleRole`) already handles bounces and noise senders; this model sits after it.

## Agreement audit
`eval/disagreements.md` lists 30 of the test rows where the encoder (not the baseline) and Haiku differ, highest confidence first, with a blank verdict line each. Nobody has adjudicated them. It was produced by the pipeline's second model, so it does not cover the baseline's errors. Regenerating it for the baseline is a small change to `scripts/04_audit.py`.

## Use
Default (baseline) with gate:
`.venv/bin/python thread-classifier/classify.py '{"direction":"i","sender_domain":"example-supplier.com","subject":"Urgent RFQ ...","text":"..."}'`
Encoder: add `--model finetuned`. Input can also be `-` (one JSON line on stdin) or plain text. Output keys: role, confidence, top3, gate, latency_ms, model.

## Retrain
`./run.sh` (or `make`) rebuilds the dataset from `data/parts/*.txt`, fits and saves the baseline, trains the encoder (about 6 to 10 minutes on this Mac), writes metrics, the disagreement list and the latency file. A full run completed with exit 0. Best next step: have humans judge `eval/disagreements.md` and relabel prequal_registration, then retrain.

## Plugging into the tracker
See `INTEGRATION-NOTE.md`.
