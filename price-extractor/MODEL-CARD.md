# NLS price-line extractor (LoRA on Qwen2.5-1.5B-Instruct-4bit)

## Purpose
Read the text of a supplier quotation (PDF, xlsx or email body) and return a JSON list of priced lines: item, spec, qty, unit, unit_price, ext_price, ccy, basis, quote_ref, doc_date, valid_until. Runs locally on a Mac, no data leaves the machine.

## Data
- Source: the company price catalogue (catalog.jsonl, PRICEBOOK_CATALOG) (2,356 records). Kept direction=received, confidence >= 0.8, origin=email: 1,293 records over 149 documents. The 236 "pricebook" records are internal cost-rate notes (md files), not supplier documents, so they were dropped.
- Text: pdftotext -layout for PDFs, openpyxl for xlsx, raw text for email bodies. Scans with no text layer skipped (3). Input capped at 2,000 tokens, keeping the first 12 lines plus the most number-dense lines in original order. Records whose unit price does not appear in the kept text were dropped (304), so the model is never asked to guess a number it cannot see.
- Survivors: 135 documents, 732 records (train 103 docs / 535 records, validation 12 / 49, held-out test 20 / 148). The 3,000-token cap in the brief was tightened to 2,000 so input plus JSON output fits in the 3,072 training window.
- Split: by document, seed 1234, about 85/15. No document is in both train and test.
- Date range of doc_date: one 2018 outlier, otherwise 2026 (mostly June to October 2026).

## Label caveat
Labels are not human-checked. They are the catalogue produced by a cloud model's extraction (the pipeline's task files), which the business later used. The adapter learns to imitate that cloud model, including its mistakes and its habit of leaving fields blank. Scores below measure agreement with the catalogue, not agreement with the truth. Small supplier mix: a few suppliers contribute many lines.

## Method
mlx_lm 0.31.3 LoRA, chat format (short system instruction, document text as user, JSON as assistant), loss on the assistant tokens only. Rank 8, scale 20, dropout 0.05, 16 layers, learning rate 5e-5, batch 1, 400 iterations (about 4 epochs), max sequence 3,072, gradient checkpointing, seed 1234. Training time 28 min 22 s, peak memory 5.2 GB. Adapter size 20 MB (shipped = this v1) (adapters/adapters.safetensors). Config: lora_config.yaml, log: train.log.

### v2 retrain (tested, not shipped)
Validation loss was lowest at iteration 100 (0.299) in the first run and rose to 0.340 by iteration 400, so a second adapter was trained: same data, seed, rank and learning rate, 120 iterations, checkpoint every 20 (adapters-v2/, config lora_config_v2.yaml, log train_v2.log, 11 min 13 s). Validation loss by iteration: 0.448 (untrained), 0.382, 0.349, 0.337, 0.320, 0.299 (iteration 100, selected), 0.311. Iteration 100 was evaluated as v2 (adapters-v2/best/).

**Decision: v1 stays the shipped adapter (adapters/).** v2 has the lower validation loss but is clearly worse at the real task: 8.1% line recall against 31.8% for v1, and 9 of 20 documents with invalid JSON against 6. Validation loss measures token-by-token imitation of the labels, not whether whole tables get extracted; 100 iterations (about one epoch) is too little to learn long tables. The overfitting worry from the first run was real for loss but did not hurt extraction. To improve, try 200 to 300 iterations with checkpoints, and choose by held-out extraction recall on a validation split, not by loss.

## Results (20 held-out documents, 148 lines; full tables in eval/RESULTS.md)
Greedy decoding, repetition_penalty 1.1, max 2,048 tokens.

| Metric | Base zero-shot | v1 (shipped, 400 it) | v2 (best ckpt, iter 100) |
|---|---|---|---|
| Valid JSON rate | 85.0% | 70.0% | 55.0% |
| Line recall | 0.7% | 31.8% | 8.1% |
| Line precision | 1.3% | 79.7% | 70.6% |
| qty exact match (matched lines) | 0.0% (1 line) | 97.9% | 100.0% |
| unit_price exact match (matched lines) | 100% | 100% | 100% |
| ccy exact match (matched lines) | 100% | 72.3% | 100% |
| Latency mean / median per doc | 8.7 s / 5.6 s | 13.0 s / 5.5 s | 13.1 s / 4.1 s |

Repetition penalty alone (1.0 to 1.1, same weights): invalid JSON base 5 to 3 of 20, v1 5 to 6 of 20. For v1 it lifted recall from 24.3% to 31.8% and precision from 76.6% to 79.7%, but did not reduce invalid JSON. It is used in extract.py.

Field exact match is conditional on a line already matching. The base matched only 1 line, so its field figures mean nothing. v1 ccy at 72.3% means it sometimes writes the wrong currency on a found line (for example EUR for a USD quote).

## Cost per document
Local: $0 marginal (electricity only). Cloud: not measured here; the pipeline's per-document cloud cost was not logged in the files I read.

## Failure modes
- Runaway repetition: on 6 of 20 documents v1 still loops until the 2,048-token cap and the JSON is invalid, even with the repetition penalty. This is why mean latency is above median. extract.py flags these for cloud fallback.
- Long line-item tables: recall collapses on documents with 13 to 24 lines (e.g. 0 of 14, 0 of 13, 2 of 23). The 2,000-token input cap and the output limit make big tables the weak spot. See eval/RESULTS.md for the per-document table.
- Spec column and item text sometimes merged or copied from a long product description, which lowers item similarity.
- Blank fields: basis, quote_ref and dates are often empty in the labels, so the model learned to leave them empty; those fields were not scored.
- Only 20 test documents: every number here has wide error bars; one document can swing recall by several points.

## Limits
Text-layer documents only (no OCR for scans or images). English mostly; Arabic untested. Output is not trustworthy without a check: extract.py reports grounded_fraction (share of predicted unit prices that literally appear in the document) and sets needs_cloud_fallback when JSON is broken or less than 90% grounded. Do not use for anything sent to a client or supplier without human review. Not trained on sent (outbound) documents.

## How to retrain
Run price-extractor/run.sh (rebuilds data, trains v2, picks the best checkpoint, evaluates base, v1 and v2, writes eval/RESULTS.md). About 25 min; `SKIP_TRAIN=1 ./run.sh` skips training and takes about 14 min. v1 is retrained with `../.venv-mlx/bin/python -m mlx_lm lora -c lora_config.yaml` (28 min). To change hyperparameters edit lora_config.yaml. To add data, refresh the pricebook catalogue first; run.sh reads it read-only.
