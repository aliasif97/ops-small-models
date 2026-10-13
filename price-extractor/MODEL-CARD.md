# NLS price-line extractor (LoRA on Qwen2.5-1.5B-Instruct-4bit)

## Purpose
Read the text of a supplier quotation (PDF, xlsx or email body) and return a JSON list of priced lines: item, spec, qty, unit, unit_price, ext_price, ccy, basis, quote_ref, doc_date, valid_until. Runs locally on a Mac, no data leaves the machine.

## Data
- Source: the company price catalogue (catalog.jsonl, see PRICEBOOK_CATALOG) (2,356 records). Kept direction=received, confidence >= 0.8, origin=email: 1,293 records over 149 documents. The 236 "pricebook" records are internal cost-rate notes (md files), not supplier documents, so they were dropped.
- Text: pdftotext -layout for PDFs, openpyxl for xlsx, raw text for email bodies. Scans with no text layer skipped (3). Input capped at 2,000 tokens, keeping the first 12 lines plus the most number-dense lines in original order. Records whose unit price does not appear in the kept text were dropped (304), so the model is never asked to guess a number it cannot see.
- Survivors: 135 documents, 732 records (train 103 docs / 535 records, validation 12 / 49, held-out test 20 / 148). The 3,000-token cap in the brief was tightened to 2,000 so input plus JSON output fits in the 3,072 training window.
- Split: by document, seed 1234, about 85/15. No document is in both train and test.
- Date range of doc_date: one 2018 outlier, otherwise 2026 (mostly June to October 2026).

## Label caveat
Labels are not human-checked. They are the catalogue produced by a cloud model's extraction (the pipeline's task files), which the business later used. The adapter learns to imitate that cloud model, including its mistakes and its habit of leaving fields blank. Scores below measure agreement with the catalogue, not agreement with the truth. Small supplier mix: a few suppliers contribute many lines.

## Method
mlx_lm 0.31.3 LoRA, chat format (short system instruction, document text as user, JSON as assistant), loss on the assistant tokens only. Rank 8, scale 20, dropout 0.05, 16 layers, learning rate 5e-5, batch 1, 400 iterations (about 4 epochs), max sequence 3,072, gradient checkpointing, seed 1234. Training time 28 min 22 s, peak memory 5.2 GB. Adapter size 20 MB (adapters/adapters.safetensors). Config: lora_config.yaml, log: train.log.

Note: validation loss was lowest at iteration 100 (0.299), then rose to 0.340 at iteration 400 while train loss fell to 0.05. The model overfits after about 100 iterations. The final adapter was kept as pre-planned; intermediate checkpoints were not retained, so a 100 to 150 iteration retrain is the first thing to try.

## Results (20 held-out documents, 148 lines, full tables in eval/RESULTS.md)
| Metric | Base zero-shot | LoRA adapter |
|---|---|---|
| Valid JSON rate | 75.0% | 75.0% |
| Line recall | 0.7% | 24.3% |
| Line precision | 2.5% | 76.6% |
| qty / unit_price / ccy exact match on matched lines | 100% / 100% / 100% | 100% / 100% / 100% |
| Latency mean / median per doc | 6.6 s / 3.7 s | 10.0 s / 4.2 s |

Field exact match is conditional on a line already matching (price within 1%, item similar), and only 1 base line matched, so the 100% figures are weak evidence. They say that when the model finds a line it copies qty, price and currency faithfully.

## Cost per document
Local: $0 marginal (electricity only). Cloud: not measured here; the pipeline's per-document cloud cost was not logged in the files I read.

## Failure modes
- Runaway repetition: on 5 of 20 documents (both models) the output loops until the 2,048-token cap and the JSON is invalid. This is why latency mean is above median.
- Long line-item tables: recall collapses on documents with 13 to 24 lines (e.g. 0 of 14, 0 of 13, 2 of 23). The 2,000-token input cap and the output limit make big tables the weak spot. 11 of 20 documents have at least one matched line from the adapter.
- Spec column and item text sometimes merged or copied from a long product description, which lowers item similarity.
- Blank fields: basis, quote_ref and dates are often empty in the labels, so the model learned to leave them empty; those fields were not scored.
- Only 20 test documents: every number here has wide error bars; one document can swing recall by several points.

## Limits
Text-layer documents only (no OCR for scans or images). English mostly; Arabic untested. Output is not trustworthy without a check: extract.py reports grounded_fraction (share of predicted unit prices that literally appear in the document) and sets needs_cloud_fallback when JSON is broken or less than 90% grounded. Do not use for anything sent to a client or supplier without human review. Not trained on sent (outbound) documents.

## How to retrain
Run price-extractor/run.sh (rebuilds data, trains, evaluates base and adapter, writes eval/RESULTS.md). It takes about 40 min. To change hyperparameters edit lora_config.yaml. To add data, refresh the pricebook catalogue first; run.sh reads it read-only.
