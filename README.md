# ops-small-models

Two small models trained on a seven-person company's own operating data, with honest held-out
evaluation, so a cloud model call can be skipped where a local model is good enough. Code, model
cards and evaluation harness only. The training data (company email, supplier pricing) and the
trained weights stay private; `examples/make_synthetic.py` writes invented data so both pipelines
run end to end.

Built by Ali Asif for Northlane Solutions (Saudi Arabia) in October 2026, by directing Claude Code.
Companion to the applied-AI case-study pack.

## What is here

| Folder | Model | Result on the untouched test set |
|---|---|---|
| `thread-classifier/` | Email-thread role classifier (7 roles, English and Arabic). TF-IDF + logistic regression vs a fine-tuned `intfloat/multilingual-e5-small`. | The simple model won: accuracy 0.879, macro-F1 0.794 (encoder 0.818 / 0.613 on 805 training rows). Shipped with a 0.90 confidence gate: 48% of threads accepted at 99.1% agreement, the rest go to the cloud model. 0.56 ms per thread on CPU, 1.7 MB. |
| `price-extractor/` | Supplier quotation text to structured price lines. LoRA adapter on `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, trained with `mlx_lm` on a Mac. | v1 (400 iterations, shipped): line recall 31.8%, precision 79.7%, every matched unit price exact, vs 0.7% / 1.3% zero-shot. A shorter v2 (120 iterations, checkpoint chosen by validation loss) scored 8.1% recall and was rejected; loss is not recall. Valid JSON 70%, weak on quotes with 13 or more lines. 20 MB adapter, $0 per document, nothing leaves the machine. |

The lesson that matters is in the first row: with 805 silver-labelled rows, n-grams beat the
encoder, so the n-grams shipped. The second row is a partial win reported as a partial win.

## Principles

- Labels are silver (produced by a cloud model in daily use), and every card says so. Agreement
  with the labeller is not accuracy.
- The test split is held out from the first script and read once, at the end.
- A baseline runs before any fine-tune, and the winner ships, whichever it is.
- Confidence gates decide what stays local and what falls back to the cloud. The gate threshold
  is reported with its coverage and agreement, not chosen by taste.
- Latency, size on disk and cost per call are measured, not estimated, except where the card says
  "estimate".
- No agent framework. Plain Python, `transformers`, `scikit-learn`, `mlx_lm`.

## Run it on synthetic data

```bash
python3 -m venv .venv --system-site-packages && .venv/bin/pip install -U transformers scikit-learn torch datasets accelerate joblib
python3 examples/make_synthetic.py
# classifier: dataset -> baseline -> fine-tune -> audit -> latency
(cd thread-classifier && ./run.sh)
# extractor (Apple silicon, mlx): point the data builder at the synthetic catalogue
export PRICEBOOK_CATALOG=$PWD/price-extractor/data/synthetic/catalog.jsonl PRICEBOOK_ROOT=$PWD/price-extractor/data/synthetic
(cd price-extractor && ./run.sh)
```

Expect weak numbers on synthetic data; it exists to prove the pipelines run, not to score.

## Run it on your own data

- Classifier: write `thread-classifier/data/parts/part_1_<name>.txt`, one thread per line, eight
  `¦`-separated fields: row number, role, category, company, direction (`i` or `o`), sender domain,
  subject, text. Roles are listed in `thread-classifier/scripts/common.py`.
- Extractor: a `catalog.jsonl` in the schema shown in `examples/make_synthetic.py`, with each record's
  `source_path` pointing at the document it came from (PDF, xlsx or txt).

## Cards

- `thread-classifier/MODEL-CARD.md`, `thread-classifier/INTEGRATION-NOTE.md`
- `price-extractor/MODEL-CARD.md`, `price-extractor/INTEGRATION-NOTE.md`, `price-extractor/eval/RESULTS.md`

## License

MIT for the code. The cards describe private datasets that are not distributed.
