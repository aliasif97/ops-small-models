# Held-out evaluation: base vs v1 (400 iters) vs v2 (best checkpoint)

Test set: 20 documents, 148 catalogue lines, split by document, never seen in training. Qwen2.5-1.5B-Instruct-4bit, greedy decoding, repetition_penalty 1.1, max 2,048 new tokens, input capped at 2,000 tokens.
v1 = final adapter after 400 iterations (adapters/). v2 = best-validation checkpoint of a 120 iteration run, iteration 100 (adapters-v2/best/, validation loss 0.299).

Match rule: item token-F1 >= 0.6 and unit_price within 1%, one-to-one. Field exact match is conditional on a line already matching.

## Main table (strict scoring: invalid JSON counts as zero lines)

| Metric | Base | v1 (400 it) | v2 (best ckpt) |
|---|---|---|---|
| Valid JSON rate | 85.0% | 70.0% | 55.0% |
| Invalid JSON documents | 3/20 | 6/20 | 9/20 |
| Predicted lines | 77 | 59 | 17 |
| Matched lines | 1/148 | 47/148 | 12/148 |
| Line recall | 0.7% | 31.8% | 8.1% |
| Line precision | 1.3% | 79.7% | 70.6% |
| qty exact match | 0.0% | 97.9% | 100.0% |
| unit_price exact match | 100.0% | 100.0% | 100.0% |
| ccy exact match | 100.0% | 72.3% | 100.0% |
| Latency mean s/doc | 8.7 | 13.0 | 13.1 |
| Latency median s/doc | 5.6 | 5.5 | 4.1 |
| Latency max s/doc | 22.9 | 31.1 | 32.3 |

## Effect of the repetition penalty alone (same weights, penalty 1.0 vs 1.1)

| Model | Invalid JSON before | Invalid JSON after | Recall before | Recall after | Precision before | Precision after |
|---|---|---|---|---|---|---|
| Base | 5/20 | 3/20 | 0.7% | 0.7% | 2.5% | 1.3% |
| v1 | 5/20 | 6/20 | 24.3% | 31.8% | 76.6% | 79.7% |

## Lenient scoring (complete lines salvaged from broken JSON), penalty 1.1

| Metric | Base | v1 | v2 |
|---|---|---|---|
| Predicted lines | 116 | 135 | 123 |
| Matched lines | 1 | 48 | 19 |
| Line recall | 0.7% | 32.4% | 12.8% |
| Line precision | 0.9% | 35.6% | 15.4% |

## Per document (predicted / catalogue / matched), penalty 1.1

| Document | Base | v1 | v2 |
|---|---|---|---|
| 19ec6687f37e5b98--SAC_QTN_1406202601.pdf | 1/1/0 | 1/1/0 | 1/1/0 |
| 19de43480896e12c--5559839026.pdf | 1/1/0 | 1/1/1 | 1/1/1 |
| 19fe1ec04779097b--QT138-Northlane Solutions Company.pdf | 16/15/0 | 16/15/13 | 0/15/0 |
| 19f7fa147be54a7c--airSF.17b - 25 Tender Offer Letter Riyadh  | 13/23/0 | 0/23/0 | 0/23/0 |
| 1a0b21b069e2617a--6-contol panlel .xlsx | 4/7/0 | 0/7/0 | 0/7/0 |
| 1a052ff10aad9ba1--Quote_EMT-ASIF ALI_59737.pdf | 0/13/0 | 13/13/10 | 0/13/0 |
| 1a085e3344981438--Neutral_Industrial_Valve_Steam_Offer.pdf | 7/7/0 | 7/7/7 | 1/7/1 |
| 19ef8e827de0b7fd--2606171ETO2.pdf | 8/24/0 | 7/24/5 | 0/24/0 |
| 1a05818221b28a04--AZR-014171.pdf | 2/2/0 | 2/2/2 | 2/2/2 |
| 19f6a9f2fc0a0eb5.txt | 0/1/0 | 0/1/0 | 0/1/0 |
| 19f3bc86b8107b8a--Q4286 (Northlane Solutions - Heliport & Me | 4/4/0 | 4/4/4 | 4/4/4 |
| 19e2b20f9832f571--Heliport Light Quotation(Northlane Solutio | 7/14/0 | 0/14/0 | 0/14/0 |
| 1a07b040250dfd26--Editorial services proposal.pdf | 1/15/0 | 0/15/0 | 0/15/0 |
| 19f383a4b716e51f.txt | 1/1/1 | 1/1/1 | 1/1/1 |
| 19ef4017c4403f01--SAC-QTN-1004.pdf | 6/1/0 | 1/1/0 | 1/1/0 |
| 19ff6d48b666194d--IN8685771.pdf | 1/1/0 | 1/1/1 | 1/1/0 |
| 19fd0ca5b782379e--vosla NARVA Quote 0020006174.PDF | 2/2/0 | 2/2/0 | 2/2/0 |
| 19efe9282047566e--TKH Airport Solutions - Quote 00004847.pdf | 0/13/0 | 0/13/0 | 0/13/0 |
| 19f990b4b3a163aa--QUOTATION 1120 17495 - Impact-Resistant Sa | 2/2/0 | 2/2/2 | 2/2/2 |
| 1a0136675face847--Quote_Belden- Asif Ali_59020.pdf | 1/1/0 | 1/1/1 | 1/1/1 |

## Validation loss by iteration (v2 run)

| Iteration | Val loss |
|---|---|
| 1 | 0.448 |
| 20 | 0.382 |
| 40 | 0.349 |
| 60 | 0.337 |
| 80 | 0.32 |
| 100 | 0.299 |
| 120 | 0.311 |

## Data counts

```
{
 "catalog_total": 2356,
 "received_conf>=0.8": 1529,
 "dropped_pricebook_origin(internal cost-rate md, not supplier docs)": 236,
 "candidate_docs": 149,
 "records_price_not_in_text_dropped": 304,
 "skip_no_visible_records": 10,
 "skip_no_text_layer": 3,
 "docs_final_pre_split": 136,
 "train_valid_docs_dropped_cannot_fit_3072": 1,
 "train_docs": 103,
 "train_records": 535,
 "valid_docs": 12,
 "valid_records": 49,
 "test_docs": 20,
 "test_records": 148,
 "docs_total_used": 135,
 "records_total_used": 732,
 "date_min": "2018-08-03",
 "date_max": "2026-10-07"
}
```
