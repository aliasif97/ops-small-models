# Integration note: baseline model first, Haiku as fallback

Nothing in the dashboard repo was edited. This describes the change.

## What the model can and cannot replace
`classifyBatch` in `src/lib/rfq-tracker/classify.ts` asks Haiku for about 14 fields per thread (role,
division, category, company, scope, items, item_count, deadline, location, urgency, client_ref and more).
The local model predicts role only. It can replace Haiku only for threads that need no extraction: roles
`admin_noise`, `vendor_pitch`, `other`. For customer_rfq, customer_followup and our_rfq_to_vendor the extraction
fields still need Haiku, so a local role saves nothing there.

## Gate
Baseline (TF-IDF + logistic regression) confidence >= 0.90 means `accept`; otherwise `fallback` to Haiku.
On the 231 test rows the 0.90 gate accepted 48% of threads at 99.1% agreement with the Haiku label.
But the saving only applies to accepted threads whose role is admin_noise, vendor_pitch or other; accepted
RFQ-type threads still go to Haiku for extraction. Count that share in the shadow run before promising a saving.
The threshold was read off the test curve, so confirm it on fresh data. Agreement is with Haiku, not truth.

## Change (describe only)
1. New file `src/lib/rfq-tracker/local-role.ts`. It calls
   `.venv/bin/python thread-classifier/classify.py -`
   per thread (baseline is the default; about 1 s Python and scikit-learn start-up, 0.6 ms inference) or, for a
   batch of hundreds, a small `--batch` loop to avoid repeated start-up (not built yet). Input JSON per thread:
   `{"direction":"i"|"o","sender_domain":...,"subject":...,"text":...}`; map digest `first_direction` "us" to "o",
   "them" to "i"; sender_domain from `domainOf(first_from)`; subject cut to 120 chars; text = first 250 chars of
   `first_text` with email addresses replaced by `<email>`. Must match training input exactly. Output:
   `{"role","confidence","top3","gate","latency_ms","model"}`.
2. In the loop that builds digests, after `ruleRole` returns null and before `classifyBatch`:
   ```
   if (process.env.RFQ_LOCAL_ROLE === "1") {
     const r = await localRole(digest);
     if (r.gate === "accept" && ["admin_noise","vendor_pitch","other"].includes(r.role)) {
       out.push(skeletonClassification(digest.id, r.role, r.confidence)); // extraction fields null, division "na"
       continue;
     }
   }
   pending.push(digest); // goes to classifyBatch / Haiku as today
   ```
   `skeletonClassification` mirrors the `Classification` shape with null extraction fields.
3. Default off. Add `RFQ_LOCAL_ROLE_SHADOW=1`: run the local model on every thread, still call Haiku for all, and log
   `local_role`, `local_conf`, `haiku_role` to `data/rfq-tracker.log`. Run 2 to 4 weeks, compute agreement per role,
   then turn the flag on.
4. Set `classify_json.source = "local-baseline-v1"` on skipped threads so they can be reviewed or reclassified.
5. Run it on the Mac that already runs the tracker, not in a Vercel function (needs Python and scikit-learn).
6. Fallback model: `--model finetuned` selects the 465 MB encoder; it scored lower and is not recommended.

## Caveats
- Retrain after human relabelling; with more clean data the encoder may overtake the baseline.
- The model file is a joblib pickle: only load files you built yourself.
