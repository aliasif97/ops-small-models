# Using extract.py

## Staff use
From a terminal:

    cd price-extractor
    ../.venv-mlx/bin/python extract.py "/path/to/supplier-quote.pdf"

Takes 5 to 25 seconds (first call also loads the model). It prints JSON: records (the price lines), valid_json, grounded_fraction and needs_cloud_fallback. Works on PDF, xlsx and txt. Scanned PDFs return an error saying no text layer. Always read the lines against the PDF before using a price: today it finds about a quarter of the lines on a typical document, with about three in four of the lines it does return being right.

## Pricebook refresh: local first, cloud on low confidence
The refresh currently sends every new document to the cloud model. Suggested order, per document:

1. Call extract(path) from extract.py (import it, load the model once per run and pass model_tok to avoid reloading).
2. Accept the local result only if valid_json is true, records is not empty, grounded_fraction >= 0.9 and needs_cloud_fallback is false. Also accept it only if the line count looks sane: if the document text has many more number-dense rows than records returned (for example more than twice), send it to the cloud.
3. Otherwise send the document to the cloud model as today, and keep its output.
4. Log which path produced each record (a field such as extractor=local or cloud) so the two can be compared and the adapter retrained on cloud-corrected rows.

Expected effect with the current adapter: small single-line quotes (the majority of documents) can stay local; long tables will mostly fall back to the cloud. Do not change the pricebook code until the adapter's recall is raised; this note is a design only, nothing in /Users/aliasif/apps/nls-pricebook was touched.
