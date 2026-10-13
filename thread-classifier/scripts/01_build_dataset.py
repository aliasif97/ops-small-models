"""Merge the exported part files into data/threads.jsonl and write data/README.md."""
import glob, json, re, collections
from common import ROOT, ROLES
rows = []
for p in sorted(glob.glob(f"{ROOT}/data/parts/part_*.txt"), key=lambda s: int(re.search(r"part_(\d+)_", s).group(1))):
    for line in open(p, encoding="utf-8").read().split("\n"):
        line = line.rstrip("\\ ").strip()
        if not line: continue
        f = line.split("¦")
        if len(f) < 8:
            f += [""] * (8 - len(f))
        n, role, cat, comp, d, dom, subj = f[:7]
        text = "¦".join(f[7:])
        rows.append(dict(rn=int(n), role=role, category=cat, company=comp, direction=d,
                         sender_domain=dom, subject=subj, text=text))
rows.sort(key=lambda r: r["rn"])
bad = [r["rn"] for r in rows if r["role"] not in ROLES]
assert not bad, f"bad roles at {bad[:10]}"
assert [r["rn"] for r in rows] == list(range(1, len(rows) + 1)), "row numbers not contiguous"
with open(f"{ROOT}/data/threads.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
c = collections.Counter(r["role"] for r in rows)
cc = collections.Counter(r["category"] or "(none)" for r in rows)
ar = sum(bool(re.search(r"[؀-ۿ]", r["subject"] + r["text"])) for r in rows)
with open(f"{ROOT}/data/README.md", "w") as f:
    f.write(f"""# Training data

Source: Supabase project "Northlane Solution - Dashboard", tables `rfq_tracker_threads` (labels) and
`rfq_tracker_messages` (text), read with SELECT only on 2026-10-12. Messages in the table span
2026-08-11 to 2026-10-13.

Rows: {len(rows)} threads (1 thread with a null role was dropped; 1,153 in the table).

## Labels are silver labels
Every label is the output of Claude Haiku in the daily RFQ tracker. No human reviewed them.
A model trained here learns to imitate Haiku, including its mistakes. Agreement with Haiku is not accuracy.

## Fields in threads.jsonl
rn (row index), role (the label), category, company (Haiku output, kept as is), direction of the first
message (i inbound, o outbound), sender_domain of the first message, subject (first 120 chars),
text (first 250 chars of body excerpt, or the Gmail snippet when no body is stored, with email
addresses replaced by the token <email>).

## De-identification
Email addresses are not stored: the sender is reduced to its domain and addresses inside text become
<email>. Phone numbers, names inside body text, company names and subjects are kept (names and phone
numbers inside body text are not scrubbed).

## Differences from the brief
Text is the first 250 chars of the body excerpt, not 1,500. This was a cost choice for the export path
(results had to be copied through a tool channel). Longer text would probably help a little.

## Label distribution (role)
| role | count | share |
|---|---|---|
""" + "\n".join(f"| {k} | {v} | {v/len(rows):.1%} |" for k, v in c.most_common()) + f"""

Rows containing Arabic characters: {ar} ({ar/len(rows):.1%}).

## Label distribution (category, top 20)
| category | count |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in cc.most_common(20)) + "\n")
print(len(rows), dict(c))
