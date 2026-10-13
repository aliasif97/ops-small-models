"""Disagreement list (30 rows) and confidence-threshold curve."""
import json, numpy as np
from common import *
rows = {r["rn"]: r for r in load_rows()}
P = [json.loads(l) for l in open(f"{ROOT}/eval/test_predictions.jsonl")]
V = [json.loads(l) for l in open(f"{ROOT}/eval/val_predictions.jsonl")]
dis = sorted([p for p in P if p["pred"] != p["label"]], key=lambda p: -p["conf"])
with open(f"{ROOT}/eval/disagreements.md", "w") as f:
    f.write(f"# Model vs Haiku label: disagreements on the untouched test set\n\n"
            f"Test rows: {len(P)}. Disagreements: {len(dis)}. Listed: {min(30, len(dis))}, highest model confidence first.\n\n"
            "The Haiku label is a silver label, not ground truth. Neither side is assumed right. "
            "A human should mark each row: model right, Haiku right, or both defensible.\n\n")
    for k, p in enumerate(dis[:30], 1):
        r = rows[p["rn"]]
        f.write(f"## {k}. row {p['rn']}\n- Haiku label: `{p['label']}` (category: {r['category'] or 'none'})\n"
                f"- Model: `{p['pred']}` (confidence {p['conf']:.2f})\n- Direction/sender domain: {r['direction']} / {r['sender_domain']}\n"
                f"- Subject: {r['subject']}\n- Snippet: {r['text']}\n- Verdict (fill in): \n\n")
def curve(S, ths):
    out = []
    for t in ths:
        a = [p for p in S if p["conf"] >= t]
        out.append(dict(threshold=t, coverage=len(a) / len(S), accuracy_on_accepted=(sum(p["pred"] == p["label"] for p in a) / len(a)) if a else None, n_accepted=len(a)))
    return out
ths = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
cv, ct = curve(V, ths), curve(P, ths)
# recommended: lowest threshold whose accuracy on the VALIDATION slice is >= 0.95 (chosen without touching test)
rec = next((c["threshold"] for c in cv if c["accuracy_on_accepted"] is not None and c["accuracy_on_accepted"] >= 0.95 and c["n_accepted"] >= 20), 0.9)
json.dump(dict(recommended_threshold=rec, validation=cv, test=ct), open(f"{ROOT}/eval/threshold_curve.json", "w"), indent=2)
print("recommended", rec); [print(c) for c in ct]
