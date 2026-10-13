#!/usr/bin/env python3
"""Classify one email thread. Default model: TF-IDF + logistic regression (baseline-v1).
  python classify.py "Subject text or free text"
  python classify.py '{"direction":"i","sender_domain":"example-supplier.com","subject":"Urgent RFQ ...","text":"..."}'
  echo '<json line>' | python classify.py -
  python classify.py --model finetuned '<same input>'     # multilingual-e5-small encoder
Prints JSON: role, confidence, top3, gate ("accept" if confidence >= 0.90 else "fallback"), latency_ms."""
import sys, json, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
GATE = 0.90
_state = {}

def _fields(x):
    if isinstance(x, str):
        return "unknown", "", "", x
    return ({"i": "inbound", "o": "outbound"}.get(x.get("direction", ""), "unknown"),
            x.get("sender_domain", ""), x.get("subject", ""), x.get("text", ""))

def to_text(x, e5=False):
    d, dom, sub, txt = _fields(x)
    return ("query: " if e5 else "") + f"{d} | {dom} | {sub} | {txt}"

def load(model="baseline", device="cpu"):
    _state["model"] = model
    if model == "baseline":
        import joblib
        _state["b"] = joblib.load(f"{HERE}/models/thread-classifier-baseline-v1/model.joblib")
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        D = f"{HERE}/models/thread-classifier-v1"
        meta = json.load(open(f"{D}/labels.json"))
        _state.update(tok=AutoTokenizer.from_pretrained(D), m=AutoModelForSequenceClassification.from_pretrained(D).to(device).eval(),
                      labels=meta["labels"], maxlen=meta["max_length"], device=device)

def classify(x):
    if _state["model"] == "baseline":
        from scipy.sparse import hstack
        b = _state["b"]; t = [to_text(x)]
        p = b["clf"].predict_proba(hstack([b["word"].transform(t), b["char"].transform(t)]).tocsr())[0]
        labels = list(b["clf"].classes_)
    else:
        import torch
        e = _state["tok"](to_text(x, True), truncation=True, max_length=_state["maxlen"], return_tensors="pt").to(_state["device"])
        with torch.no_grad():
            p = torch.softmax(_state["m"](**e).logits.float(), -1)[0].cpu().numpy()
        labels = _state["labels"]
    order = p.argsort()[::-1][:3]
    conf = float(p[order[0]])
    return {"role": labels[order[0]], "confidence": round(conf, 4),
            "top3": [{"role": labels[i], "p": round(float(p[i]), 4)} for i in order],
            "gate": "accept" if conf >= GATE else "fallback"}

if __name__ == "__main__":
    args = sys.argv[1:]; model = "baseline"
    if "--model" in args:
        i = args.index("--model"); model = args[i + 1]; del args[i:i + 2]
        if model in ("finetuned", "encoder", "e5"): model = "finetuned"
        elif model != "baseline": sys.exit("--model must be baseline or finetuned")
    if not args: sys.exit(__doc__)
    a = sys.stdin.readline().strip() if args[0] == "-" else args[0]
    try: x = json.loads(a)
    except Exception: x = a
    load(model)
    t = time.perf_counter(); r = classify(x); r["latency_ms"] = round((time.perf_counter() - t) * 1000, 1)
    r["model"] = model
    print(json.dumps(r, ensure_ascii=False, indent=2))
