import json, os, random
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
ROLES = ["customer_rfq", "customer_followup", "our_rfq_to_vendor", "vendor_pitch",
         "prequal_registration", "admin_noise", "other"]

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

def load_rows():
    with open(f"{ROOT}/data/threads.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def model_text(r):
    """Single string fed to both models. 'query: ' prefix is the e5 convention."""
    d = {"i": "inbound", "o": "outbound"}.get(r.get("direction", ""), "unknown")
    return f"query: {d} | {r.get('sender_domain','')} | {r.get('subject','')} | {r.get('text','')}"

def split(rows):
    """80/20 stratified by role, then 12.5% of train as validation (10% of all)."""
    from sklearn.model_selection import train_test_split
    y = [r["role"] for r in rows]
    idx = np.arange(len(rows))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    ytr = [y[i] for i in tr]
    tr2, va = train_test_split(tr, test_size=0.125, stratify=ytr, random_state=SEED)
    return list(tr2), list(va), list(te)

def metrics(y_true, y_pred, labels=ROLES):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, l in enumerate(labels)},
        "labels": labels,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
