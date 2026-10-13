"""Fine-tune multilingual-e5-small for 7-way sequence classification."""
import json, time, os, numpy as np, torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from common import *
BASE = "intfloat/multilingual-e5-small"
MAXLEN, BS, LR, HEAD_LR, EPOCHS, PATIENCE = 256, 8, 1e-4, 2e-3, 5, 2  # run 1 used BS16/LR5e-5 and underfit
seed_all()
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device", dev)
rows = load_rows(); tr, va, te = split(rows)
l2i = {l: i for i, l in enumerate(ROLES)}
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(ROLES),
        id2label={i: l for l, i in l2i.items()}, label2id=l2i).to(dev)

def enc(idx):
    e = tok([model_text(rows[i]) for i in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    return e, torch.tensor([l2i[rows[i]["role"]] for i in idx])

def predict(idx, bs=32):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            e, _ = enc(idx[s:s+bs])
            out.append(torch.softmax(model(**{k: v.to(dev) for k, v in e.items()}).logits.float(), -1).cpu())
    return torch.cat(out).numpy()

cnt = np.bincount([l2i[rows[i]["role"]] for i in tr], minlength=len(ROLES))
cw = torch.tensor((len(tr) / (len(ROLES) * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32).to(dev)  # sqrt-balanced
lossf = nn.CrossEntropyLoss(weight=cw)
head = [p for n, p in model.named_parameters() if n.startswith("classifier")]
body = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
opt = torch.optim.AdamW([{"params": body, "lr": LR}, {"params": head, "lr": HEAD_LR}], weight_decay=0.01)
steps = EPOCHS * ((len(tr) + BS - 1) // BS)
sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
rng = np.random.RandomState(SEED)
best, bad, t0, log = -1, 0, time.time(), []
for ep in range(EPOCHS):
    model.train(); order = rng.permutation(tr); tot = 0
    for s in range(0, len(order), BS):
        e, yb = enc(list(order[s:s+BS]))
        loss = lossf(model(**{k: v.to(dev) for k, v in e.items()}).logits, yb.to(dev))
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(); tot += loss.item() * len(yb)
        if dev.type == 'mps' and (s // BS) % 10 == 0: torch.mps.empty_cache()
    pv = predict(va); yv = [rows[i]["role"] for i in va]
    mv = metrics(yv, [ROLES[j] for j in pv.argmax(1)])
    log.append(dict(epoch=ep + 1, train_loss=tot / len(tr), val_acc=mv["accuracy"], val_macro_f1=mv["macro_f1"], elapsed_s=time.time() - t0))
    print(log[-1], flush=True)
    if mv["macro_f1"] > best:
        best, bad = mv["macro_f1"], 0
        model.save_pretrained(f"{ROOT}/models/thread-classifier-v1"); tok.save_pretrained(f"{ROOT}/models/thread-classifier-v1")
    else:
        bad += 1
        if bad >= PATIENCE: print("early stop"); break
train_s = time.time() - t0
# reload the best checkpoint, evaluate val + the untouched test set
model = AutoModelForSequenceClassification.from_pretrained(f"{ROOT}/models/thread-classifier-v1").to(dev)
json.dump({"labels": ROLES, "base_model": BASE, "max_length": MAXLEN, "input_format": "query: {direction} | {sender_domain} | {subject} | {text}"},
          open(f"{ROOT}/models/thread-classifier-v1/labels.json", "w"), indent=2)
pv, pt = predict(va), predict(te)
yt = [rows[i]["role"] for i in te]
m = metrics(yt, [ROLES[j] for j in pt.argmax(1)])
m["train_seconds"] = train_s; m["epoch_log"] = log; m["device"] = str(dev)
json.dump(m, open(f"{ROOT}/eval/finetuned_metrics.json", "w"), indent=2)
with open(f"{ROOT}/eval/test_predictions.jsonl", "w") as f:
    for i, p in zip(te, pt):
        f.write(json.dumps(dict(rn=rows[i]["rn"], label=rows[i]["role"], pred=ROLES[int(p.argmax())], conf=float(p.max()), probs=[float(x) for x in p]), ensure_ascii=False) + "\n")
with open(f"{ROOT}/eval/val_predictions.jsonl", "w") as f:
    for i, p in zip(va, pv):
        f.write(json.dumps(dict(rn=rows[i]["rn"], label=rows[i]["role"], pred=ROLES[int(p.argmax())], conf=float(p.max()))) + "\n")
print("TEST acc %.4f macroF1 %.4f train_s %.0f" % (m["accuracy"], m["macro_f1"], train_s))
