"""TF-IDF (word + char n-grams) + logistic regression baseline."""
import json, os, joblib, numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import *
rows = load_rows(); tr, va, te = split(rows)
trall = tr + va   # baseline has no early stopping, so it can use train+val (same 80%)
T = [model_text(r).replace("query: ", "", 1) for r in rows]
y = np.array([r["role"] for r in rows])
w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True, max_features=200000)
Xtr = hstack([w.fit_transform([T[i] for i in trall]), c.fit_transform([T[i] for i in trall])]).tocsr()
Xte = hstack([w.transform([T[i] for i in te]), c.transform([T[i] for i in te])]).tocsr()
clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced", random_state=SEED)
clf.fit(Xtr, y[trall])
pred = clf.predict(Xte)
m = metrics(list(y[te]), list(pred))
json.dump(m, open(f"{ROOT}/eval/baseline_metrics.json", "w"), indent=2)
print("baseline acc %.4f macroF1 %.4f" % (m["accuracy"], m["macro_f1"]))
pr = clf.predict_proba(Xte); conf = pr.max(1)
curve = []
for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
    a = conf >= t
    curve.append(dict(threshold=t, coverage=float(a.mean()), accuracy_on_accepted=float((pred[a] == y[te][a]).mean()) if a.any() else None))
json.dump(curve, open(f"{ROOT}/eval/baseline_threshold_curve.json", "w"), indent=1)
print(curve)

out = f"{ROOT}/models/thread-classifier-baseline-v1"; os.makedirs(out, exist_ok=True)
joblib.dump({"word": w, "char": c, "clf": clf}, f"{out}/model.joblib", compress=3)
json.dump({"labels": list(clf.classes_), "input_format": "{direction} | {sender_domain} | {subject} | {text}", "gate_threshold": 0.90,
           "note": "fit on train+validation (80% of data), seed 42"}, open(f"{out}/labels.json", "w"), indent=2)
