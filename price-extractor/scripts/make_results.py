import json, os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from score import evaluate, ROOT
E_=lambda t,len_=False: evaluate(f"{ROOT}/eval/pred_{t}.jsonl",lenient=len_)
tags=["base","v1","v2"]; R={t:E_(t) for t in tags}; RL={t:E_(t,True) for t in tags}
R0={t:E_(t+"_norp") for t in ("base","v1")}
sel=json.load(open(f"{ROOT}/adapters-v2/best/selection.json"))
with open(f"{ROOT}/eval/predictions.jsonl","w") as f:
    for t in tags+["base_norp","v1_norp"]:
        for l in open(f"{ROOT}/eval/pred_{t}.jsonl"): f.write(l)
st=json.load(open(f"{ROOT}/data/stats.json"))
pct=lambda x:"n/a" if x is None else f"{100*x:.1f}%"
L=["# Held-out evaluation: base vs v1 (400 iters) vs v2 (best checkpoint)\n",
f"Test set: {R['base']['docs']} documents, {R['base']['gold_lines']} catalogue lines, split by document, never seen in training. Qwen2.5-1.5B-Instruct-4bit, greedy decoding, repetition_penalty 1.1, max 2,048 new tokens, input capped at 2,000 tokens.",
f"v1 = final adapter after 400 iterations (adapters/). v2 = best-validation checkpoint of a 120 iteration run, iteration {sel['best_iter']} (adapters-v2/best/, validation loss {sel['val_loss']}).\n",
"Match rule: item token-F1 >= 0.6 and unit_price within 1%, one-to-one. Field exact match is conditional on a line already matching.\n",
"## Main table (strict scoring: invalid JSON counts as zero lines)\n","| Metric | Base | v1 (400 it) | v2 (best ckpt) |\n|---|---|---|---|"]
def row(n,f): L.append(f"| {n} | "+" | ".join(str(f(R[t])) for t in tags)+" |")
row("Valid JSON rate",lambda r:pct(r['valid_rate'])); row("Invalid JSON documents",lambda r:f"{r['docs']-r['valid_json']}/{r['docs']}")
row("Predicted lines",lambda r:r['pred_lines']); row("Matched lines",lambda r:f"{r['matched']}/{r['gold_lines']}")
row("Line recall",lambda r:pct(r['recall'])); row("Line precision",lambda r:pct(r['precision']))
row("qty exact match",lambda r:pct(r['qty_em'])); row("unit_price exact match",lambda r:pct(r['price_em'])); row("ccy exact match",lambda r:pct(r['ccy_em']))
row("Latency mean s/doc",lambda r:f"{r['lat_mean']:.1f}"); row("Latency median s/doc",lambda r:f"{r['lat_median']:.1f}"); row("Latency max s/doc",lambda r:f"{r['lat_max']:.1f}")
L+=["\n## Effect of the repetition penalty alone (same weights, penalty 1.0 vs 1.1)\n","| Model | Invalid JSON before | Invalid JSON after | Recall before | Recall after | Precision before | Precision after |\n|---|---|---|---|---|---|---|"]
for t,n in(("base","Base"),("v1","v1")):
    a,b=R0[t],R[t]; L.append(f"| {n} | {a['docs']-a['valid_json']}/{a['docs']} | {b['docs']-b['valid_json']}/{b['docs']} | {pct(a['recall'])} | {pct(b['recall'])} | {pct(a['precision'])} | {pct(b['precision'])} |")
L+=["\n## Lenient scoring (complete lines salvaged from broken JSON), penalty 1.1\n","| Metric | Base | v1 | v2 |\n|---|---|---|---|"]
for n,k in(("Predicted lines","pred_lines"),("Matched lines","matched")): L.append(f"| {n} | "+" | ".join(str(RL[t][k]) for t in tags)+" |")
for n,k in(("Line recall","recall"),("Line precision","precision")): L.append(f"| {n} | "+" | ".join(pct(RL[t][k]) for t in tags)+" |")
L+=["\n## Per document (predicted / catalogue / matched), penalty 1.1\n","| Document | Base | v1 | v2 |\n|---|---|---|---|"]
for pb,p1,p2 in zip(R['base']['per_doc'],R['v1']['per_doc'],R['v2']['per_doc']):
    f=lambda p:f"{p[1]}/{p[2]}/{p[3]}"; L.append(f"| {pb[0][:60]} | {f(pb)} | {f(p1)} | {f(p2)} |")
L+=["\n## Validation loss by iteration (v2 run)\n","| Iteration | Val loss |\n|---|---|"]+[f"| {i} | {v} |" for i,v in sel['all_val_loss']]
L.append("\n## Data counts\n\n```\n"+json.dumps(st,indent=1)+"\n```")
open(f"{ROOT}/eval/RESULTS.md","w").write("\n".join(L)+"\n"); print("\n".join(L[:28]))
