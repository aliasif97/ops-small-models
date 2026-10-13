"""Per-thread latency, model already loaded, single process, 100 test threads."""
import sys, json, time, statistics, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import classify as C
from common import load_rows, split, ROOT
rows = load_rows(); te = split(rows)[2][:100]
out = {}
for name, model, dev in [("baseline_cpu", "baseline", "cpu"), ("finetuned_cpu", "finetuned", "cpu"), ("finetuned_mps", "finetuned", "mps")]:
    C.load(model, dev); C.classify(rows[0])
    ts = []
    for i in te:
        t = time.perf_counter(); C.classify(rows[i]); ts.append((time.perf_counter() - t) * 1000)
    ts.sort(); out[name] = dict(mean_ms=round(statistics.mean(ts), 2), p50_ms=round(ts[50], 2), p95_ms=round(ts[94], 2), n=len(ts))
print(json.dumps(out, indent=1)); json.dump(out, open(f"{ROOT}/eval/latency.json", "w"), indent=1)
