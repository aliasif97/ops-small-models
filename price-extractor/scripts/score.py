import json, re, os, statistics
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def parse(raw):
    s=raw.strip()
    s=re.sub(r"^```(?:json)?|```$","",s,flags=re.M).strip()
    try: v=json.loads(s)
    except Exception: return None
    if isinstance(v,dict): v=v.get("records",[v])
    if not isinstance(v,list) or not all(isinstance(x,dict) for x in v): return None
    return v
def salvage(raw):
    """Lenient: recover every complete {...} object from truncated / broken JSON output."""
    s=raw; dec=json.JSONDecoder(); out=[]; i=s.find("{")
    while i!=-1:
        try:
            o,end=dec.raw_decode(s,i)
            if isinstance(o,dict) and "item" in o: out.append(o)
            i=s.find("{",end)
        except Exception:
            i=s.find("{",i+1)
    return out

def toks(x): return re.findall(r"[a-z0-9]+",str(x or "").lower())
def f1(a,b):
    a,b=toks(a),toks(b)
    if not a or not b: return 0.0
    from collections import Counter
    c=sum((Counter(a)&Counter(b)).values())
    if c==0: return 0.0
    p,r=c/len(a),c/len(b); return 2*p*r/(p+r)
def num(x):
    try: return float(str(x).replace(",",""))
    except Exception: return None
def close(a,b):
    a,b=num(a),num(b)
    if a is None or b is None: return False
    return abs(a-b)<=0.01*max(abs(b),1e-9)
def match(pred,gold):
    """greedy one-to-one matching; item token-F1>=0.6 and unit_price within 1%"""
    cands=[]
    for i,p in enumerate(pred):
        for j,g in enumerate(gold):
            if close(p.get("unit_price"),g.get("unit_price")):
                s=f1(p.get("item"),g.get("item"))
                if s>=0.6: cands.append((s,i,j))
    cands.sort(reverse=True); ui,uj,m=set(),set(),[]
    for s,i,j in cands:
        if i in ui or j in uj: continue
        ui.add(i);uj.add(j);m.append((i,j))
    return m
def norm(x): return str(x).strip().upper() if x is not None else None
def evaluate(path,lenient=False):
    rows=[json.loads(l) for l in open(path)]
    n=len(rows); valid=0; tp=0; npred=0; ngold=0; lat=[]
    fm={"qty":0,"unit_price":0,"ccy":0}; nm=0; per_doc=[]
    for r in rows:
        lat.append(r["seconds"]); gold=r["gold"]; ngold+=len(gold)
        pred=parse(r["raw"])
        if pred is None and lenient: pred=salvage(r["raw"]); valid+=0
        if pred is None: per_doc.append((r["doc_id"],0,len(gold),0)); continue
        valid+=1 if parse(r["raw"]) is not None else 0; npred+=len(pred); m=match(pred,gold); tp+=len(m)
        for i,j in m:
            nm+=1
            fm["qty"]+=close(pred[i].get("qty"),gold[j].get("qty")) or (pred[i].get("qty") is None and gold[j].get("qty") is None)
            fm["unit_price"]+=close(pred[i].get("unit_price"),gold[j].get("unit_price")) and num(pred[i].get("unit_price"))==num(gold[j].get("unit_price"))
            fm["ccy"]+=norm(pred[i].get("ccy"))==norm(gold[j].get("ccy"))
        per_doc.append((r["doc_id"],len(pred),len(gold),len(m)))
    return {"docs":n,"valid_json":valid,"valid_rate":valid/n,"pred_lines":npred,"gold_lines":ngold,"matched":tp,
        "recall":tp/ngold if ngold else 0,"precision":tp/npred if npred else 0,
        "qty_em":fm["qty"]/nm if nm else None,"price_em":fm["unit_price"]/nm if nm else None,"ccy_em":fm["ccy"]/nm if nm else None,
        "lat_mean":statistics.mean(lat),"lat_median":statistics.median(lat),"lat_max":max(lat),"per_doc":per_doc}
if __name__=="__main__":
    for w in("base","adapter"):
        p=f"{ROOT}/eval/pred_{w}.jsonl"
        if os.path.exists(p):
            r=evaluate(p); r.pop("per_doc"); print(w,json.dumps(r,indent=1))
