"""Build (document text -> JSON records) pairs from the NLS pricebook catalogue. Read-only on inputs."""
import json, os, re, glob, subprocess, random, collections, sys
PB=os.environ.get("PRICEBOOK_ROOT", "data/synthetic")
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/data"
SEED=1234; MAX_TOK=3000; TEST_TOK=2000; SEQ=3072
FIELDS=["item","spec","qty","unit","unit_price","ext_price","ccy","basis","quote_ref","doc_date","valid_until"]

def xlsx_text(p):
    import openpyxl
    wb=openpyxl.load_workbook(p,data_only=True,read_only=True); out=[]
    for ws in wb.worksheets:
        out.append(f"=== SHEET: {ws.title} ===")
        for row in ws.iter_rows(values_only=True):
            if any(c is not None for c in row):
                out.append("\t".join("" if c is None else str(c) for c in row))
    return "\n".join(out)

def pdf_text(p):
    r=subprocess.run(["pdftotext","-layout",p,"-"],capture_output=True,timeout=60)
    return r.stdout.decode("utf8","ignore")

def get_text(path, sha_map):
    ext=os.path.splitext(path)[1].lower()
    try:
        if ext==".pdf": t=pdf_text(path)
        elif ext in(".xlsx",".xlsm"): t=xlsx_text(path)
        elif ext==".txt": t=open(path,encoding="utf8",errors="ignore").read()
        else: return None
    except Exception as e:
        print("EXTRACT ERR",path,type(e).__name__,str(e)[:80],file=sys.stderr); return None
    return t

def clean(t):
    t=t.replace("\f","\n"); t=re.sub(r"[ \t]{3,}","  ",t); t=re.sub(r"\n{3,}","\n\n",t)
    return t.strip()

_tok=None
def ntok(x):
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok=AutoTokenizer.from_pretrained("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    return len(_tok(x)["input_ids"])

def truncate(t, tok):
    """Keep first lines + the most number-dense lines (table region) in original order, within tok tokens."""
    if ntok(t)<=tok: return t
    lines=[l[:400] for l in t.split("\n")]
    def score(l): return len(re.findall(r"\d[\d,\.]*",l))
    head=lines[:12]; rest=lines[12:]
    size=sum(ntok(l)+1 for l in head); keep=set()
    for i in sorted(range(len(rest)),key=lambda i:-score(rest[i])):
        c=ntok(rest[i])+1
        if size+c>tok: continue
        keep.add(i); size+=c
    return "\n".join(head+[rest[i] for i in sorted(keep)])

def price_seen(r,t):
    up=r["unit_price"]
    if up is None: return True
    return any(x in t for x in [f"{up:,.2f}",f"{up:,.0f}",f"{up:.2f}",f"{up:g}",f"{up:,.3f}"])

def main():
    cat=[json.loads(l) for l in open(PB+"/state/catalog.jsonl")]
    stats=collections.Counter()
    sel=[x for x in cat if x["direction"]=="received" and (x["confidence"] or 0)>=0.8]
    stats["catalog_total"]=len(cat); stats["received_conf>=0.8"]=len(sel)
    emails=[x for x in sel if x["origin"]=="email"]
    stats["dropped_pricebook_origin(internal cost-rate md, not supplier docs)"]=len(sel)-len(emails)
    by=collections.defaultdict(list)
    for x in emails: by[x["source_path"]].append(x)
    stats["candidate_docs"]=len(by)
    docs=[]
    for p,recs in sorted(by.items()):
        ext=os.path.splitext(p)[1].lower()
        if ext not in(".pdf",".xlsx",".txt"): stats["skip_ext_"+ext]+=1; continue
        if not os.path.exists(p): stats["skip_missing"]+=1; continue
        t=get_text(p,None)
        if t is None: stats["skip_extract_error"]+=1; continue
        t=clean(t)
        if len(re.findall(r"[A-Za-z0-9]",t))<80: stats["skip_no_text_layer"]+=1; continue
        full=t
        labs=[{k:r.get(k) for k in FIELDS} for r in recs]
        # test-style view: fixed 2000-token input cap, labels = records whose price is visible
        t2=truncate(full,TEST_TOK)
        kept=[l for l in labs if price_seen(l,t2)]
        stats["records_price_not_in_text_dropped"]+=len(labs)-len(kept)
        if not kept: stats["skip_no_visible_records"]+=1; continue
        docs.append({"doc_id":os.path.basename(p),"source_path":p,"full":full,"text":t2,"records":kept})
    stats["docs_final_pre_split"]=len(docs)
    random.Random(SEED).shuffle(docs)
    for d in docs: d["train_view"]=None
    ntest=max(1,round(len(docs)*0.15))
    test,rest=docs[:ntest],docs[ntest:]
    nval=max(3,round(len(rest)*0.1))
    val,train=rest[:nval],rest[nval:]
    SYS=open(os.path.dirname(OUT)+"/scripts/system_prompt.txt").read().strip()
    def fit(d):
        """train/valid only: shrink input until system+input+output fits SEQ; keep only records still visible."""
        recs=d["records"]; budget=TEST_TOK
        while True:
            t=truncate(d["full"],budget); rr=[r for r in recs if price_seen(r,t)]
            out=json.dumps(rr,ensure_ascii=False)
            if not rr: return None
            if ntok(t)+ntok(out)+120<=SEQ or budget<300: break
            budget=int(budget*0.8)
        if ntok(t)+ntok(out)+120>SEQ: return None
        return {**d,"text":t,"records":rr}
    def chat(d): return {"messages":[{"role":"system","content":SYS},{"role":"user","content":d["text"]},
        {"role":"assistant","content":json.dumps(d["records"],ensure_ascii=False)}]}
    os.makedirs(OUT,exist_ok=True)
    ntr0=(len(train),len(val))
    train=[x for x in map(fit,train) if x]; val=[x for x in map(fit,val) if x]
    stats["train_valid_docs_dropped_cannot_fit_3072"]=ntr0[0]+ntr0[1]-len(train)-len(val)
    for name,ds in(("train",train),("valid",val)):
        with open(f"{OUT}/{name}.jsonl","w") as f:
            for d in ds: f.write(json.dumps(chat(d),ensure_ascii=False)+"\n")
    with open(f"{OUT}/test_docs.jsonl","w") as f:
        for d in test: f.write(json.dumps({k:v for k,v in d.items() if k not in("full","train_view")},ensure_ascii=False)+"\n")
    for name,ds in(("train",train),("valid",val),("test",test)):
        stats[f"{name}_docs"]=len(ds); stats[f"{name}_records"]=sum(len(d["records"]) for d in ds)
    dates=sorted(r["doc_date"] for d in docs for r in d["records"] if r["doc_date"])
    stats["docs_total_used"]=len(train)+len(val)+len(test); stats["records_total_used"]=sum(len(d["records"]) for d in train+val+test)
    stats["date_min"]=dates[0]; stats["date_max"]=dates[-1]
    json.dump(stats,open(f"{OUT}/stats.json","w"),indent=1)
    print(json.dumps(stats,indent=1))
if __name__=="__main__": main()
