#!/usr/bin/env python3
"""Extract price lines from a supplier quote (PDF, xlsx or txt) with the local LoRA model.
Usage: .venv-mlx/bin/python extract.py quote.pdf [--no-adapter] [--tokens 2000]
Prints a JSON object: {"records":[...], "valid_json":bool, "grounded_fraction":float, "needs_cloud_fallback":bool}
"""
import sys, os, json, re, argparse
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE+"/scripts")
from build_data import get_text, clean, truncate, price_seen
from score import parse, salvage
MODEL="mlx-community/Qwen2.5-1.5B-Instruct-4bit"

def extract(path, use_adapter=True, tokens=2000, model_tok=None):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    text=get_text(path,None)
    if text is None: return {"records":[],"valid_json":False,"grounded_fraction":0.0,"needs_cloud_fallback":True,"error":"unsupported or unreadable file"}
    text=clean(text)
    if len(re.findall(r"[A-Za-z0-9]",text))<80:
        return {"records":[],"valid_json":False,"grounded_fraction":0.0,"needs_cloud_fallback":True,"error":"no text layer (scanned/image), use OCR or cloud"}
    text=truncate(text,tokens)
    model,tok=model_tok or load(MODEL, adapter_path=(HERE+"/adapters" if use_adapter else None))
    sysmsg=open(HERE+"/scripts/system_prompt.txt").read().strip()
    prompt=tok.apply_chat_template([{"role":"system","content":sysmsg},{"role":"user","content":text}],tokenize=False,add_generation_prompt=True)
    raw=generate(model,tok,prompt=prompt,max_tokens=2048,sampler=make_sampler(temp=0.0))
    recs=parse(raw)
    if recs is None:
        part=salvage(raw)  # runaway/truncated output: keep complete lines but always flag for cloud fallback
        return {"records":part,"valid_json":False,"grounded_fraction":0.0,"needs_cloud_fallback":True,"note":"output was not valid JSON; partial lines salvaged"}
    # grounding check: a line is grounded if its unit_price literally appears in the document text
    g=[r for r in recs if r.get("unit_price") is not None and isinstance(r.get("unit_price"),(int,float)) and price_seen({"unit_price":r["unit_price"]},text)]
    frac=len(g)/len(recs) if recs else 0.0
    return {"records":recs,"valid_json":True,"grounded_fraction":round(frac,3),"needs_cloud_fallback":(not recs) or frac<0.9}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--no-adapter",action="store_true"); ap.add_argument("--tokens",type=int,default=2000)
    a=ap.parse_args()
    print(json.dumps(extract(a.path,not a.no_adapter,a.tokens),indent=2,ensure_ascii=False))
