"""Usage: evaluate.py TAG [ADAPTER_DIR|none] [REPETITION_PENALTY]
Writes eval/pred_TAG.jsonl. Greedy decoding, max 2,048 new tokens."""
import json, sys, time, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL="mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SYS=open(ROOT+"/scripts/system_prompt.txt").read().strip()
tag=sys.argv[1]; ad=sys.argv[2] if len(sys.argv)>2 else "none"; rp=float(sys.argv[3]) if len(sys.argv)>3 else 1.0
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
model,tok=load(MODEL, adapter_path=(None if ad=="none" else (ad if ad.startswith("/") else ROOT+"/"+ad)))
lp=make_logits_processors(repetition_penalty=rp) if rp!=1.0 else None
docs=[json.loads(l) for l in open(ROOT+"/data/test_docs.jsonl")]
out=open(ROOT+f"/eval/pred_{tag}.jsonl","w")
for i,d in enumerate(docs):
    p=tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":d["text"]}],tokenize=False,add_generation_prompt=True)
    t0=time.time()
    txt=generate(model,tok,prompt=p,max_tokens=2048,sampler=make_sampler(temp=0.0),logits_processors=lp)
    dt=time.time()-t0
    out.write(json.dumps({"doc_id":d["doc_id"],"model":tag,"repetition_penalty":rp,"seconds":round(dt,2),"raw":txt,"gold":d["records"]},ensure_ascii=False)+"\n"); out.flush()
    print(tag,i+1,len(docs),round(dt,1),flush=True)
