"""Pick the checkpoint with the lowest validation loss from train_v2.log; copy it to adapters-v2/best/."""
import re, os, shutil, json
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log=open(ROOT+"/train_v2.log").read()
vals=[(int(i),float(v)) for i,v in re.findall(r"Iter (\d+): Val loss ([0-9.]+)",log)]
# iteration 1 is the untrained model; only trained checkpoints are candidates
cands=[(v,i) for i,v in vals if i>0 and os.path.exists(f"{ROOT}/adapters-v2/{i:07d}_adapters.safetensors")]
v,i=min(cands)
d=ROOT+"/adapters-v2/best"; os.makedirs(d,exist_ok=True)
shutil.copy(f"{ROOT}/adapters-v2/{i:07d}_adapters.safetensors",d+"/adapters.safetensors")
shutil.copy(ROOT+"/adapters-v2/adapter_config.json",d+"/adapter_config.json")
json.dump({"best_iter":i,"val_loss":v,"all_val_loss":vals},open(d+"/selection.json","w"),indent=1)
print("best checkpoint iter",i,"val loss",v,"all:",vals)
