from pathlib import Path
import sys,json,shutil,time,hashlib,traceback
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'outputs'
import training_core as core
from sgsa_model import SGSA
from multigsa_adapted import MultiGSAAdapted

def build(name,n,f,drop,rho=.25):
 if name=='SGSA':return SGSA(96,n,f,48,64,drop,16,rho)
 if name=='MultiGSA_adapted':return MultiGSAAdapted(96,n,f,48,64,drop)
 if name in ['LSTM','Transformer','MTGNN','FEDformer']:
  if name=='FEDformer':
   from types import ModuleType
   pkg=ModuleType('layers');pkg.__path__=[str(Path(__file__).resolve().parents[3]/'vendor/FEDformer/layers')];sys.modules['layers']=pkg
  import adapters_base
  return adapters_base.build(name)
 if name in ['GraphWaveNet','AGCRN','STID_no_calendar','DLinear','PatchTST']:
  if name=='PatchTST':
   from types import ModuleType
   pkg=ModuleType('layers');pkg.__path__=[str(Path(__file__).resolve().parents[3]/'vendor/PatchTST/PatchTST_supervised/layers')];sys.modules['layers']=pkg
  import adapters_extended
  return adapters_extended.build(name)
 import adapters_extra
 return adapters_extra.build(name)

def main(name):
 if name=='FEDformer':raise RuntimeError('Historical Hill21 FEDformer evaluation requires frequency-index audit; see REPRODUCIBILITY.md. This entry deliberately blocks repeating the faulty reload.')
 assert torch.cuda.is_available()
 core.ROOT=ROOT;core.build=build;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=dict(np.load(ROOT/'data/panel.npz'));cfg=dict(learning_rate=.001,dropout=.1,rho=.25);rows=[]
 for strategy in ['clean','noise']:
  for seed in [2024,2025,2026]:
   path=ROOT/f'runs/{name}/{strategy}/{seed}'
   core.train(name,cfg,seed,strategy,path,data)
   model=build(name,21,7,.1,.25).cuda();model.load_state_dict(torch.load(path/'best.pt',map_location='cpu'));model.eval()
   parameters=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
   for condition in ['clean','corrupted']:
    score=core.evaluate(model,core.Windows(data,'test',condition=condition),path/condition)
    rows.append(dict(model=name,strategy=strategy,seed=seed,condition=condition,parameters=parameters,**score))
   dest=PAPER/f'runs/{name}/{strategy}/{seed}';dest.mkdir(parents=True,exist_ok=True)
   for f in ['best.pt','config.json','history.csv','complete.json']:shutil.copy2(path/f,dest/f)
   (dest/'checkpoint_sha256.txt').write_text(hashlib.sha256((path/'best.pt').read_bytes()).hexdigest())
   # Same-device inference efficiency; no AMP or TF32. Complex parameters count as two real scalars.
   if strategy=='clean':
    measures=[]
    with torch.no_grad():
     t=int(data['test_starts'][0]);sample=torch.from_numpy(data['x'][None,t-96:t]).cuda()
     for batch in [1,32]:
      x=sample.repeat(batch,1,1,1)
      for _ in range(10):model(x)
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
      for _ in range(30):
       a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();model(x);b.record();torch.cuda.synchronize();times.append(a.elapsed_time(b))
      measures.append(dict(model=name,seed=seed,batch=batch,parameters=parameters,latency_ms=float(np.median(times)),peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,gpu=torch.cuda.get_device_name()))
      del x
    (dest/'efficiency.json').write_text(json.dumps(measures,indent=2));del sample
   pd.DataFrame(rows).to_csv(PAPER/f'{name}_metrics.csv',index=False)
   del model;torch.cuda.empty_cache()
 core.status(stage='model_complete',model=name,evaluations=len(rows))
if __name__=='__main__':
 try:main(sys.argv[1])
 except Exception:core.ROOT=ROOT;core.status(stage='failed',model=sys.argv[1],error=traceback.format_exc());raise
