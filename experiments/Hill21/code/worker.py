from pathlib import Path
import sys,json,shutil,time,hashlib,traceback
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'outputs'
import training_core as core
from sgsa_model import SGSA
from multigsa_adapted import MultiGSAAdapted
from sgsa_utils.checkpoints import load_checkpoint


def validate_protocol(cfg):
 expected=dict(num_nodes=21,input_dim=7,d_model=64,graph_rank=16,rho=.25,
               history=96,horizon=48,learning_rate=.001,dropout=.1,batch_size=32,
               epochs=50,patience=8,weight_decay=.0001,gradient_clip=1.,
               selection='validation MAE',metric_scale='training-only z-score',
               seeds=[2024,2025,2026],split=[0,3058,3713,4369])
 changed=[key for key,value in expected.items() if cfg.get(key)!=value]
 if changed:raise ValueError('Hill21 config disagrees with the paper protocol: '+', '.join(changed))

def build(name,n,f,drop,rho=.25,device='cuda'):
 if name=='SGSA':return SGSA(96,n,f,48,64,drop,16,rho)
 if name=='MultiGSA_adapted':return MultiGSAAdapted(96,n,f,48,64,drop)
 if name in ['LSTM','Transformer','MTGNN','FEDformer']:
  if name=='FEDformer':
   from types import ModuleType
   pkg=ModuleType('layers');pkg.__path__=[str(Path(__file__).resolve().parents[3]/'vendor/FEDformer/layers')];sys.modules['layers']=pkg
  import adapters_base
  if name=='MTGNN':return adapters_base.MTGNNAdapter(device=device)
  return adapters_base.build(name)
 if name in ['GraphWaveNet','AGCRN','STID_no_calendar','DLinear','PatchTST']:
  if name=='PatchTST':
   from types import ModuleType
   pkg=ModuleType('layers');pkg.__path__=[str(Path(__file__).resolve().parents[3]/'vendor/PatchTST/PatchTST_supervised/layers')];sys.modules['layers']=pkg
  import adapters_extended
  return adapters_extended.build(name)
 import adapters_extra
 return adapters_extra.build(name)

def main(name,device='cuda',evaluate_only=False,seeds=None,strategies=None):
 if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; use --evaluate-only --device cpu for frozen inference.')
 if device=='cpu' and not evaluate_only:raise ValueError('CPU is supported for frozen evaluation only.')
 core.ROOT=ROOT;core.build=build;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=dict(np.load(ROOT/'data/panel.npz'));cfg=json.loads((ROOT/'config.json').read_text());rows=[]
 validate_protocol(cfg)
 assert data['x'].shape[1:]==(cfg['num_nodes'],cfg['input_dim'])==(21,7)
 assert (cfg['history'],cfg['horizon'])==(core.SEQ_LEN,core.PRED_LEN),'config.json window shape drifted'
 for strategy in strategies or ['clean','noise']:
  for seed in seeds or cfg['seeds']:
   path=ROOT/f'runs/{name}/{strategy}/{seed}'
   if not evaluate_only:core.train(name,cfg,seed,strategy,path,data)
   model=build(name,21,7,cfg['dropout'],cfg['rho'],device=device)
   load_checkpoint(model,path/'best.pt',ROOT/'checkpoint_metadata')
   model=model.to(device);model.eval()
   parameters=core.real_scalars(model)
   for condition in ['clean','corrupted']:
    score=core.evaluate(model,core.Windows(data,'test',condition=condition),path/condition)
    # Paper Table 5: the test partition has a fixed number of valid prediction points.
    assert score['valid_points']==core.EXPECTED_VALID_TEST_POINTS,'Valid test point drift'
    rows.append(dict(model=name,strategy=strategy,seed=seed,condition=condition,parameters=parameters,**score))
   dest=PAPER/f'runs/{name}/{strategy}/{seed}';dest.mkdir(parents=True,exist_ok=True)
   for f in ['best.pt','config.json','history.csv','complete.json']:shutil.copy2(path/f,dest/f)
   sidecar=path/'best.pt.metadata.json'
   if sidecar.exists():shutil.copy2(sidecar,dest/sidecar.name)
   (dest/'checkpoint_sha256.txt').write_text(hashlib.sha256((path/'best.pt').read_bytes()).hexdigest())
   # Same-device inference efficiency; no AMP or TF32. Complex parameters count as two real scalars.
   if strategy=='clean' and device=='cuda':
    measures=[]
    with torch.no_grad():
     t=int(data['test_starts'][0]);sample=torch.from_numpy(data['x'][None,t-96:t]).cuda()
     for batch in [1,32]:
      x=sample.repeat(batch,1,1,1)
      for _ in range(10):model(x)
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
      for _ in range(30):
       a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();model(x);b.record();torch.cuda.synchronize();times.append(a.elapsed_time(b))
      measures.append(dict(model=name,seed=seed,batch=batch,parameters=parameters,warmup=10,repeats=30,latency_ms=float(np.median(times)),peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,gpu=torch.cuda.get_device_name()))
      del x
    (dest/'efficiency.json').write_text(json.dumps(measures,indent=2));del sample
   pd.DataFrame(rows).to_csv(PAPER/f'{name}_metrics.csv',index=False)
   del model;torch.cuda.empty_cache()
 core.status(stage='model_complete',model=name,evaluations=len(rows))
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description='Fixed-protocol Hill of Towie benchmark; no hyperparameter search.')
 parser.add_argument('model',choices=['SGSA','MultiGSA_adapted','LSTM','Transformer','MTGNN','DLinear','STID_no_calendar','GraphWaveNet','AGCRN','PatchTST','FEDformer','PDFormer_train_graph','StemGNN_power_only'])
 parser.add_argument('--evaluate-only',action='store_true');parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
 parser.add_argument('--seeds',type=int,nargs='+');parser.add_argument('--strategies',choices=['clean','noise'],nargs='+')
 args=parser.parse_args()
 try:main(args.model,args.device,args.evaluate_only,args.seeds,args.strategies)
 except Exception:core.ROOT=ROOT;core.status(stage='failed',model=sys.argv[1],error=traceback.format_exc());raise
