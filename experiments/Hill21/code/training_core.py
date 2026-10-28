from pathlib import Path
import os,json,time,random,csv,hashlib,shutil,traceback,itertools,sys
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from sgsa_utils.checkpoints import save_checkpoint
# Model factory is supplied explicitly by worker.py
ROOT=Path(__file__).resolve().parents[1]; PAPER=ROOT/'outputs'
# Window shape and the paper's valid-test-point count (Table 5).
SEQ_LEN,PRED_LEN=96,48
EXPECTED_VALID_TEST_POINTS=26187
def real_scalars(model):
 # A complex parameter counts as two real scalars, as stated in the paper.
 return sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def status(**kw):write(ROOT/'status.json',dict(updated=time.time(),**kw))
class Windows(Dataset):
 def __init__(self,data,part,noise=False,condition='clean'):
  self.data=data;self.starts=data[part+'_starts'];self.noise=noise;self.condition=condition;self.epoch=0
 def __len__(self):return len(self.starts)
 def __getitem__(self,i):
  t=int(self.starts[i]);x=self.data['x'][t-SEQ_LEN:t].copy();y=self.data['y'][t:t+PRED_LEN];m=self.data['mask'][t:t+PRED_LEN]
  seed=(2024+t*1009+self.epoch*1000003+(701 if self.condition=='corrupted' else 0))%2**32;rng=np.random.default_rng(seed)
  level='none'
  if self.condition=='corrupted':level='heavy'
  elif self.noise:
   u=rng.random();level='none' if u<.5 else ('light' if u<.8 else 'heavy')
  if level!='none':
   rate=.05 if level=='light' else .2;packet=rng.random(x.shape[:2])<rate
   fill=(self.data['median']-self.data['mean'])/self.data['std'];x[packet]=fill
   if level=='heavy':
    bad=(rng.random(x.shape)<.05)&(~packet[...,None]);x[bad]=rng.choice([-8.,8.],bad.sum())
  return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(m),t

def loader(ds,shuffle,seed,batch_size=32):
 g=torch.Generator().manual_seed(seed)
 return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=0,generator=g)
@torch.no_grad()
def evaluate(model,ds,save=None):
 model.eval();sa=ss=count=0;ah=np.zeros(PRED_LEN);sh=ah.copy();ch=ah.copy();pred=[];target=[];mask=[];times=[]
 for x,y,m,t in loader(ds,False,0):
  p=model(x.to(next(model.parameters()).device)).cpu().numpy();yy=y.numpy();mm=m.numpy();e=p-yy
  sa+=np.abs(e)[mm].sum(dtype=np.float64);ss+=np.square(e)[mm].sum(dtype=np.float64);count+=int(mm.sum())
  ah+=(np.abs(e)*mm).sum((0,2));sh+=(e**2*mm).sum((0,2));ch+=mm.sum((0,2))
  if save is not None:pred.append(p);target.append(yy);mask.append(mm);times.extend(t.numpy().tolist())
 if not count:raise ValueError('Empty evaluation')
 d=dict(mae=sa/count,rmse=(ss/count)**.5,mse=ss/count,valid_points=count,windows=len(ds))
 if save is not None:
  np.savez_compressed(str(save)+'.npz',prediction=np.concatenate(pred),target=np.concatenate(target),mask=np.concatenate(mask),origins=np.array(times),turbine_ids=ds.data['turbine_ids'])
  np.savetxt(str(save)+'_horizon.csv',np.stack([np.arange(1,PRED_LEN+1),ah/np.maximum(ch,1),(sh/np.maximum(ch,1))**.5,ch],1),delimiter=',',header='horizon,mae,rmse,count',comments='')
 return d

def train(name,cfg,seed,strategy,path,data):
 if (path/'complete.json').exists():
  saved=json.loads((path/'config.json').read_text())
  for key in ['learning_rate','dropout']:
   if saved[key]!=cfg[key]:raise ValueError('Existing checkpoint configuration mismatch')
  if name=='SGSA' and saved.get('rho',.25)!=cfg.get('rho',.25):raise ValueError('Existing checkpoint rho mismatch')
  return json.loads((path/'complete.json').read_text())
 path.mkdir(parents=True,exist_ok=True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 model=build(name,data['x'].shape[1],data['x'].shape[2],cfg['dropout'],rho=cfg.get('rho',.25)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg.get('weight_decay',1e-4))
 tr=Windows(data,'train',noise=strategy=='noise');val=Windows(data,'validation',noise=strategy=='noise')
 best=float('inf');stale=0;history=[];write(path/'config.json',dict(cfg,seed=seed,strategy=strategy,model=name,rho=cfg.get('rho',.25) if name=='SGSA' else None))
 for epoch in range(1,cfg.get('epochs',50)+1):
  model.train();tr.epoch=epoch;sa=ss=count=0;tick=time.time()
  for x,y,m,_ in loader(tr,True,seed+epoch,cfg.get('batch_size',32)):
   x=x.cuda();y=y.cuda();m=m.cuda();opt.zero_grad(set_to_none=True);e=(model(x)-y)[m]
   if not len(e):continue
   loss=e.square().mean()
   if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.get('gradient_clip',1));opt.step();ss+=float(e.detach().square().sum());count+=e.numel()
  score=evaluate(model,val);history.append(dict(epoch=epoch,train_mse=ss/count,val_mae=score['mae'],val_rmse=score['rmse'],seconds=time.time()-tick))
  with (path/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(history[-1]));w.writeheader();w.writerows(history)
  print(name,strategy,seed,cfg,history[-1],flush=True)
  status(stage='training',model=name,strategy=strategy,seed=seed,epoch=epoch,run=str(path),validation_mae=score['mae'])
  if score['mae']<best:
   best=score['mae'];stale=0;save_checkpoint(model,path/'best.pt');best_epoch=epoch
  else:stale+=1
  if stale>=cfg.get('patience',8):break
 result=dict(best_validation_mae=best,best_epoch=best_epoch,epochs=len(history),parameters=real_scalars(model));write(path/'complete.json',result)
 del model,opt;torch.cuda.empty_cache();return result
