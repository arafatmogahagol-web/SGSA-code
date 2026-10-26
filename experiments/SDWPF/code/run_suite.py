"""Restartable 36-training CUDA queue. Each optimizer step covers 32 windows."""
import argparse,csv,hashlib,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import load_and_prepare,Windows,loader
from models import NAMES,build,parameter_counts,frequency_indices

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'runs'
PAPER=ROOT/'outputs'
DATA=ROOT/'data'/'data_20000.npz'
CFG=dict(epochs=30,patience=5,batch_size=32,learning_rate=.0005,weight_decay=.0001,gradient_clip=1.,
    seeds=[2024,2025,2026],strategies=['clean','noise'],models=NAMES,seq_len=96,pred_len=48,d_model=64,
    split={'train':[0,14000],'val':[14000,15000],'unused':[15000,16000],'test':[16000,20000]},
    train_stride=4,evaluation_stride=24,dropout=.1,loss='masked MSE',selection='validation MSE only',
    augmentation_seed=2024,validation_noise_seed=502024,corruption_seed=910024,
    noise='50% clean / 30% 5% packet loss / 20% (20% packet loss + 5% replacement +/-8 z outliers)',
    test_corruption='20% packet loss + 5% replacement +/-8 z outliers, inputs only',
    scale='clean training-only z-score',precision='float32/complex64, no AMP, no TF32',
    tuning='Fixed controlled configuration, no extra per-model test-driven tuning',
    warning='Previously inspected development test panel, not untouched confirmatory evaluation')

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def csvwrite(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def cuda_setup():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
def load_data():
    d=load_and_prepare(DATA,.7,.1)
    assert d.x_scaled.shape==(20000,134,16) and d.feature_names[-1]=='Patv'
    return d
def split_data(data,strategy):
    return (Windows(data,96,14000,2024,strategy=='noise',4),
            Windows(data,14000,15000,502024,strategy=='noise',24))
def atomic_save(path,state):
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
def rng_state():return dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
    random.setstate(s['py']);np.random.set_state(s['np']);torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda'])

@torch.no_grad()
def evaluate(model,ds,micro,save=None):
    model.eval();absolute=squared=count=0.; arrays={k:[] for k in ['prediction','target','mask','starts']}
    # MTGNN official graph top-k adds random tie-breaking noise even in eval.
    # Fixed evaluation stream, restored afterwards; no validation/test RNG leaks into training.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(910024);torch.cuda.manual_seed_all(910024)
        for x,y,m,starts,_ in loader(ds):
            pred=torch.cat([model(x[i:i+micro].cuda()).cpu() for i in range(0,len(x),micro)])
            if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite predictions')
            e=(pred.double()-y.double())[m.bool()]
            absolute+=e.abs().sum().item();squared+=e.square().sum().item();count+=e.numel()
            if save:
                for k,v in zip(arrays,[pred,y,m,starts]):arrays[k].append(v.numpy())
    if not count:raise RuntimeError('Empty target mask')
    stats=dict(mae=absolute/count,rmse=(squared/count)**.5,mse=squared/count,valid_points=int(count))
    if save:
        arr={k:np.concatenate(v) for k,v in arrays.items()}
        np.savez_compressed(save,**arr,turbine_ids=ds.data.turbine_ids)
        return stats,arr
    return stats

def plot_loss(history,path):
    fig,ax=plt.subplots(figsize=(7,4));ax.plot([r['epoch'] for r in history],[r['train_mse'] for r in history],label='Train')
    ax.plot([r['epoch'] for r in history],[r['val_mse'] for r in history],label='Validation')
    ax.set(xlabel='Epoch',ylabel='Standardized masked MSE');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(str(path)+'.'+ext,dpi=300)
    plt.close(fig)

def write_metrics(arr,folder):
    pred,y,m=arr['prediction'].astype('float64'),arr['target'].astype('float64'),arr['mask'].astype(bool)
    err=pred-y; rows=[]
    for h in range(48):
        e=err[:,h][m[:,h]];rows.append(dict(horizon=h+1,mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.square(e).mean())),valid_points=len(e)))
    csvwrite(folder/'horizon_metrics.csv',rows)
    rows=[]
    for i,s in enumerate(arr['starts']):
        e=err[i][m[i]];rows.append(dict(start=int(s),valid_points=len(e),mae=float(np.abs(e).mean()) if len(e) else '',sse=float(np.square(e).sum()),sae=float(np.abs(e).sum())))
    csvwrite(folder/'per_window.csv',rows)
    # First valid window and first turbine with valid labels, deterministic selection.
    good=np.argwhere(m.any(axis=1));i,n=map(int,good[0])
    fig,ax=plt.subplots(figsize=(8,3));ax.plot(np.where(m[i,:,n],y[i,:,n],np.nan),label='Target');ax.plot(pred[i,:,n],label='Prediction')
    ax.set(xlabel='Horizon',ylabel='Standardized power',title=f'Forecast start={arr["starts"][i]}, turbine index={n}');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'prediction_example.{ext}',dpi=300)
    plt.close(fig)

def preflight():
    cuda_setup();PAPER.mkdir(parents=True,exist_ok=True)
    data=load_data();train,val=split_data(data,'noise')
    np.savez_compressed(PAPER/'indices_and_scaling.npz',train_starts=train.starts,val_starts=val.starts,test_starts=np.arange(16000,19953,24),
        mean=data.feature_mean,std=data.feature_std,median=data.feature_median,turbine_ids=data.turbine_ids,feature_names=data.feature_names,timestamps=data.timestamps)
    dump(PAPER/'protocol.json',dict(CFG,data_sha256=sha(DATA),gpu=torch.cuda.get_device_name(),torch=torch.__version__))
    source={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if 'runs' not in p.parts}
    dump(PAPER/'source_hashes.json',source)
    x,y,m,_,_=next(iter(loader(train)))
    results=[]
    for name in NAMES:
        passed=False
        for micro in [32,16,8,4,2,1]:
            seed_all(2024);model=build(name).cuda();model.train()
            try:
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                pred=model(x[:micro].cuda());mask=m[:micro].cuda().bool()
                assert pred.shape==(micro,48,134)
                loss=(pred-y[:micro].cuda())[mask].square().mean();loss.backward();torch.cuda.synchronize()
                assert torch.isfinite(loss)
                for p in model.parameters():
                    if p.grad is not None:assert torch.isfinite(p.grad).all()
                backward_seconds=time.perf_counter()-start
                reachable=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters() if p.grad is not None)
                unused=[n for n,p in model.named_parameters() if p.grad is None]
                model.zero_grad(set_to_none=True);model.eval()
                times=[]
                with torch.inference_mode():
                    for j in range(8):
                        torch.cuda.synchronize();t=time.perf_counter();model(x[:micro].cuda());torch.cuda.synchronize()
                        if j>=3:times.append(time.perf_counter()-t)
                row=dict(model=name,**parameter_counts(model),autograd_reachable_real_scalars=reachable,micro_batch=micro,one_forward_backward_seconds=backward_seconds,
                    exploratory_inference_seconds=float(np.median(times)),smoke_peak_memory_bytes=torch.cuda.max_memory_allocated(),status='passed')
                # Batch8 probes are engineering smoke measurements, not final speed comparisons.
                results.append(row);dump(PAPER/f'architecture_{name}.json',dict(architecture=str(model),frequency_indices=frequency_indices(model),parameters_without_gradient_in_probe=unused))
                print(row,flush=True);passed=True
            except torch.cuda.OutOfMemoryError:
                print(name,'OOM at microbatch',micro,flush=True)
            finally:
                del model;torch.cuda.empty_cache()
            if passed:break
        if not passed:raise RuntimeError('Cannot fit '+name)
        csvwrite(PAPER/'preflight.csv',results)
    dump(PAPER/'preflight.json',results)

def train_one(name,strategy,seed):
    cuda_setup();seed_all(seed)
    micro=next(r['micro_batch'] for r in json.loads((PAPER/'preflight.json').read_text()) if r['model']==name)
    run=WORK/name/strategy/str(seed);out=PAPER/name/strategy/str(seed)
    run.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    data=load_data();train,val=split_data(data,strategy)
    model=build(name).cuda();opt=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    initial_indices=frequency_indices(model)
    dump(out/'config.json',dict(CFG,model=name,strategy=strategy,seed=seed,micro_batch=micro,parameters=parameter_counts(model),frequency_indices=initial_indices))
    history=[];best=float('inf');stale=0;start_epoch=1
    if (run/'last.pt').exists():
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        assert state['frequency_indices']==initial_indices
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
        history=state['history'];best=state['best'];stale=state['stale'];start_epoch=state['epoch']+1;restore_rng(state['rng'])
    for epoch in range(start_epoch,31):
        if stale>=5:break
        model.train();train.epoch=epoch;total=count=0.;begin=time.perf_counter()
        for step,(x,y,m,_,_) in enumerate(loader(train,True,seed),1):
            denom=m.sum().item()
            if not denom:continue
            opt.zero_grad(set_to_none=True)
            for i in range(0,len(x),micro):
                target=y[i:i+micro].cuda();mask=m[i:i+micro].cuda().bool()
                pred=model(x[i:i+micro].cuda());err=(pred-target)[mask]
                loss=err.square().sum()/denom
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite train loss')
                loss.backward();total+=err.detach().square().sum().item();count+=err.numel()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step==1 or step%10==0:
                dump(PAPER/'current_training.json',dict(model=name,strategy=strategy,seed=seed,epoch=epoch,step=step,steps=109,time=time.time()))
        vm=evaluate(model,val,micro)
        record=dict(epoch=epoch,train_mse=total/count,val_mse=vm['mse'],val_mae=vm['mae'],val_rmse=vm['rmse'],seconds=time.perf_counter()-begin)
        history.append(record);csvwrite(out/'history.csv',history);print(name,strategy,seed,record,flush=True)
        if vm['mse']<best:best=vm['mse'];stale=0;atomic_save(run/'best.pt',model.state_dict())
        else:stale+=1
        atomic_save(run/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=history,best=best,stale=stale,rng=rng_state(),frequency_indices=initial_indices))
        plot_loss(history,out/'loss')
    model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True),strict=True)
    dump(out/'checkpoint.json',dict(sha256=sha(run/'best.pt'),best_val_mse=best,epochs_run=len(history),frequency_indices=initial_indices))
    for cond,severity in [('clean',None),('corrupted',2)]:
        folder=out/cond;folder.mkdir(exist_ok=True)
        ds=Windows(data,16000,20000,910024,False,24,severity)
        metrics,arr=evaluate(model,ds,micro,run/f'{cond}_predictions.npz')
        assert metrics['valid_points']==966746
        dump(folder/'metrics.json',dict(model=name,strategy=strategy,seed=seed,condition=cond,**metrics));write_metrics(arr,folder)
    if name=='SGSA':
        adj=torch.softmax(torch.nn.functional.elu(model.node_left@model.node_right),-1).detach().cpu().numpy()
        np.save(out/'static_adjacency.npy',adj)
        fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(adj,cmap='viridis');fig.colorbar(im,ax=ax);ax.set(xlabel='Source turbine',ylabel='Receiving turbine');fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/f'adjacency.{ext}',dpi=300)
        plt.close(fig)
    dump(out/'complete.json',dict(status='complete',time=time.time()))

def report():
    import pandas as pd
    rows=[json.loads(p.read_text()) for p in PAPER.glob('*/*/*/*/metrics.json')]
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(PAPER/'all_metrics.csv',index=False)
    summary=df.groupby(['model','strategy','condition'])[['mae','rmse']].agg(['mean','std','count'])
    summary.to_csv(PAPER/'summary.csv')
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,(s,c) in zip(axes.flat,[(s,c) for s in ['clean','noise'] for c in ['clean','corrupted']]):
        v=df[(df.strategy==s)&(df.condition==c)].groupby('model').mae.agg(['mean','std','count'])
        if len(v):ax.bar(v.index,v['mean'],yerr=v['std'].fillna(0),capsize=3)
        ax.set(title=f'{s} training -> {c} test',ylabel='Standardized MAE');ax.tick_params(axis='x',rotation=30)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(PAPER/f'comparison.{ext}',dpi=300)
    plt.close(fig)
    complete=list(PAPER.glob('*/*/*/complete.json'))
    text='# SGSA unified 30-epoch benchmark\n\nCompleted trainings: '+str(len(complete))+'/36.\n\n'+summary.to_string()
    text+='\n\nMeans with fewer than 3 seeds are provisional. Controlled fixed configurations, no test-driven tuning. Previously inspected development panel. MultiGSA_adapted is not exact original-paper reproduction. LSTM/Transformer/FEDformer share weights across turbines; MTGNN/SGSA/MultiGSA_adapted aggregate turbines. No additional calendar inputs. Complex FEDformer parameters counted both as tensor entries and real scalar degrees of freedom. No significance claim from three-seed means alone.\n'
    (PAPER/'report.md').write_text(text)

def queue():
    import fcntl
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Check immutable source manifest before starting/restarting the queue.
    for rel,digest in json.loads((PAPER/'source_hashes.json').read_text()).items():
        assert sha(ROOT/rel)==digest, 'Source changed after preflight: '+rel
    errors=[]
    for strategy in ['clean','noise']:
        for seed in [2024,2025,2026]:
            for name in NAMES:
                out=PAPER/name/strategy/str(seed)
                if (out/'complete.json').exists():continue
                out.mkdir(parents=True,exist_ok=True)
                dump(PAPER/'queue_status.json',dict(stage='running',model=name,strategy=strategy,seed=seed,completed=len(list(PAPER.glob('*/*/*/complete.json'))),total=36,errors=errors))
                log=out/'train.log'
                with log.open('a') as f:
                    rc=subprocess.call([sys.executable,'-u',__file__,'train','--model',name,'--strategy',strategy,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT)
                if rc:
                    error=dict(model=name,strategy=strategy,seed=seed,exit_code=rc)
                    errors.append(error);dump(out/'failure.json',error)
                report()
    dump(PAPER/'queue_status.json',dict(stage='complete' if not errors else 'finished_with_errors',completed=len(list(PAPER.glob('*/*/*/complete.json'))),total=36,errors=errors))
    report()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','train','queue','report']);p.add_argument('--model',choices=NAMES);p.add_argument('--strategy',choices=['clean','noise']);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.action=='preflight':preflight()
    elif a.action=='train':train_one(a.model,a.strategy,a.seed)
    elif a.action=='queue':queue()
    else:report()
