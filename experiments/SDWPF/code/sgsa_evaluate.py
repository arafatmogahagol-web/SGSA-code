"""Evaluate a converted checkpoint on the frozen historical test protocol."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from model import SGSA
from data_loader import load_and_prepare,Windows,loader

@torch.inference_mode()
def evaluate(model,dataset,save=None):
    model.eval(); absolute=squared=count=0.; arrays={k:[] for k in ['prediction','target','mask','starts']}
    for x,y,m,starts,_ in loader(dataset):
        pred=model(x.cuda()).cpu()
        valid=m.bool(); e=(pred.double()-y.double())[valid]
        absolute+=e.abs().sum().item();squared+=e.square().sum().item();count+=valid.sum().item()
        if save:
            for key,value in zip(arrays,[pred,y,m,starts]):arrays[key].append(value.numpy())
    if not count: raise ValueError('No valid targets')
    if save: np.savez_compressed(save,**{k:np.concatenate(v) for k,v in arrays.items()},turbine_ids=dataset.data.turbine_ids)
    return dict(mae=absolute/count,mse=squared/count,rmse=(squared/count)**.5,valid_points=int(count))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--condition',choices=['clean','corrupted'],default='clean');a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    data=load_and_prepare(a.data,.7,.1)
    if data.x_scaled.shape!=(20000,134,16):raise ValueError('Protocol requires the audited 20000-point array')
    model=SGSA().cuda();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True),strict=True)
    ds=Windows(data,16000,20000,seed=910024,severity=None if a.condition=='clean' else 2)
    result=evaluate(model,ds,out/'predictions.npz')
    (out/'metrics.json').write_text(json.dumps(result,indent=2)); print(result)
