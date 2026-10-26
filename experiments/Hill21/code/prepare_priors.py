"""Training-only priors. No validation/test arrays used to estimate graph/patterns."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.sparse.csgraph import shortest_path


ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'panel.npz'

def main():
    data=np.load(DATA)
    boundary=int(json.loads((DATA.parents[1]/'data_audit.json').read_text())['boundaries'][1])
    power=data['x'][:boundary,:,-1].astype('float64')
    # Historical missing values already use training medians, as in benchmark.
    corr=np.nan_to_num(np.corrcoef(power.T),nan=0.)
    np.fill_diagonal(corr,1.)
    affinity=np.abs(corr);np.fill_diagonal(affinity,0.)
    adjacency=np.zeros((21,21))
    for i in range(21):
        ind=np.argsort(-affinity[i],kind='stable')[:10];adjacency[i,ind]=affinity[i,ind]
    adjacency=np.maximum(adjacency,adjacency.T)
    transition=adjacency+np.eye(21);transition/=transition.sum(axis=1,keepdims=True)
    hops=shortest_path((adjacency>0).astype(float),directed=False,unweighted=True)
    deg=adjacency.sum(1);inv=1/np.sqrt(np.maximum(deg,1e-12))
    lap=np.eye(21)-inv[:,None]*adjacency*inv[None,:]
    _,vec=np.linalg.eigh(lap);pe=vec[:,1:9]
    for j in range(8):
        if pe[np.argmax(np.abs(pe[:,j])),j]<0:pe[:,j]*=-1
    # Exact DTW on mean 48-slot profiles computed only from the training slice.
    profile=np.stack([power[k::48].mean(0) for k in range(48)],axis=1)
    dtw=np.zeros((21,21))
    for i in range(21):
        for j in range(i):
            d=np.full((49,49),np.inf);d[0,0]=0
            for a in range(1,49):
                for b in range(1,49):
                    d[a,b]=(profile[i,a-1]-profile[j,b-1])**2+min(d[a-1,b],d[a,b-1],d[a-1,b-1])
            dtw[i,j]=dtw[j,i]=np.sqrt(d[-1,-1])
    # Three-step power patterns; predeclared k=16, seed=2024, 30 Lloyd steps.
    patterns=np.stack([power[:-2],power[1:-1],power[2:]],axis=-1).reshape(-1,3)
    rng=np.random.default_rng(2024);sample=patterns[rng.choice(len(patterns),min(20000,len(patterns)),replace=False)]
    centers=sample[rng.choice(len(sample),16,replace=False)].copy()
    for _ in range(30):
        assignment=((sample[:,None]-centers[None])**2).sum(-1).argmin(1)
        for k in range(16):
            if (assignment==k).any():centers[k]=sample[assignment==k].mean(0)
    np.savez_compressed(ROOT/'train_priors.npz',transition=transition.astype('float32'),adjacency=adjacency.astype('float32'),
        correlation=corr.astype('float32'),hops=hops.astype('float32'),lap_pe=pe.astype('float32'),dtw=dtw.astype('float32'),pattern_keys=centers[...,None].astype('float32'))
    (ROOT/'prior_audit.json').write_text(json.dumps(dict(source=str(DATA),source_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
        fit_partition=[0,boundary],validation_used=False,test_used=False,graph='abs training power correlation top10 symmetric',
        dtw='exact DTW of training-only mean 48-slot profiles',patterns='3-step training segments, 16 centers, seed2024,30 Lloyd iterations',
        prior_sha256=hashlib.sha256((ROOT/'train_priors.npz').read_bytes()).hexdigest()),indent=2))
    print('Training-only priors saved',flush=True)

if __name__=='__main__':main()
