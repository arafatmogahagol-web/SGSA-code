"""Historical chronological windows and deterministic input-only corruption."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import load_and_prepare

class Windows(Dataset):
    def __init__(self,data,start,end,seed=2024,augmentation=False,stride=24,severity=None):
        if start<96 or end>len(data.timestamps):
            raise ValueError('History/target crosses data bounds')
        self.data=data
        self.starts=list(range(start,end-48+1,stride))
        self.seed,self.augmentation,self.severity,self.epoch=seed,augmentation,severity,0
        self.median=((data.feature_median-data.feature_mean)/data.feature_std).astype('float32')
    def __len__(self): return len(self.starts)
    def __getitem__(self,index):
        s=self.starts[index]; x=self.data.x_scaled[s-96:s].copy()
        rng=np.random.default_rng(self.seed+1000003*self.epoch+104729*index)
        level=self.severity if self.severity is not None else int(rng.choice(3,p=[.5,.3,.2])) if self.augmentation else 0
        if level:
            packets=rng.random(x.shape[:2])<(.05 if level==1 else .2)
            x[packets]=self.median
            if level==2:
                finite=np.isfinite(self.data.values[s-96:s])
                outliers=(rng.random(x.shape)<.05)&finite&~packets[...,None]
                x[outliers]=rng.choice([-8.,8.],size=int(outliers.sum()))
        return torch.from_numpy(x),torch.from_numpy(self.data.y_scaled[s:s+48]),torch.from_numpy(self.data.y_mask[s:s+48]),s,level

def loader(ds,shuffle=False,seed=2024):
    return DataLoader(ds,batch_size=32,shuffle=shuffle,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(seed))
