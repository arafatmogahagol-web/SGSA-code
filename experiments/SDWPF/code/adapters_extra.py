"""Author-code adapters. Additional graph priors use the training partition only."""
from pathlib import Path
from types import ModuleType
import importlib.util, sys
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
VENDOR=Path(__file__).resolve().parents[3]/'vendor'
NAMES=['PDFormer_train_graph','StemGNN_power_only']

def module(name,path,package=False):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(Path(path).parent)] if package else None)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def priors():
    with np.load(ROOT/'train_priors.npz') as a:return {k:a[k] for k in a.files}

class PDFormer(nn.Module):
    def __init__(self):
        super().__init__()
        # Import only base classes/loss and the selected architecture, avoiding
        # libcity's global model registry and its unrelated optional packages.
        pkg=ModuleType('libcity');pkg.__path__=[str(VENDOR/'PDFormer/libcity')];sys.modules['libcity']=pkg
        pkg=ModuleType('libcity.model');pkg.__path__=[str(VENDOR/'PDFormer/libcity/model')];sys.modules['libcity.model']=pkg
        path=VENDOR/'PDFormer/libcity/model/traffic_flow_prediction/PDFormer.py'
        source=path.read_text()
        # Author loss registry imports sklearn for unused R² functions. The
        # benchmark uses external masked MSE; no vendor loss is ever called.
        assert source.count('from libcity.model import loss')==1
        source=source.replace('from libcity.model import loss','loss = None  # external benchmark masked MSE only')
        official=ModuleType('official_pdformer');exec(compile(source,str(path),'exec'),official.__dict__)
        p=priors()
        cfg=dict(dataset='SDWPF_train_graph',embed_dim=64,skip_dim=256,lape_dim=8,
            geo_num_heads=4,sem_num_heads=2,t_num_heads=2,enc_depth=6,mlp_ratio=4,
            drop=.1,attn_drop=.1,drop_path=.1,input_window=96,output_window=48,
            output_dim=1,add_time_in_day=False,add_day_in_week=False,device=torch.device('cpu'),
            use_curriculum_learning=False,max_epoch=30,far_mask_delta=3,dtw_delta=5)
        features=dict(num_nodes=134,feature_dim=16,ext_dim=0,num_batches=109,scaler=None,
            dtw_matrix=p['dtw'],adj_mx=p['adjacency'],sd_mx=1-p['correlation'],sh_mx=p['hops'],pattern_keys=p['pattern_keys'])
        self.net=official.PDFormer(cfg,features)
        for key in ['geo_mask','sem_mask','pattern_keys']:
            value=getattr(self.net,key);delattr(self.net,key);self.net.register_buffer(key,value)
        self.register_buffer('lap_pe',torch.from_numpy(p['lap_pe']))
    def forward(self,x):
        # Pattern extractor uses its first channel as the target power.
        x=torch.cat([x[...,-1:],x[...,:-1]],dim=-1)
        return self.net({'X':x},self.lap_pe).squeeze(-1)

class StemGNN(nn.Module):
    def __init__(self):
        super().__init__()
        path=VENDOR/'StemGNN/models/base_model.py'
        source=path.read_text()
        # Exact replacement of the removed two-sided real FFT APIs; preserves
        # both real and imaginary components with the original FFT normalization.
        old='torch.rfft(input, 1, onesided=False)'
        new='torch.view_as_real(torch.fft.fft(input, dim=-1, norm="backward"))'
        assert source.count(old)==1;source=source.replace(old,new)
        old='torch.irfft(time_step_as_inner, 1, onesided=False)'
        new='torch.fft.ifft(torch.view_as_complex(time_step_as_inner.contiguous()), dim=-1, norm="backward").real'
        assert source.count(old)==1;source=source.replace(old,new)
        official=ModuleType('official_stemgnn_compat');exec(compile(source,str(path),'exec'),official.__dict__)
        self.net=official.Model(units=134,stack_cnt=2,time_step=96,multi_layer=5,horizon=48,dropout_rate=.1,device='cpu')
    def forward(self,x):return self.net(x[...,-1])[0]

def build(name):
    return {'PDFormer_train_graph':PDFormer,'StemGNN_power_only':StemGNN}[name]()

def parameter_counts(model):
    p=list(model.parameters())
    return dict(tensor_elements=sum(v.numel() for v in p),real_scalar_parameters=sum(v.numel()*(2 if v.is_complex() else 1) for v in p),
        parameter_bytes=sum(v.numel()*v.element_size() for v in p),trainable_real_scalars=sum(v.numel()*(2 if v.is_complex() else 1) for v in p if v.requires_grad))

def frequency_indices(model):return {}
