"""Author implementations with explicit SDWPF adapters; no test-based selection.

Vendor source is immutable. See adapter_notes.md for all deviations.
"""
from pathlib import Path
from types import SimpleNamespace, ModuleType
import importlib.util
import sys
import torch
from torch import nn

VENDOR=Path(__file__).resolve().parents[3]/'vendor'
NAMES=['GraphWaveNet','AGCRN','STID_no_calendar','DLinear','PatchTST']


def module(name,path,package=False):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(Path(path).parent)] if package else None)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod);return mod

class GraphWaveNet(nn.Module):
    def __init__(self):
        super().__init__()
        # The official 2019 file uses Conv1d with a two-dimensional kernel and
        # four-dimensional input. Conv2d is the intended operation on current torch.
        path=VENDOR/'GraphWaveNet/model.py'
        source=path.read_text();assert source.count('nn.Conv1d(')==3
        source=source.replace('nn.Conv1d(','nn.Conv2d(')
        official=ModuleType('official_graphwavenet_compat')
        exec(compile(source,str(path),'exec'),official.__dict__)
        self.net=official.gwnet(torch.device('cpu'),134,dropout=.1,supports=None,
            gcn_bool=True,addaptadj=True,in_dim=16,out_dim=48,residual_channels=64,
            dilation_channels=64,skip_channels=64,end_channels=128,kernel_size=2,blocks=4,layers=5)
        assert self.net.receptive_field>=96
    def forward(self,x):
        y=self.net(x.permute(0,3,2,1))
        assert y.shape[-1]==1,y.shape
        return y.squeeze(-1)

class AGCRN(nn.Module):
    def __init__(self):
        super().__init__()
        # Original imports are model.AGCRNCell / model.AGCN.
        pkg=ModuleType('model');pkg.__path__=[str(VENDOR/'AGCRN/model')]
        sys.modules['model']=pkg
        official=module('official_agcrn',VENDOR/'AGCRN/model/AGCRN.py')
        cfg=SimpleNamespace(num_nodes=134,input_dim=16,rnn_units=64,output_dim=1,
            horizon=48,num_layers=2,default_graph=True,embed_dim=10,cheb_k=2)
        self.net=official.AGCRN(cfg)
        # The author's Run.py initializes all parameters, including empty pools.
        for p in self.net.parameters():
            if p.dim()>1:nn.init.xavier_uniform_(p)
            else:nn.init.uniform_(p)
    def forward(self,x):return self.net(x,None,teacher_forcing_ratio=0.).squeeze(-1)

class STID(nn.Module):
    def __init__(self):
        super().__init__()
        pkg=module('official_stid_arch',VENDOR/'STID/stid/arch/__init__.py',True)
        arch=module('official_stid_arch.stid_arch',VENDOR/'STID/stid/arch/stid_arch.py')
        self.net=arch.STID(num_nodes=134,node_dim=32,input_len=96,input_dim=16,embed_dim=64,
            output_len=48,num_layer=3,temp_dim_tid=32,temp_dim_diw=32,time_of_day_size=48,
            day_of_week_size=7,if_T_i_D=False,if_D_i_W=False,if_node=True)
        for m in self.net.modules():
            if isinstance(m,nn.Dropout):m.p=.1
    def forward(self,x):return self.net(x,None,0,0,self.training).squeeze(-1)

class DLinear(nn.Module):
    def __init__(self):
        super().__init__()
        official=module('official_dlinear',VENDOR/'DLinear/models/DLinear.py')
        self.net=official.Model(SimpleNamespace(seq_len=96,pred_len=48,individual=False,enc_in=16))
    def forward(self,x):
        b,t,n,c=x.shape
        z=x.permute(0,2,1,3).reshape(b*n,t,c)
        return self.net(z)[...,-1].reshape(b,n,48).permute(0,2,1)

class PatchTST(nn.Module):
    def __init__(self):
        super().__init__()
        sys.path.insert(0,str(VENDOR/'PatchTST/PatchTST_supervised'))
        official=module('official_patchtst',VENDOR/'PatchTST/PatchTST_supervised/models/PatchTST.py')
        cfg=SimpleNamespace(enc_in=16,seq_len=96,pred_len=48,e_layers=2,n_heads=4,d_model=64,d_ff=128,
            dropout=.1,fc_dropout=.1,head_dropout=.1,individual=False,patch_len=16,stride=8,
            padding_patch='end',revin=True,affine=True,subtract_last=False,decomposition=False,kernel_size=25)
        self.net=official.Model(cfg,attn_dropout=.1)
    def forward(self,x):
        b,t,n,c=x.shape
        z=x.permute(0,2,1,3).reshape(b*n,t,c)
        return self.net(z)[...,-1].reshape(b,n,48).permute(0,2,1)


def build(name):
    cls={'GraphWaveNet':GraphWaveNet,'AGCRN':AGCRN,'STID_no_calendar':STID,
         'DLinear':DLinear,'PatchTST':PatchTST}[name]
    return cls()

def parameter_counts(model):
    params=list(model.parameters())
    return dict(tensor_elements=sum(p.numel() for p in params),
        real_scalar_parameters=sum(p.numel()*(2 if p.is_complex() else 1) for p in params),
        parameter_bytes=sum(p.numel()*p.element_size() for p in params),
        trainable_real_scalars=sum(p.numel()*(2 if p.is_complex() else 1) for p in params if p.requires_grad))

def frequency_indices(model):return {}
