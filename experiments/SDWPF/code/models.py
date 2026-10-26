"""Six predeclared methods. Official MTGNN/FEDformer sources remain unchanged."""
import importlib.util, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from sgsa import SGSA
from multigsa_adapted import MultiGSAAdapted

VENDOR=Path(__file__).resolve().parents[3]/'vendor'
NAMES=['SGSA','MultiGSA_adapted','LSTM','Transformer','MTGNN','FEDformer']

class Temporal(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.proj=nn.Linear(16,64)
        if kind=='LSTM':self.encoder=nn.LSTM(64,64,2,batch_first=True,dropout=.1)
        else:
            self.position=nn.Parameter(torch.zeros(1,96,64));nn.init.normal_(self.position,std=.02)
            layer=nn.TransformerEncoderLayer(64,4,128,.1,activation='gelu',batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(64)
        self.head=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,48))
    def forward(self,x):
        b,t,n,f=x.shape;z=self.proj(x.permute(0,2,1,3).reshape(b*n,t,f))
        z=self.encoder(z)[0] if self.kind=='LSTM' else self.encoder(z+self.position)
        return self.head(self.norm(z[:,-1])).reshape(b,n,48).permute(0,2,1)

class MTGNNAdapter(nn.Module):
    def __init__(self):
        super().__init__();sys.path.insert(0,str(VENDOR/'MTGNN'))
        from net import gtnet
        self.net=gtnet(True,True,2,134,torch.device('cuda'),dropout=.1,subgraph_size=20,node_dim=40,
            dilation_exponential=2,conv_channels=64,residual_channels=64,skip_channels=64,end_channels=128,
            seq_length=96,in_dim=16,out_dim=48,layers=3,propalpha=.05,tanhalpha=3,layer_norm_affline=True)
    def forward(self,x):return self.net(x.permute(0,3,2,1)).squeeze(-1)

class FEDformerAdapter(nn.Module):
    def __init__(self):
        super().__init__();sys.path.insert(0,str(VENDOR/'FEDformer'))
        spec=importlib.util.spec_from_file_location('official_fedformer',VENDOR/'FEDformer/models/FEDformer.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        # Eight heads required by official FourierBlock (hard-coded internal head count).
        cfg=SimpleNamespace(version='Fourier',mode_select='random',modes=32,seq_len=96,label_len=48,pred_len=48,
            output_attention=False,moving_avg=25,enc_in=16,dec_in=16,c_out=16,d_model=64,embed='timeF',freq='h',
            dropout=.1,n_heads=8,d_ff=128,e_layers=2,d_layers=1,activation='gelu')
        self.net=mod.Model(cfg)
    def forward(self,x):
        b,t,n,f=x.shape;z=x.permute(0,2,1,3).reshape(b*n,t,f)
        # No extra calendar covariates for any model. All time-mark channels are zero.
        mark=z.new_zeros((b*n,96,4))
        dec=torch.cat([z[:,-48:],z.new_zeros((b*n,48,16))],dim=1)
        y=self.net(z,mark,dec,mark)
        return y[:,:,-1].reshape(b,n,48).permute(0,2,1)

def build(name):
    if name=='SGSA':return SGSA()
    if name=='MultiGSA_adapted':return MultiGSAAdapted(96,134,16,48,64,.1)
    if name in ['LSTM','Transformer']:return Temporal(name)
    if name=='MTGNN':return MTGNNAdapter()
    if name=='FEDformer':return FEDformerAdapter()
    raise ValueError(name)

def parameter_counts(model):
    # Complex coefficients need two real scalars; report both conventions.
    params=list(model.parameters())
    return dict(tensor_elements=sum(p.numel() for p in params),
                real_scalar_parameters=sum(p.numel()*(2 if p.is_complex() else 1) for p in params),
                parameter_bytes=sum(p.numel()*p.element_size() for p in params),
                trainable_real_scalars=sum(p.numel()*(2 if p.is_complex() else 1) for p in params if p.requires_grad))

def frequency_indices(model):
    return {name:{attr:getattr(m,attr) for attr in ['index','index_q','index_kv'] if hasattr(m,attr)}
            for name,m in model.named_modules() if any(hasattr(m,a) for a in ['index','index_q','index_kv'])}
