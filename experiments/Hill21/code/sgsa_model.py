"""SGSA: static turbine graph, scaled self-feature residual, temporal SimAM.

Input [B,96,134,16]; output [B,48,134]. No Transformer or dynamic graph.
State names preserve the active historical T0 checkpoint keys.
"""
import torch
from torch import nn
from torch.nn import functional as F
from layers import SimAM1D

class SGSA(nn.Module):
    def __init__(self, seq_len=96, num_nodes=134, input_dim=16,
                 pred_len=48, d_model=64, dropout=0.1, graph_rank=16, rho=0.25):
        super().__init__()
        self.seq_len, self.num_nodes = seq_len, num_nodes
        self.input_dim, self.d_model, self.rho = input_dim, d_model, rho
        self.input_proj = nn.Linear(input_dim, d_model)
        self.node_left = nn.Parameter(torch.empty(num_nodes, graph_rank))
        self.node_right = nn.Parameter(torch.empty(graph_rank, num_nodes))
        nn.init.xavier_uniform_(self.node_left)
        nn.init.xavier_uniform_(self.node_right)
        self.spatial_proj = nn.Linear(d_model, d_model)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.position = nn.Parameter(torch.zeros(1, seq_len, 1, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.filter = SimAM1D()
        self.forecast = nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),
                                     nn.Dropout(dropout),nn.Linear(d_model,pred_len))
        self.latest_adj = None

    def forward(self, x):
        b, length, nodes, features = x.shape
        if (length,nodes,features) != (self.seq_len,self.num_nodes,self.input_dim):
            raise ValueError('Unexpected input dimensions')
        h = self.input_proj(x)
        s = torch.softmax(F.elu(self.node_left @ self.node_right), dim=-1)
        a = s[None].expand(b,-1,-1)
        self.latest_adj = a.detach()
        spatial = torch.einsum('bij,btjd->btid',a,h)
        h = self.spatial_norm(self.rho*h+self.dropout(F.gelu(self.spatial_proj(spatial))))
        z = (h+self.position).permute(0,2,1,3).reshape(b*nodes,length,self.d_model)
        z = self.temporal_norm(z)
        h = z.reshape(b,nodes,length,self.d_model).permute(0,2,1,3)
        return self.forecast(self.filter(h)[:,-1]).permute(0,2,1)

def load_historical_checkpoint(model, state):
    """Drop only the two proven-unused Q/K matrices; all active keys load strictly."""
    unused = {'query.weight', 'key.weight'}
    extra = set(state)-set(model.state_dict())
    if extra != unused:
        raise ValueError(f'Unexpected historical checkpoint keys: {extra}')
    model.load_state_dict({k:v for k,v in state.items() if k not in unused}, strict=True)
    return sorted(unused)
