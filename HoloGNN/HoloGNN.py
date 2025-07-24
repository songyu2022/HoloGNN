# %%
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from HIL import HIL
import torch


class HoloGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.lin1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.lin2 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())

        self.gconv1 = HIL(hidden_dim, hidden_dim)
        self.gconv2 = HIL(hidden_dim, hidden_dim)
        self.gconv3 = HIL(hidden_dim, hidden_dim)

        self.atom_attention = atom_Attention(hidden_dim)
        self.fc = FC(hidden_dim * 2, hidden_dim, 3, 0.1, 1)
    
    def forward(self, data):

        x, edge_index_intra, edge_index_inter, pos =\
            data.x, data.edge_index_intra, data.edge_index_inter, data.pos

        x_gnn = self.lin1(x)
        x_gnn = self.gconv1(x_gnn, edge_index_intra, edge_index_inter, pos)
        x_gnn = self.gconv2(x_gnn, edge_index_intra, edge_index_inter, pos)
        x_gnn = self.gconv3(x_gnn, edge_index_intra, edge_index_inter, pos)

        x_gnn = global_add_pool(x_gnn, data.batch)

        x_atte = self.lin2(x)
        x_atte = self.atom_attention(x_atte)
        x_atte = global_add_pool(x_atte, data.batch)
        x_combine = torch.cat([x_gnn, x_atte], dim=-1)
        
        x = self.fc(x_combine)

        return x.view(-1)
        


class FC(nn.Module):
    def __init__(self, graph_dim, hidden_dim, n_layers, dropout, n_tasks):
        super().__init__()
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_tasks = n_tasks
        self.predict = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                self.predict.append(nn.Linear(graph_dim, hidden_dim))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(hidden_dim))
            elif i == n_layers - 1:
                self.predict.append(nn.Linear(hidden_dim, n_tasks))
            else:
                self.predict.append(nn.Linear(hidden_dim, hidden_dim))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
# heterogeneous interaction layer
class HIL(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__(aggr='add')
        '''
        in_channels: 256
        out_channels: 256
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)    
        )
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

        self.mlp_coord_cov = nn.Sequential(nn.Linear(9, in_channels), nn.SiLU())
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, in_channels), nn.SiLU())

        self.phi_coord = nn.Linear(out_channels, 1)
        self.out = nn.Sequential(                 
            nn.Linear(out_channels, out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_channels)
        )    

    def forward(self, x, edge_index_intra, edge_index_inter, pos=None,
                size=None):


        row_cov, col_cov = edge_index_intra
        coord_diff_cov = pos[row_cov] - pos[col_cov]

        nor_diff = torch.norm(coord_diff_cov, dim=-1)
        rbf1 = _rbf(nor_diff, D_min=0., D_max=6., D_count=9, device=x.device)
        radial_cov = self.mlp_coord_cov(rbf1)
        out_node_intra = self.propagate(edge_index=edge_index_intra, x=x, radial=radial_cov,coord_diff=coord_diff_cov,size=size)

        row_ncov, col_ncov = edge_index_inter
        coord_diff_ncov = pos[row_ncov] - pos[col_ncov]
        rbf2 = _rbf(torch.norm(coord_diff_ncov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device)
        radial_ncov = self.mlp_coord_ncov(rbf2)
        out_node_inter = self.propagate(edge_index=edge_index_inter, x=x, radial=radial_ncov,coord_diff = coord_diff_ncov,size=size)
        out_node = self.mlp_node_cov(x + out_node_intra) + self.mlp_node_ncov(x + out_node_inter)

        return out_node 

    def message(self, x_j: Tensor, x_i: Tensor, radial, coord_diff, index):
        m = x_j * radial

        return m
    


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

class atom_Attention(nn.Module):
    def __init__(self, embed_dim):
        super(atom_Attention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.query(x)  
        K = self.key(x)  
        V = self.value(x)  

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  
        attn_scores = attn_scores / (Q.size(-1) ** 0.5)  
        attn_weights = F.softmax(attn_scores, dim=-1)  
        output = torch.matmul(attn_weights, V)  

        return output + x
        



























# %%
import torch
import torch.nn.functional as F

# class atom_Attention(nn.Module):
#     def __init__(self, embed_dim):
#         super(atom_Attention, self).__init__()
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)

#     def forward(self, x):
#         Q = self.query(x)  # 查询
#         K = self.key(x)  # 键
#         V = self.value(x)  # 值

#         # 计算自注意力
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # 点积 Q 和 K 的转置
#         attn_scores = attn_scores / (Q.size(-1) ** 0.5)  # 缩放
#         attn_weights = F.softmax(attn_scores, dim=-1)  # softmax 归一化
#         output = torch.matmul(attn_weights, V)  # 加权求和
#         # print(f'after attention output is {output.shape}')

#         # 输出是加权后的特征
#         return output + x