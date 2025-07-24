import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
import torch

class SimpleGINlayer(nn.Module):
    def __init__(self, hidden_dim, eps = 0.0, train_eps=True) -> None:
        super().__init__()
    
        self.eps = nn.Parameter(torch.Tensor([eps])) if train_eps else eps
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index):
        
        row, col = edge_index

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])

        out = (1 + self.eps) * x + agg

        return out


class my_GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers) -> None:
        super().__init__()

        self.lin1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())

        self.layers = nn.ModuleList([
            SimpleGINlayer(hidden_dim, True)
            for _ in range(num_layers)
        ])

        self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)
    
    def forward(self, data):
        x, edge_index_intra, edge_index_inter, pos =\
            data.x, data.edge_index_intra, data.edge_index_inter, data.pos
        
        edge_index = torch.cat([edge_index_intra, edge_index_inter], dim=1)  

        x_lin = self.lin1(x)

        for layer in self.layers:
            x_gin = layer(x_lin, edge_index)

        x_gin = global_add_pool(x_gin, data.batch)

        out = self.fc(x_gin)

        return out.view(-1)     


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