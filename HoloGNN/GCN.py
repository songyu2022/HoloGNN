from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)

        self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)

    def forward(self, data):
        x, edge_index_intra, edge_index_inter = \
            data.x, data.edge_index_intra, data.edge_index_inter
        
        edge_index = torch.cat([edge_index_intra, edge_index_inter], dim=1)
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = self.gcn3(x, edge_index)
        
        x = global_add_pool(x, data.batch)
        
        x = self.fc(x)

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