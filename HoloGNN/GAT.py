from torch_geometric.nn import GATConv, global_add_pool
import torch.nn as nn
import torch
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.lin1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())

        self.gat1 = GATConv(hidden_dim, hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.gat3 = GATConv(hidden_dim, hidden_dim)

        self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)

    
    def forward(self, data):
        # 这里的边的数目和节点的数目不是一张图的，而是一个batchsize，128张图的节点数和边的数目
        # x shape is torch.Size([24231, 35])
        # edge_index_intra shape is torch.Size([2, 47802])
        # edge_index_inter shape is torch.Size([2, 75530])
        # after gat1 shape is torch.Size([24231, 256])
        # after global_add_pool shape is torch.Size([128, 256])
        # after fc x shape is torch.Size([128, 1])
        x, edge_index_intra, edge_index_inter = \
            data.x, data.edge_index_intra, data.edge_index_inter
        
        # print(f'x shape {x.shape}')
        # print(f'edge_index_intra shape {edge_index_intra.shape}')
        # print(f'edge_index_inter shape {edge_index_inter.shape}')
        
        edge_index = torch.cat([edge_index_intra, edge_index_inter], dim=1)
        #print(f'edge_index shape {edge_index.shape}')
        x_gnn = self.lin1(x)
        x_gnn = self.gat1(x_gnn, edge_index)
        #print(f'after gat1 x shape {x.shape}')
        x_gnn = F.relu(x_gnn)
        x_gnn = self.gat2(x_gnn, edge_index)
        x_gnn = F.relu(x_gnn)
        x_gnn = self.gat3(x_gnn, edge_index)
        #print(f'after gat3 x shape {x.shape}')

        x_gnn = global_add_pool(x_gnn, data.batch)



        #print(f'after global_add_pool x shape {x.shape}')

        x = self.fc(x_gnn)
        #print(f'after fc x shape {x.shape}')

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
    



    


