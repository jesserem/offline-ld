import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


def aggr_obs(obs_mb, n_node):
    # obs_mb is [m, n_nodes_each_state, fea_dim], m is number of nodes in batch
    idxs = obs_mb.coalesce().indices()
    vals = obs_mb.coalesce().values()
    new_idx_row = idxs[1] + idxs[0] * n_node
    new_idx_col = idxs[2] + idxs[0] * n_node
    idx_mb = torch.stack((new_idx_row, new_idx_col))
    # print(idx_mb)
    # print(obs_mb.shape[0])
    adj_batch = torch.sparse.FloatTensor(indices=idx_mb,
                                         values=vals,
                                         size=torch.Size([obs_mb.shape[0] * n_node,
                                                          obs_mb.shape[0] * n_node]),
                                         ).to(obs_mb.device)

    # print('aggr_obs', adj_batch.shape)
    return adj_batch


def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    # batch_size is the shape of batch
    # for graph pool sparse matrix
    if graph_pool_type == 'average':
        elem = torch.full(size=(batch_size[0]*n_nodes, 1),
                          fill_value=1 / n_nodes,
                          dtype=torch.float32,
                          device=device).view(-1)
    else:
        elem = torch.full(size=(batch_size[0] * n_nodes, 1),
                          fill_value=1,
                          dtype=torch.float32,
                          device=device).view(-1)
    idx_0 = torch.arange(start=0, end=batch_size[0],
                         device=device,
                         dtype=torch.long)
    # print(idx_0)
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size[0]*n_nodes, 1)).squeeze()

    idx_1 = torch.arange(start=0, end=n_nodes*batch_size[0],
                         device=device,
                         dtype=torch.long)
    idx = torch.stack((idx_0, idx_1))
    graph_pool = torch.sparse.FloatTensor(idx, elem,
                                          torch.Size([batch_size[0],
                                                      n_nodes*batch_size[0]])
                                          ).to(device)

    return graph_pool


class GraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 # final_dropout,
                 learn_eps,
                 neighbor_pooling_type,
                 device,
                 dropout=0.0,
                 activation=nn.LeakyReLU):
        """
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        device: which device to use
        """

        super(GraphCNN, self).__init__()

        # self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.dropout = dropout
        # common out the eps if you do not need to use it, otherwise the it will cause
        # error "not in the computational graph"
        # if self.learn_eps:
        #     self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        # List of MLPs
        self.mlps = torch.nn.ModuleList()
        self.activation_f = activation()

        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.p_dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.p_dropout)

        for layer in range(self.num_layers-1):
            if layer == 0:
                curr_mlps = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim, dropout=0, activation=activation)
                # curr_mlps.apply(lambda m: init_module_weights(m, True))

                self.mlps.append(curr_mlps)
            else:
                curr_mlps = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim, dropout=0, activation=activation)
                # curr_mlps.apply(lambda m: init_module_weights(m, True))
                self.mlps.append(curr_mlps)

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(self.dropout))
        # init_module_weights(self.mlps[-1])

    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.
        # h = self.dropouts[layer](h)

        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.mm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        # pooled_rep = self.mlps[layer](pooled)
        pooled_rep = self.mlps[layer](pooled)
        pooled_rep = self.batch_norms[layer](pooled_rep)
        pooled_rep = self.activation_f(pooled_rep)
        h = self.dropouts[layer](pooled_rep)


        # non-linearity

        return h

    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        # h = self.dropouts[layer](h)

        # pooling neighboring nodes and center nodes altogether
        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            # print(Adj_block.dtype)
            # print(h.dtype)
            pooled = torch.mm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        pooled_rep = self.batch_norms[layer](pooled_rep)
        pooled_rep = self.activation_f(pooled_rep)
        pooled_rep = self.dropouts[layer](pooled_rep)


        # non-linearity

        return pooled_rep

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj):

        x_concat = x
        graph_pool = graph_pool

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = padded_nei
        else:
            Adj_block = adj

        # list of hidden representation at each layer (including input)
        h = x_concat

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)
            # if layer != self.num_layers-1:
            # h = self.dropouts[layer](h)
        # h = F.dropout(h, self.p_dropout, training=self.training)

        h_nodes = h.clone()
        # print(graph_pool.shape, h.shape)
        pooled_h = torch.sparse.mm(graph_pool, h)
        # pooled_h = graph_pool.spmm(h)
        # h_nodes = F.dropout(h_nodes, self.p_dropout, training=self.training)
        # pooled_h = F.dropout(pooled_h, self.p_dropout, training=self.training)

        return pooled_h, h_nodes

    def init_weights(self):
        for mlp in self.mlps:
            mlp.init_weights()



class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0.0, activation=nn.LeakyReLU):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers


        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.net = nn.Linear(input_dim, output_dim)
        else:
            net = torch.nn.ModuleList()
            # Multi-layer model
            self.linear_or_not = False
            # self.linears = torch.nn.ModuleList()
            # self.batch_norms = torch.nn.ModuleList()
            # self.dropouts = torch.nn.ModuleList()
            net.append(nn.Linear(input_dim, hidden_dim))
            net.append(nn.BatchNorm1d(hidden_dim))
            net.append(activation())
            if dropout > 0:
                net.append(nn.Dropout(p=dropout))




            for layer in range(num_layers - 2):
                net.append(nn.Linear(hidden_dim, hidden_dim))

                net.append(activation())
                if dropout > 0:
                    net.append(nn.Dropout(p=dropout))
                net.append(nn.BatchNorm1d(hidden_dim))
            net.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*net)



            # self.linears.append(nn.Linear(input_dim, hidden_dim))
            # for layer in range(num_layers - 2):
                # self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            # self.linears.append(nn.Linear(hidden_dim, output_dim))

            # for layer in range(num_layers - 1):
                # self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            # for layer in range(num_layers - 1):
                # self.dropouts.append(nn.Dropout(p=dropout))

    def forward(self, x):
        return self.net(x)
        # if self.linear_or_not:
        #     # If linear model
        #     return self.linear(x)
        # else:
        #     # If MLP
        #     h = x
        #     for layer in range(self.num_layers - 1):
        #         h = self.linears[layer](h)
        #
        #
        #         h = self.batch_norms[layer](h)
        #
        #         h = F.relu(h)
        #         h = self.dropouts[layer](h)
        #         # h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        #     return self.linears[self.num_layers - 1](h)

    def init_weights(self):
        pass
        # for layer in self.linears:
        #     # nn.init.xavier_uniform_(layer.weight, gain=1e-2)
        #     nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        #     nn.init.constant_(layer.bias, 0.0)


# class MLP(nn.Module):
#     def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0.0):
#         '''
#             num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
#             input_dim: dimensionality of input features
#             hidden_dim: dimensionality of hidden units at ALL layers
#             output_dim: number of classes for prediction
#             device: which device to use
#         '''
#
#         super(MLP, self).__init__()
#
#         self.linear_or_not = True  # default is linear model
#         self.num_layers = num_layers
#
#         if num_layers < 1:
#             raise ValueError("number of layers should be positive!")
#         elif num_layers == 1:
#             # Linear model
#             self.linear = nn.Linear(input_dim, output_dim)
#         else:
#             # Multi-layer model
#             self.linear_or_not = False
#             self.linears = torch.nn.ModuleList()
#             self.batch_norms = torch.nn.ModuleList()
#             self.dropouts = torch.nn.ModuleList()
#
#             self.linears.append(nn.Linear(input_dim, hidden_dim))
#             for layer in range(num_layers - 2):
#                 self.linears.append(nn.Linear(hidden_dim, hidden_dim))
#             self.linears.append(nn.Linear(hidden_dim, output_dim))
#
#             for layer in range(num_layers - 1):
#                 self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
#             for layer in range(num_layers - 1):
#                 self.dropouts.append(nn.Dropout(p=dropout))
#
#     def forward(self, x):
#         if self.linear_or_not:
#             # If linear model
#             return self.linear(x)
#         else:
#             # If MLP
#             h = x
#             for layer in range(self.num_layers - 1):
#                 h = self.linears[layer](h)
#
#
#                 h = self.batch_norms[layer](h)
#
#                 h = F.relu(h)
#                 h = self.dropouts[layer](h)
#                 # h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
#             return self.linears[self.num_layers - 1](h)
#
#     def init_weights(self):
#         pass
#         # for layer in self.linears:
#         #     # nn.init.xavier_uniform_(layer.weight, gain=1e-2)
#         #     nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
#         #     nn.init.constant_(layer.bias, 0.0)


class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPActor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)