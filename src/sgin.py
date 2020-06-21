import time
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from tqdm import trange

from utils import setup_features
from sklearn.model_selection import train_test_split
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
from signed_gcn import SignedGCN
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        input_layer (bool): whethe the GIN conv is applied to input layer or not. (Input node labels are uniform...)
    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add", input_layer = False):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2*emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(3, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 3)
        self_loop_attr[:, 0] = 1 # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return torch.cat([x_j, edge_attr], dim = 1)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gat, graphsage, gcn

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0.5, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False

            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add", input_layer=input_layer))
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim, input_layer=input_layer))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim, input_layer=input_layer))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim, input_layer=input_layer))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            if layer == self.num_layer - 1:
                # remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]

        return node_representation


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)

class InfoMaxNetwork(torch.nn.Module):
    """
    Signed Graph Convolutional Network Class.
    For details see: Signed Graph Convolutional Network.
    Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.
    https://arxiv.org/abs/1808.06354
    """
    def __init__(self, device, args, X):
        super(InfoMaxNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.X = X
        self.setup_layers()


    def setup_layers(self):
        """
        Adding Base Layers, Deep Signed GraphSAGE layers.
        Assing Regression Parameters if the model is not a single layer model.
        """
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers
        self.layers = len(self.neurons)
        self.discriminator = Discriminator(self.neurons[-1])
        self.gnn = GNN(self.args.num_layers, self.neurons[-1])


    def forward(self, positive_edges, negative_edges, target):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param positive_edges: Positive edges.
        :param negative_edges: Negative edges.
        :param target: Target vectors.
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.
        """

        self.z = self.aggregator.forward(self.X, positive_edges, negative_edges)
        loss = self.aggregator.loss(self.z, positive_edges, negative_edges)

        return loss, self.z

class SignedGCNTrainer(object):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """
    def __init__(self, args, edges):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure with positive and negative edges separated.
        """
        self.args = args
        self.edges = edges
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logs()

    def setup_logs(self):
        """
        Creating a log dictionary.
        """
        self.logs = {}
        self.logs["parameters"] = vars(self.args)
        self.logs["loss"] = []
        self.logs["performance"] = [["Epoch", "AUC", "F1"]]
        self.logs["training_time"] = [["Epoch", "Seconds"]]

    def setup_dataset(self):
        """
        Creating train and test split.
        """
        self.positive_edges, self.test_positive_edges = train_test_split(self.edges["positive_edges"],
                                                                         test_size=self.args.test_size,
                                                                         random_state=self.args.seed)

        self.negative_edges, self.test_negative_edges = train_test_split(self.edges["negative_edges"],
                                                                         test_size=self.args.test_size,
                                                                         random_state=self.args.seed)
        self.ecount = len(self.positive_edges + self.negative_edges)

        self.X = setup_features(self.args,
                                self.positive_edges,
                                self.negative_edges,
                                self.edges["ncount"])

        self.positive_edges = torch.from_numpy(np.array(self.positive_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)

        self.negative_edges = torch.from_numpy(np.array(self.negative_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)

        self.y = np.array([0 if i < int(self.ecount/2) else 1 for i in range(self.ecount)]+[2]*(self.ecount*2))
        self.y = torch.from_numpy(self.y).type(torch.LongTensor).to(self.device)
        self.X = self.X.to(self.device)


    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number.
        """

        score_positive_edges = torch.from_numpy(np.array(self.test_positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        score_negative_edges = torch.from_numpy(np.array(self.test_negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)

        loss, self.z = self.model(self.positive_edges, self.negative_edges, self.y)
        auc, f1 = self.model.aggregator.test(self.z, score_positive_edges, score_negative_edges)
        self.logs["performance"].append([epoch+1, auc, f1])

    def create_and_train_model(self):
        """
        Model training and scoring.
        """
        print("\nTraining started.\n")

        self.model = SignedGCNTrainer(self.device, self.args, self.X).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        self.model.train()
        self.epochs = trange(self.args.epochs, desc="Loss")
        for epoch in self.epochs:
            start_time = time.time()
            self.optimizer.zero_grad()
            loss, _ = self.model(self.positive_edges, self.negative_edges, self.y)
            self.logs["loss"].append(loss.item())
            loss.backward()
            self.epochs.set_description("SGCN (Loss=%g)" % round(loss.item(), 4))
            self.optimizer.step()
            self.logs["training_time"].append([epoch+1, time.time()-start_time])
            if self.args.test_size > 0:
                self.score_model(epoch)

    def save_model(self):
        """
        Saving the embedding and model weights.
        """
        print("\nEmbedding is saved.\n")
