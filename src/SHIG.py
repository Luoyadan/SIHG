import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import coalesce
from signed_conv import SignedConv
import manifolds
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import numpy as np
import warnings
warnings.filterwarnings('always')
warnings.simplefilter("ignore")
warnings.simplefilter('always')

class MutualInfoNet(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(InfoNet, self).__init__()
        self.fc_x = nn.Linear(hidden_channels, hidden_channels)
        self.fc_y = nn.Linear(1, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_x.reset_parameters()
        self.fc_y.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, y):
        out = F.relu(self.fc_x(x) + self.fc_y(y.unsqueeze(-1)))
        out = self.fc(out)
        return out


class SHIG_Model(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers, lamb=1, trial=None, args=None,
                 bias=True):
        super(SignedGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.trial = trial
        self.num_layers = num_layers
        self.lamb = lamb
        self.args = args
        self.manifolds = getattr(manifolds, args.manifolds)()

        if self.manifolds.name == 'Hyperboloid':
            in_channels = in_channels + 1
        self.conv1 = SignedConv(in_channels, hidden_channels, self.manifolds, self.args,
                                first_aggr=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                SignedConv(hidden_channels // 2, hidden_channels // 2, self.manifolds, self.args,
                           first_aggr=False))

        self.lin = torch.nn.Linear(2 * hidden_channels, 3)
        self.info_net = MutualInfoNet(2 * hidden_channels)
        self.r = args.r
        self.t = args.t
        self.c = args.c
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
        self.info_net.reset_parameters()

    def split_edges(self, edge_index, test_ratio=0.2):

        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[torch.randperm(mask.size(0))[:int(test_ratio * mask.size(0))]] = 0

        train_edge_index = edge_index[:, mask]
        test_edge_index = edge_index[:, ~mask]

        return train_edge_index, test_edge_index

    def forward(self, x, pos_edge_index, neg_edge_index):
        """
        Args:
            x (Tensor): The input node features.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        if self.manifolds.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        # x = self.manifolds.proj(self.manifolds.expmap0(self.manifolds.proj_tan0(x, self.c), c=self.c), c=self.c)

        # Aggregation for different layers
        z = self.conv1(x, pos_edge_index, neg_edge_index)
        for conv in self.convs:
            z = conv(z, pos_edge_index, neg_edge_index)
        return z

    def discriminate(self, z, edge_index, id=None, last=False):
        """
        Args:
            x (Tensor): The input node features.
            edge_index (LongTensor): The edge indices.
        """
        if id is not None:
            value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
            out = self.info_net(value, id)

        else:
            out = torch.clamp_min(1. / (torch.exp((self.manifolds.sqdist(z[edge_index[0]], z[edge_index[1]], 1) - self.r) / self.t) + 1.0), 0)
        del z
        return out

    def mutual_loss(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        pos_y = pos_edge_index.new_full((pos_edge_index.size(1), ), 0).float()
        neg_y = neg_edge_index.new_full((neg_edge_index.size(1), ), 1).float()
        neu_y = none_edge_index.new_full((none_edge_index.size(1), ), 2).float()
        all_y = torch.cat((pos_y, neg_y, neu_y))
        idx = torch.randperm(all_y.size()[0])
        shuffle_y = all_y[idx]
        index = torch.cat((pos_edge_index, neg_edge_index, none_edge_index), 1)
        info_pred = self.discriminate(z, index, id=all_y)
        info_shuffle = self.discriminate(z, index, id=shuffle_y)
        mutual_loss = torch.mean(info_pred) - torch.log(torch.mean(torch.exp(info_shuffle)))

        return -mutual_loss


    def nll_loss(self, z, pos_edge_index, neg_edge_index):
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative nedges
        :obj:`neg_edge_index`.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """

        # edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        # none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.binary_cross_entropy(
            self.discriminate(z, pos_edge_index).squeeze(),
            pos_edge_index.new_full((pos_edge_index.size(1), ), 1).float())
        nll_loss += F.binary_cross_entropy(
            self.discriminate(z, neg_edge_index).squeeze(),
            neg_edge_index.new_full((neg_edge_index.size(1), ), 0).float())

        return nll_loss

    def pos_embedding_loss(self, z, pos_edge_index):
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))
        torch.cuda.empty_cache()
        out = self.manifolds.sqdist(z[i], z[j], 1) - self.manifolds.sqdist(z[i], z[k], 1)
        if torch.isinf(out).any():
            print("check here")
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z, neg_edge_index):
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))
        torch.cuda.empty_cache()
        out = self.manifolds.sqdist(z[i], z[k], 1) - self.manifolds.sqdist(z[i], z[j], 1)
        return torch.clamp(out, min=0).mean()


    def loss(self, z, pos_edge_index, neg_edge_index, device):
        """Computes the overall objective.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        alpha = self.trial.suggest_uniform("alpha", 0, 3)
        gamma = self.trial.suggest_uniform("gamma", 0, 3)

        beta = 0.83
        mutual_info_loss = self.mutual_loss(z, pos_edge_index, neg_edge_index)

        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + alpha * loss_1 + beta * loss_2 + gamma * mutual_info_loss

    def test(self, z, pos_edge_index, neg_edge_index, neg_ratio, last=False):
        """Evaluates node embeddings :obj:`z` on positive and negative test
        edges by computing AUC and F1 scores.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """

        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)
            neg_p = self.discriminate(z, neg_edge_index)
        pred = torch.cat([pos_p, neg_p]).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.int().numpy()

        auc = roc_auc_score(y, pred, average='weighted')
        f1 = f1_score(y, [1 if p > neg_ratio else 0 for p in pred], average='binary')
        f1_micro = f1_score(y, [1 if p > neg_ratio else 0 for p in pred], average='micro')
        f1_macro = f1_score(y, [1 if p > neg_ratio else 0 for p in pred], average='macro')

        return auc, f1, f1_macro, f1_micro


    def __repr__(self):
        return '{}({}, {}, num_layers={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.hidden_channels,
                                                  self.num_layers)
