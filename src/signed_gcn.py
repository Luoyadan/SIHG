import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import coalesce
from signed_conv import SignedConv
from sklearn.metrics import normalized_mutual_info_score
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)

class SignedGCN(torch.nn.Module):
    r"""The signed graph convolutional network model from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper.
    Internally, this module uses the
    :class:`torch_geometric.nn.conv.SignedConv` operator.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of layers.
        lamb (float, optional): Balances the contributions of the overall
            objective. (default: :obj:`5`)
        bias (bool, optional): If set to :obj:`False`, all layers will not
            learn an additive bias. (default: :obj:`True`)
    """

    def __init__(self, in_channels, hidden_channels, num_layers, lamb=1,
                 bias=True):
        super(SignedGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lamb = lamb

        self.conv1 = SignedConv(in_channels, hidden_channels//2,
                                first_aggr=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                SignedConv(hidden_channels//2, hidden_channels//2,
                           first_aggr=False))

        self.lin = torch.nn.Linear(2*hidden_channels, 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def split_edges(self, edge_index, test_ratio=0.2):
        r"""Splits the edges :obj:`edge_index` into train and test edges.

        Args:
            edge_index (LongTensor): The edge indices.
            test_ratio (float, optional): The ratio of test edges.
                (default: :obj:`0.2`)
        """
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[torch.randperm(mask.size(0))[:int(test_ratio * mask.size(0))]] = 0

        train_edge_index = edge_index[:, mask]
        test_edge_index = edge_index[:, ~mask]

        return train_edge_index, test_edge_index

    def forward(self, x, pos_edge_index, neg_edge_index):
        """Computes node embeddings :obj:`z` based on positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`.

        Args:
            x (Tensor): The input node features.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        z = torch.tanh(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = torch.tanh(conv(z, pos_edge_index, neg_edge_index))
        return z

    def discriminate(self, z, edge_index, id=None):
        """Given node embeddings :obj:`z`, classifies the link relation
        between node pairs :obj:`edge_index` to be either positive,
        negative or non-existent.

        Args:
            x (Tensor): The input node features.
            edge_index (LongTensor): The edge indices.
        """
        # test_a = z[edge_index[0]].detach().cpu().numpy()
        # test_b = z[edge_index[1]].detach().cpu().numpy()
        # mi = torch.tensor([normalized_mutual_info_score(test_a[i, :], test_b[i, :]) for i in range(edge_index.size()[1])])
        # print('mutual info: max {}; min {}'.format(mi.max(), mi.min()))
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        # if id is not None:
        #     info_scores = self.info_net(value, id)
        #     return info_scores
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

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
        # neg_shuffle = all_y[idx][pos_edge_index.size(1):]
        #
        # pos_pred = self.discriminate(z, pos_edge_index, id=pos_y)
        # pos_shuffle = self.discriminate(z, pos_edge_index, id=pos_shuffle)
        # neg_pred = self.discriminate(z, neg_edge_index, id=neg_y)
        # neg_shuffle = self.discriminate(z, neg_edge_index, id=neg_shuffle)
        info_pred = self.discriminate(z, index, id=all_y)
        info_shuffle = self.discriminate(z, index, id=shuffle_y)
        # pos_loss = torch.mean(pos_pred) - torch.log(torch.mean(torch.exp(pos_shuffle)))
        # neg_loss = torch.mean(neg_pred) - torch.log(torch.mean(torch.exp(neg_shuffle)))

        # print("mutual loss: pos {}; neg {}".format(pos_loss, neg_loss))
        mutual_loss = torch.mean(info_pred) - torch.log(torch.mean(torch.exp(info_shuffle)))
        # print(mutual_loss.item())
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

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1), ), 0))
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1), ), 1))
        nll_loss += 0.5 * F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1), ), 2))
        return nll_loss

    def pos_embedding_loss(self, z, pos_edge_index):
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z, neg_edge_index):
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(self, z, pos_edge_index, neg_edge_index):
        """Computes the overall objective.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        # mutual_info_loss = self.mutual_loss(z, pos_edge_index, neg_edge_index)
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + 1 * (loss_1 + loss_2)

    def test(self, z, pos_edge_index, neg_edge_index):
        """Evaluates node embeddings :obj:`z` on positive and negative test
        edges by computing AUC and F1 scores.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        auc = roc_auc_score(y, pred, average='macro')
        f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0

        return auc, f1

    def __repr__(self):
        return '{}({}, {}, num_layers={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.hidden_channels,
                                                  self.num_layers)
