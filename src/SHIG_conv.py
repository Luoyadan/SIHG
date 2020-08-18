import torch
import math
from torch.nn import Linear
from message_passing import MessagePassing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul_
# from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot

class SignedConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 manifolds,
                 args,
                 first_aggr,
                 **kwargs):
        super(SignedConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr
        self.heads = args.heads
        self.use_bias = args.use_bias


        if first_aggr:
            self.lin_pos = Linear(2 * out_channels, out_channels // 2, bias=self.use_bias)
            self.lin_neg = Linear(2 * out_channels, out_channels // 2, bias=self.use_bias)
            self.lin_pos_agg = Linear(out_channels, out_channels * self.heads, bias=self.use_bias)
            self.lin_neg_agg = Linear(out_channels, out_channels * self.heads, bias=self.use_bias)
        else:
            self.lin_pos = Linear(3 * out_channels, out_channels, bias=self.use_bias)
            self.lin_neg = Linear(3 * out_channels, out_channels, bias=self.use_bias)
            self.lin_pos_agg = Linear(out_channels, out_channels * self.heads, bias=self.use_bias)
            self.lin_neg_agg = Linear(out_channels, out_channels * self.heads, bias=self.use_bias)

        self.manifolds = manifolds
        self.dropout = args.dropout
        self.c = args.c

        if self.first_aggr:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.att_i = Parameter(torch.Tensor(1, self.heads, out_channels))
            self.att_j = Parameter(torch.Tensor(1, self.heads, out_channels))
        else:
            self.weight = nn.Parameter(torch.Tensor(2 * out_channels, 2 * out_channels))
            self.bias = nn.Parameter(torch.Tensor(2 * out_channels))
            self.att_i = Parameter(torch.Tensor(1, self.heads, out_channels))
            self.att_j = Parameter(torch.Tensor(1, self.heads, out_channels))


        self.negative_slope = 0.2
        self.act = F.leaky_relu
        self.reset_parameters()


    def reset_parameters(self):
        self.lin_pos.reset_parameters()
        self.lin_neg.reset_parameters()
        glorot(self.lin_pos_agg.weight)
        glorot(self.lin_neg_agg.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x, pos_edge_index, neg_edge_index, return_attention_weights=True):
        """"""
        # hyper linear
        pos_edge_index = add_remaining_self_loops(pos_edge_index, num_nodes=x.size(0))[0]

        x = self.manifolds.proj(self.manifolds.expmap0(self.manifolds.proj_tan0(x, self.c), c=self.c), c=self.c)
        if self.manifolds.name != 'PoincareBall':
            drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
            mv = self.manifolds.mobius_matvec(drop_weight, x, self.c)
            res = self.manifolds.proj(mv, self.c)
        else:
            res = x
        if torch.isnan(res).any():
            print("check here")
        assert not torch.isnan(res).any()
        if self.use_bias:
            bias = self.manifolds.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifolds.expmap0(bias, self.c)
            hyp_bias = self.manifolds.proj(hyp_bias, self.c)
            res = self.manifolds.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifolds.proj(res, self.c)
        torch.cuda.empty_cache()
        x = (self.manifolds.logmap0(res, c=self.c)).cuda()

        if self.first_aggr:
            if self.manifolds.name == 'Hyperboloid':
                assert x.size(1) == self.in_channels - 1
            else:
                assert x.size(1) == self.in_channels

            if return_attention_weights:
                x_trans_pos = (self.lin_pos_agg(x), self.lin_pos_agg(x))
                x_trans_neg = (self.lin_neg_agg(x), self.lin_neg_agg(x))
            else:
                x_trans_pos = x
                x_trans_neg = x

            x_pos = torch.cat(
                [self.propagate(pos_edge_index, x=x_trans_pos, size=None, return_attention_weights=return_attention_weights), x], dim=1)
            x_neg = torch.cat(
                [self.propagate(neg_edge_index, x=x_trans_neg, size=None, return_attention_weights=return_attention_weights), x], dim=1)

        else:
            assert x.size(1) == 2 * self.in_channels

            x_1, x_2 = x[:, :self.in_channels], x[:, self.in_channels:]

            x_pos = torch.cat([
                self.propagate(pos_edge_index, x=(self.lin_pos_agg(x_1), self.lin_pos_agg(x_1)), size=None, return_attention_weights=return_attention_weights),
                self.propagate(neg_edge_index, x=(self.lin_neg_agg(x_2), self.lin_neg_agg(x_2)), size=None, return_attention_weights=return_attention_weights),
                x_1,
            ], dim=1)

            x_neg = torch.cat([
                self.propagate(pos_edge_index, x=(self.lin_pos_agg(x_2), self.lin_pos_agg(x_2)), size=None, return_attention_weights=return_attention_weights),
                self.propagate(neg_edge_index, x=(self.lin_neg_agg(x_1), self.lin_neg_agg(x_1)), size=None, return_attention_weights=return_attention_weights),
                x_2,
            ], dim=1)
        assert not torch.isnan(x_pos).any()
        assert not torch.isnan(x_neg).any()
        x_pos = self.manifolds.proj(self.manifolds.expmap0(self.lin_pos(x_pos), c=self.c), c=self.c)
        x_neg = self.manifolds.proj(self.manifolds.expmap0(self.lin_neg(x_neg), c=self.c), c=self.c)

        x_out = torch.cat([x_pos, x_neg], dim=1)

        xt = self.act(self.manifolds.logmap0(x_out, c=self.c), self.negative_slope)
        xt = self.manifolds.proj_tan0(xt, c=self.c)
        xt = self.manifolds.proj(self.manifolds.expmap0(xt, c=self.c), c=self.c)
        if torch.isnan(xt).any():
            print("check here")
        assert not torch.isnan(xt).any()

        return xt

    def message(self, x_i, x_j, edge_index_i, size_i,
                return_attention_weights):
        # Compute attention coefficients.
        if return_attention_weights is False:
            return x_j
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = 2 * softmax(alpha, edge_index_i, size_i) - 1

        if return_attention_weights:
            self.__alpha__ = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return (x_j * alpha.view(-1, self.heads, 1)).squeeze()

    def __repr__(self):
        return '{}({}, {}, first_aggr={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.first_aggr)
