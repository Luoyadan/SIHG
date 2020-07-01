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
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


class SignedConv(MessagePassing):
    r"""The signed graph convolutional operator from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper

    .. math::
        \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w , \mathbf{x}_v \right]

        \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{neg})}
        \left[ \frac{1}{|\mathcal{N}^{-}(v)|} \sum_{w \in \mathcal{N}^{-}(v)}
        \mathbf{x}_w , \mathbf{x}_v \right]

    if :obj:`first_aggr` is set to :obj:`True`, and

    .. math::
        \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w^{(\textrm{pos})}, \frac{1}{|\mathcal{N}^{-}(v)|}
        \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{neg})} ,
        \mathbf{x}_v^{(\textrm{pos})} \right]

        \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w^{(\textrm{neg})}, \frac{1}{|\mathcal{N}^{-}(v)|}
        \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{pos})} ,
        \mathbf{x}_v^{(\textrm{neg})} \right]

    otherwise.
    In case :obj:`first_aggr` is :obj:`False`, the layer expects :obj:`x` to be
    a tensor where :obj:`x[:, :in_channels]` denotes the positive node features
    :math:`\mathbf{X}^{(\textrm{pos})}` and :obj:`x[:, in_channels:]` denotes
    the negative node features :math:`\mathbf{X}^{(\textrm{neg})}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        first_aggr (bool): Denotes which aggregation formula to use.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 manifolds,
                 first_aggr,
                 bias=True,
                 **kwargs):
        super(SignedConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr

        if first_aggr:
            self.lin_pos = Linear(2 * out_channels, out_channels//2, bias=bias)
            self.lin_neg = Linear(2 * out_channels, out_channels//2, bias=bias)
        else:
            self.lin_pos = Linear(3 * out_channels, out_channels, bias=bias)
            self.lin_neg = Linear(3 * out_channels, out_channels, bias=bias)
        self.manifolds = manifolds
        self.dropout = 0.1
        self.c = 1.
        if self.first_aggr:
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.weight = nn.Parameter(torch.Tensor(2*out_channels, 2*out_channels))
            self.bias = nn.Parameter(torch.Tensor(2*out_channels))
        self.use_bias = True
        self.reset_parameters()
        self.act = F.leaky_relu

    def reset_parameters(self):
        self.lin_pos.reset_parameters()
        self.lin_neg.reset_parameters()
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x, pos_edge_index, neg_edge_index):
        """"""
        # hyper linear
        pos_edge_index = add_remaining_self_loops(pos_edge_index, num_nodes=x.size(0))[0]

        x = self.manifolds.proj(self.manifolds.expmap0(self.manifolds.proj_tan0(x, self.c), c=self.c), c=self.c)
        # drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        # mv = self.manifolds.mobius_matvec(drop_weight, x, self.c)
        # res = self.manifolds.proj(mv, self.c)
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
        x = self.manifolds.logmap0(res, c=self.c)


        if self.first_aggr:
            if self.manifolds.name == 'Hyperboloid':
                assert x.size(1) == self.in_channels - 1
            else:
                assert x.size(1) == self.in_channels
            x_pos = torch.cat(
                [self.propagate(pos_edge_index, x=x, size=None), x], dim=1)
            x_neg = torch.cat(
                [self.propagate(neg_edge_index, x=x, size=None), x], dim=1)

        else:
            assert x.size(1) == 2 * self.in_channels

            x_1, x_2 = x[:, :self.in_channels], x[:, self.in_channels:]

            x_pos = torch.cat([
                self.propagate(pos_edge_index, x=x_1, size=None),
                self.propagate(neg_edge_index, x=x_2, size=None),
                x_1,
            ], dim=1)

            x_neg = torch.cat([
                self.propagate(pos_edge_index, x=x_2, size=None),
                self.propagate(neg_edge_index, x=x_1, size=None),
                x_2,
            ], dim=1)
        assert not torch.isnan(x_pos).any()
        assert not torch.isnan(x_neg).any()
        x_pos = self.manifolds.proj(self.manifolds.expmap0(self.lin_pos(x_pos), c=self.c), c=self.c)
        x_neg = self.manifolds.proj(self.manifolds.expmap0(self.lin_neg(x_neg), c=self.c), c=self.c)

        x_out = torch.cat([x_pos, x_neg], dim=1)

        xt = self.act(self.manifolds.logmap0(x_out, c=self.c))
        xt = self.manifolds.proj_tan0(xt, c=self.c)
        xt = self.manifolds.proj(self.manifolds.expmap0(xt, c=self.c), c=self.c)
        if torch.isnan(xt).any():
            print("check here")
        assert not torch.isnan(xt).any()

        return xt

    def __repr__(self):
        return '{}({}, {}, first_aggr={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.first_aggr)
