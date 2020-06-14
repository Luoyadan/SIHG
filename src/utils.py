"""Data reading utils."""

import json
import numpy as np
import pandas as pd
from scipy import sparse
from texttable import Texttable
from sklearn.decomposition import TruncatedSVD, KernelPCA
from sklearn.metrics import roc_auc_score, f1_score
import torch
import scipy
from torch_sparse import coalesce
from tensorboardX import SummaryWriter
import datetime
def read_graph(args):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers.
    :param args: Arguments object.
    :return edges: Edges dictionary.
    """
    if 'bitcoin' in args.edge_path:
        dataset = pd.read_csv(args.edge_path).values.tolist()
    else:
        dataset = pd.read_csv(args.edge_path, sep='\t').values.tolist()
    edges = {}
    edges["positive_edges"] = [edge[0:2] for edge in dataset if edge[2] == 1]
    edges["negative_edges"] = [edge[0:2] for edge in dataset if edge[2] == -1]
    edges["ecount"] = len(dataset)
    edges["ncount"] = len(set([edge[0] for edge in dataset]+[edge[1] for edge in dataset]))
    return edges

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def calculate_auc(targets, predictions, edges):
    """
    Calculate performance measures on test dataset.
    :param targets: Target vector to predict.
    :param predictions: Predictions vector.
    :param edges: Edges dictionary with number of edges etc.
    :return auc: AUC value.
    :return f1: F1-score.
    """
    neg_ratio = len(edges["negative_edges"])/edges["ecount"]
    targets_rev = [0 if target == 1 else 1 for target in targets] # turn the first indicator (1) to positive
    auc = roc_auc_score(targets_rev, predictions[:, 0])
    f1 = f1_score(targets_rev, [1 if p > neg_ratio else 0 for p in predictions[:, 0]])
    # f1 = f1_score(targets, np.argmax(predictions, axis=1))
    return auc, f1

def score_printer(logs):
    """
    Print the performance for every 10th epoch on the test dataset.
    :param logs: Log dictionary.
    """
    t = Texttable()
    t.add_rows([per for i, per in enumerate(logs["performance"]) if i % 10 == 0])
    print(t.draw())

def save_logs(args, logs):
    """
    Save the logs at the path.
    :param args: Arguments objects.
    :param logs: Log dictionary.
    """
    writer = SummaryWriter(args.log_path + args.dataset + '_Layer_{}/'.format(args.num_layers)+'_{}'.format((datetime.datetime.now()).strftime("%Y%m%d%H%M%S")))
    for i, item in enumerate(logs["performance"]):
        if i > 0:
            writer.add_scalar('AUC', item[1], item[0])
            writer.add_scalar('F1', item[2], item[0])
            writer.add_scalar('Loss', logs["loss"][i-1], item[0])
    writer.close()
    # with open(args.log_path, "w") as f:
    #     json.dump(logs, f)

def setup_features(args, positive_edges, negative_edges, node_count):
    """
    Setting up the node features as a numpy array.
    :param args: Arguments object.
    :param positive_edges: Positive edges list.
    :param negative_edges: Negative edges list.
    :param node_count: Number of nodes.
    :return X: Node features.
    """
    if args.spectral_features:
        X = create_spectral_features(args, positive_edges, negative_edges, node_count)
    else:
        X = create_general_features(args)
    return X

def create_general_features(args):
    """
    Reading features using the path.
    :param args: Arguments object.
    :return X: Node features.
    """
    X = np.array(pd.read_csv(args.features_path))
    return X

def create_spectral_features(args, pos_edge_index, neg_edge_index,
                             num_nodes=None):
    r"""Creates :obj:`in_channels` spectral node features based on
    positive and negative edges.

    Args:
        pos_edge_index (LongTensor): The positive edge indices.
        neg_edge_index (LongTensor): The negative edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`pos_edge_index` and
            :attr:`neg_edge_index`. (default: :obj:`None`)
    """
    pos_edge_index = torch.tensor(pos_edge_index).t().long()
    neg_edge_index = torch.tensor(neg_edge_index).t().long()
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    N = edge_index.max().item() + 1 # if num_nodes is None else num_nodes
    edge_index = edge_index.to(torch.device('cpu')).long()

    pos_val = torch.full((pos_edge_index.size(1), ), 2, dtype=torch.float)
    neg_val = torch.full((neg_edge_index.size(1), ), 0, dtype=torch.float)
    val = torch.cat([pos_val, neg_val], dim=0)

    row, col = edge_index
    edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
    val = torch.cat([val, val], dim=0)

    edge_index, val = coalesce(edge_index, val, N, N)
    val = val - 1

    # Borrowed from:
    # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
    edge_index = edge_index.detach().numpy()
    val = val.detach().numpy()
    A = scipy.sparse.coo_matrix((val, edge_index), shape=(N, N))
    svd = TruncatedSVD(n_components=args.reduction_dimensions,
                           n_iter=args.reduction_iterations,
                           random_state=args.seed)
    svd.fit(A)
    x = svd.components_.T
    return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)

# def create_spectral_features(args, positive_edges, negative_edges, node_count):
#     """
#     Creating spectral node features using the train dataset edges.
#     :param args: Arguments object.
#     :param positive_edges: Positive edges list.
#     :param negative_edges: Negative edges list.
#     :param node_count: Number of nodes.
#     :return X: Node features.
#     """
#     dec = 'TSVD' # 'KPCA'
#     p_edges = positive_edges + [[edge[1], edge[0]] for edge in positive_edges]
#     n_edges = negative_edges + [[edge[1], edge[0]] for edge in negative_edges]
#     train_edges = p_edges + n_edges
#     index_1 = [edge[0] for edge in train_edges]
#     index_2 = [edge[1] for edge in train_edges]
#     values = [1]*len(p_edges) + [-1]*len(n_edges)
#     shaping = (node_count, node_count)
#     signed_A = sparse.csr_matrix(sparse.coo_matrix((values, (index_1, index_2)),
#                                                    shape=shaping,
#                                                    dtype=np.float32))
#
#     if dec == 'TSVD':
#         svd = TruncatedSVD(n_components=args.reduction_dimensions,
#                            n_iter=args.reduction_iterations,
#                            random_state=args.seed)
#         svd.fit(signed_A)
#         X = svd.components_.T
#
#     elif dec == 'KPCA':
#         transformer = KernelPCA(n_components=args.reduction_dimensions,
#                                 kernel='rbf',
#                                 random_state=args.seed)
#         X = transformer.fit_transform(signed_A)
#     return X
