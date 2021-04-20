import time
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import trange

from utils import setup_features
from sklearn.model_selection import train_test_split
from torch_geometric.nn import MessagePassing
from SHIG import SHIG_Model
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
import datetime


class SHIGNetwork(torch.nn.Module):

    def __init__(self, device, args, trial, X):
        super(SHIGNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        self.trial = trial
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

        self.aggregator = SHIG_Model(self.X.shape[1], self.neurons[-1], num_layers=self.args.num_layers,
                                     trial=self.trial, args=self.args).cuda()


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
        loss = self.aggregator.loss(self.z, positive_edges, negative_edges, self.device)

        return loss, self.z

class SHIGTrainer(object):

    def __init__(self, args, edges):

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
        # tensorboard
        self.writer = SummaryWriter(self.args.log_path + self.args.dataset + '_Layer_{}/'.format(self.args.num_layers)+'_{}'.
                                    format((datetime.datetime.now()).strftime("%Y%m%d%H%M%S")))

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
        self.neg_ratio = len(self.negative_edges) / self.ecount

        self.X = setup_features(self.args,
                                self.positive_edges,
                                self.negative_edges,
                                self.edges["ncount"])

        self.positive_edges = torch.from_numpy(np.array(self.positive_edges,
                                                        dtype=np.int64).T).type(torch.long).cuda()

        self.negative_edges = torch.from_numpy(np.array(self.negative_edges,
                                                        dtype=np.int64).T).type(torch.long).cuda()

        self.y = np.array([0 if i < int(self.ecount/2) else 1 for i in range(self.ecount)]+[2]*(self.ecount*2))
        self.y = torch.from_numpy(self.y).type(torch.LongTensor).cuda()
        self.X = self.X.cuda()


    def score_model(self, epoch, last=False):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number.
        """
        self.model.eval()
        score_positive_edges = torch.from_numpy(np.array(self.test_positive_edges, dtype=np.int64).T).type(torch.long).cuda()
        score_negative_edges = torch.from_numpy(np.array(self.test_negative_edges, dtype=np.int64).T).type(torch.long).cuda()

        loss, self.z = self.model(self.positive_edges, self.negative_edges, self.y)
        auc, f1, f1_macro, f1_micro = self.model.aggregator.test(self.z, score_positive_edges, score_negative_edges, self.neg_ratio, last)
        # self.trial.report(auc, epoch+1)
        self.logs["performance"].append([epoch+1, auc, f1_micro, f1, f1_macro])
        self.writer.add_scalar('AUC', auc, epoch)
        self.writer.add_scalar('F1', f1, epoch)
        self.writer.add_scalar('F1_macro', f1_macro, epoch)
        self.writer.add_scalar('F1_micro', f1_micro, epoch)
        if last:
            embedding_pos = torch.cat([self.z[score_positive_edges[0]], self.z[score_positive_edges[1]]], dim=1).cpu()
            embedding_neg = torch.cat([self.z[score_negative_edges[0]], self.z[score_negative_edges[1]]], dim=1).cpu()
            embedding = torch.cat([embedding_pos, embedding_neg], dim=0)
            y = torch.cat(
                [embedding.new_ones((score_positive_edges.size(1))),
                 -1*embedding.new_ones(score_negative_edges.size(1))])
            embedding, y = embedding.detach().numpy(), y.int().numpy()
            self.writer.add_embedding(embedding, metadata=y, global_step=epoch)
            self.writer.close()

        if self.args.verbose:
            print('{}{} Val(auc,f1,f1_macro,f1_micro):{} {} {} {}'.format("#" * 10, "BEST EPOCH",
                                                                      self.logs["performance"][-1][1],
                                                                      self.logs["performance"][-1][3],
                                                                      self.logs["performance"][-1][4],
                                                                      self.logs["performance"][-1][2]))

        self.model.train()

    def create_and_train_model(self, trial):
        """
        Model training and scoring.
        """
        print("\nTraining started.\n")
        self.trial = trial
        self.model = SHIGNetwork(self.device, self.args, trial, self.X).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epochs)
        last = False
        self.model.train()
        self.epochs = trange(self.args.epochs, desc="Loss")

        for epoch in self.epochs:
            start_time = time.time()
            self.optimizer.zero_grad()
            loss, _ = self.model(self.positive_edges, self.negative_edges, self.y)
            self.logs["loss"].append(loss.item())
            loss.backward()
            self.epochs.set_description("SHIG (Loss=%g)" % round(loss.item(), 4))
            self.optimizer.step()
            self.lr_scheduler.step()
            self.logs["training_time"].append([epoch+1, time.time()-start_time])
            if self.args.test_size > 0:
                if epoch == self.args.epochs -1:
                    last = True
                self.score_model(epoch, last)
