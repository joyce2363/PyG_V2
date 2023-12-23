import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import argparse
import time
import torch
from tqdm import tqdm
from torch_geometric.nn import GCNConv, GATConv


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, dropout=0.5):
#         super(GCN, self).__init__()
#         # self.gc1 = spectral_norm(GCNConv(nfeat, nhid).lin)
#         self.gc1 = GCNConv(nfeat, nhid)

#     def forward(self, edge_index, x):
#         x = self.gc1(x, edge_index)
#         return x
    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5, nheads=1): 
        super(GAT, self).__init__()
        # nfeat = 96, 
        # nhid = 16,
        self.conv1 = GATConv(nfeat, nhid, heads=nheads, dropout=dropout)
        self.conv1.att = None 
        self.transition = nn.Sequential(
            nn.ReLU(), 
            nn.BatchNorm1d(nhid * nheads), 
            nn.Dropout(p=dropout)
        )
        for m in self.modules(): 
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        print("CALLING THIS FORWARD FUNCTION :p")
        print("CHECK THIS OUT: ", x.size())
        print("EDGE_INDEX: ", edge_index.size())
        x = self.conv1(x, edge_index)
        x = x.flatten(start_dim=1)  # Flatten node features across heads
        x = self.transition(x)
        return x 
    

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_model(nfeat, num_hidden):
    model = GAT(nfeat, num_hidden)

    return model


class FairGNN_3(nn.Module):
    def __init__(
        self, 
        adj, 
        features, 
        labels, 
        idx_train, 
        idx_val, 
        idx_test, 
        sens, 
        sens_idx, 
        num_hidden, 
        alpha, 
        beta, 
        acc, 
        sim_coeff, 
        lr, 
        weight_decay, 
        proj_hidden, 
        n_order=10, 
        subgraph_size=30, 
        epoch=2000
    ):
        super(FairGNN_3, self).__init__()

        parser = argparse.ArgumentParser()
        parser.add_argument("--no-cuda",action="store_true", default=False, help="Disables CUDA training.",
        )
        parser.add_argument("--seed", type=int, default=1, help="Random seed.")
        parser.add_argument("--epochs", type=int, default=epoch, help="Number of epochs to train.",
        )
        parser.add_argument( "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability).",
        )
        parser.add_argument("--dataset", type=str, default="nba", choices=["bail", "pokec_n", "pokec_z", "nba"],
        )
        parser.add_argument("--encoder", type=str, default="sage", choices=["gcn", "gin", "sage", "infomax", "jk"],
        )
        parser.add_argument("--batch_size", type=int, help="batch size", default=100)
        parser.add_argument("--subgraph_size", type=int, help="subgraph size", default=subgraph_size
        )
        parser.add_argument("--n_order", type=int, help="order of neighbor nodes", default=n_order
        )
        parser.add_argument("--hidden_size", type=int, help="hidden size", default=1024)
        parser.add_argument("--experiment_type", type=str, default="train", choices=["train", "cf", "test"],
        )  # train, cf, test

        args = parser.parse_known_args()[0]
        # args.num_hidden = num_hidden
        args.alpha = alpha
        args.beta = beta
        args.acc = acc
        args.lr = lr
        args.weight_decay = weight_decay

        self.sim_coeff = sim_coeff
        #self.encoder = encoder
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        self.num_hidden = num_hidden
        # nhid = args.num_hidden
        dropout = args.dropout
        self.estimator = GAT(features.shape[1], num_hidden)
        print("FIRST WORK")
        self.GNN = get_model(features.shape[1], num_hidden)
        print("SECOND WORK")
        self.classifier = nn.Linear(num_hidden, 1)
        self.adv = nn.Linear(num_hidden, 1)

        G_params = (
            list(self.GNN.parameters())
            + list(self.classifier.parameters())
            + list(self.estimator.parameters())
        )
        self.optimizer_G = torch.optim.Adam(
            G_params, lr=args.lr, weight_decay=args.weight_decay
        )
        self.optimizer_A = torch.optim.Adam(
            self.adv.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0

    def fair_metric(self, sens, labels, output, idx):
        val_y = labels[idx].cpu().numpy()
        idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
        idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

        idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

        pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
        parity = abs(
            sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1)
        )
        equality = abs(
            sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1)
            - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1)
        )

        return parity, equality

    def fair_metric_direct(self, pred, labels, sens):
        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
        parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
        equality = abs(
            sum(pred[idx_s0_y1]) / sum(idx_s0_y1)
            - sum(pred[idx_s1_y1]) / sum(idx_s1_y1)
        )
        return parity.item(), equality.item()

    def forward(self, g, x):
        s = self.estimator(g, x)
        z = self.GNN(g, x)
        y = self.classifier(z)
        return y, s

    def optimize(self, g, features, labels, idx_train, sens, idx_sens_train, edge_index):
        self.train()

        ### update E, G
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()
        # print("features.shape[1], optimize: ",features.shape[1])
        print("self.num_hidden, optimize: ", self.num_hidden)

        s = self.estimator(edge_index, features)
        h = self.GNN(edge_index, features)
        y = self.classifier(h)
        print("S: ", s)
        print("H: ", h)
        print("Y: ", y)
        s_g = self.adv(h)

        s_score = torch.sigmoid(s.detach())
        # s_score = (s_score > 0.5).float()
        s_score[idx_sens_train] = sens[idx_sens_train].unsqueeze(1).float()
        y_score = torch.sigmoid(y)
        self.cov = torch.abs(
            torch.mean(
                (s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))
            )
        )

        self.cls_loss = self.criterion(
            y[idx_train], labels[idx_train].unsqueeze(1).float()
        )
        self.adv_loss = self.criterion(s_g, s_score)

        self.G_loss = (
            self.cls_loss + self.args.alpha * self.cov - self.args.beta * self.adv_loss
        )
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.criterion(s_g, s_score)
        self.A_loss.backward()
        self.optimizer_A.step()

    def fit(
        self,
        g: torch.Tensor = None,
        features: torch.Tensor = None,
        labels: torch.Tensor = None,
        idx_train: torch.Tensor = None,
        idx_val: torch.Tensor = None,
        idx_test: torch.Tensor = None,
        sens: torch.Tensor = None,
        idx_sens_train: torch.Tensor = None,
        device="cuda",
    ):
        # with args
        if idx_sens_train is None:
            idx_sens_train = idx_train
        self = self.to(device)
        features = features.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        # idx_test = idx_test.to(device)
        sens = sens.to(device)
        idx_sens_train = idx_sens_train.to(device)

        args = self.args
        t_total = time.time()
        best_fair = 1000
        best_acc = 0

        self.g = g
        self.features = features
        self.labels = labels
        self.sens = sens

        self.edge_index = (
            torch.tensor(g.to_dense().nonzero(), dtype=torch.long).t().cuda()
        )
        self.val_loss = 0
        for epoch in tqdm(range(args.epochs)):
            t = time.time()
            self.train()
            print("SELF.FEATURES: ", self.features)
            print("SELF.EDGE_INDEX: ", self.edge_index)
            self.optimize(
                g, self.features, labels, idx_train, sens, idx_sens_train, self.edge_index
            )
            self.eval()

            output, s = self(self.edge_index, features)
            acc_val = accuracy(output[idx_val], labels[idx_val])

            parity_val, equality_val = self.fair_metric(sens, labels, output, idx_val)

            # if acc_val > args.acc: #and roc_val > args.roc:

            if acc_val > args.acc or epoch == 0:
                #if parity_val + equality_val < best_fair:
                if acc_val>best_acc:
                    best_epoch = epoch
                    best_fair = parity_val + equality_val
                    best_acc = acc_val
                    self.val_loss = -acc_val.detach().cpu().item()
                    self.eval()
                    output, s = self.forward(self.edge_index, self.x)

                    output = (output > 0).long().detach().cpu().numpy()
                    F1 = f1_score(
                        self.labels[idx_test].detach().cpu().numpy(),
                        output[idx_test],
                        average="micro",
                    )
                    ACC = accuracy_score(
                        self.labels[idx_test].detach().cpu().numpy(),
                        output[idx_test],
                    )
                    if self.labels.max() > 1:
                        AUCROC = 0
                    else:
                        AUCROC = roc_auc_score(
                            self.labels[idx_test].detach().cpu().numpy(),
                            output[idx_test],
                        )
                    (
                        ACC_sens0,
                        AUCROC_sens0,
                        F1_sens0,
                        ACC_sens1,
                        AUCROC_sens1,
                        F1_sens1,
                    ) = self.predict_sens_group(output[idx_test], idx_test)
                    SP, EO = self.fair_metric_direct(
                        output[idx_test],
                        self.labels[idx_test].detach().cpu().numpy(),
                        self.sens[idx_test].detach().cpu().numpy(),
                    )

                    self.temp_result = (
                        ACC,
                        AUCROC,
                        F1,
                        ACC_sens0,
                        AUCROC_sens0,
                        F1_sens0,
                        ACC_sens1,
                        AUCROC_sens1,
                        F1_sens1,
                        SP,
                        EO,
                    )

            if epoch <= 10 and acc_val > args.acc:
                args.acc = acc_val

        print("Optimization Finished! Best Epoch:", best_epoch)
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    def predict(self, idx_test):
        return self.temp_result

    def predict_(self, idx_test):
        self.eval()
        # output, s = self.forward(self.edge_index, self.x)
        output, s = self.forward (self.features.to(self.device), self.edge_index.to(self.device))
        output = (output > 0).long().detach().cpu().numpy()
        F1 = f1_score(
            self.labels[idx_test].detach().cpu().numpy(),
            output[idx_test],
            average="micro",
        )
        ACC = accuracy_score(
            self.labels[idx_test].detach().cpu().numpy(),
            output[idx_test],
        )
        if self.labels.max() > 1:
            AUCROC = 0
        else:
            AUCROC = roc_auc_score(
                self.labels[idx_test].detach().cpu().numpy(), output[idx_test]
            )
        (
            ACC_sens0,
            AUCROC_sens0,
            F1_sens0,
            ACC_sens1,
            AUCROC_sens1,
            F1_sens1,
        ) = self.predict_sens_group(output[idx_test], idx_test)
        SP, EO = self.fair_metric_direct(
            output[idx_test],
            self.labels[idx_test].detach().cpu().numpy(),
            self.sens[idx_test].detach().cpu().numpy(),
        )

        return (
            ACC,
            AUCROC,
            F1,
            ACC_sens0,
            AUCROC_sens0,
            F1_sens0,
            ACC_sens1,
            AUCROC_sens1,
            F1_sens1,
            SP,
            EO,
        )

    def predict_sens_group(self, output, idx_test):
        # pred = self.lgreg.predict(self.embs[idx_test])
        pred = output
        result = []
        for sens in [0, 1]:
            F1 = f1_score(
                self.labels[idx_test][
                    self.sens[idx_test].detach().cpu().numpy() == sens
                ]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test].detach().cpu().numpy() == sens],
                average="micro",
            )
            ACC = accuracy_score(
                self.labels[idx_test][
                    self.sens[idx_test].detach().cpu().numpy() == sens
                ]
                .detach()
                .cpu()
                .numpy(),
                pred[self.sens[idx_test].detach().cpu().numpy() == sens],
            )
            if self.labels.max() > 1:
                AUCROC = 0
            else:
                AUCROC = roc_auc_score(
                    self.labels[idx_test][
                        self.sens[idx_test].detach().cpu().numpy() == sens
                    ]
                    .detach()
                    .cpu()
                    .numpy(),
                    pred[self.sens[idx_test].detach().cpu().numpy() == sens],
                )
            result.extend([ACC, AUCROC, F1])

        return result