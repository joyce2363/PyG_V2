import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import argparse
import time
import torch
from tqdm import tqdm
from torch_geometric.nn import GCNConv
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#         self.body = GCN_Body(nfeat,nhid,dropout)
#         self.fc = nn.Linear(nhid,nclass)

#     def forward(self, g, x):
#         x = self.body(g,x)
#         x = self.fc(x)
#         return x



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        # self.gc1 = spectral_norm(GCNConv(nfeat, nhid).lin)
        self.gc1 = GCNConv(nfeat, nhid)
        # self.gc2 = GCNConv(nhid, nhid)
        # self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(nhid, nclass) #nhid, nhid

    def forward(self, edge_index, x):
        # x = F.relu(self.gc1(x, edge_index))
        # x = self.dropout(x)
        x = self.gc1(x, edge_index)
        # x = self.gc2(x, edge_index)
        x = self.fc(x)
        return x


def accuracy(output, labels):
    output = output.squeeze()
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_model(nfeat, args):
    model = GCN(nfeat, nhid=args.num_hidden, nclass=args.num_hidden, dropout=args.dropout)

    return model


class FairGNN_correct1(nn.Module):
    def __init__(
        self, nfeat, sim_coeff=0.61, n_order=27, subgraph_size=117, acc=0.43, epoch=2000
    ):
        super(FairGNN_correct1, self).__init__()

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="Disables CUDA training.",
        )
        parser.add_argument("--seed", type=int, default=1, help="Random seed.")
        parser.add_argument(
            "--epochs",
            type=int,
            default=2000,  # 1000
            help="Number of epochs to train.",
        )
        parser.add_argument(
            "--lr", type=float, default=0.004, help="Initial learning rate."
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-5,
            help="Weight decay (L2 loss on parameters).",
        )
        parser.add_argument(
            "--proj_hidden",
            type=int,
            default=30, #og 16
            help="Number of hidden units in the projection layer of encoder.",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.5,
            help="Dropout rate (1 - keep probability).",
        )
        parser.add_argument(
            "--sim_coeff",
            type=float,
            default=sim_coeff,
            help="regularization similarity",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="bail",
            choices=["synthetic", "bail", "credit"],
        )
        parser.add_argument(
            "--encoder",
            type=str,
            default="gcn",
            choices=["gcn", "gin", "sage", "infomax", "jk"],
        )
        parser.add_argument("--batch_size", type=int, help="batch size", default=100)
        parser.add_argument(
            "--subgraph_size", type=int, help="subgraph size", default=subgraph_size
        )
        parser.add_argument(
            "--n_order", type=int, help="order of neighbor nodes", default=n_order
        )
        parser.add_argument("--hidden_size", type=int, help="hidden size", default=1024)
        parser.add_argument(
            "--experiment_type",
            type=str,
            default="train",
            choices=["train", "cf", "test"],
        )  # train, cf, test

        args = parser.parse_known_args()[0]
        args.num_hidden = 101
        args.alpha = 14
        args.beta = 1
        args.acc = args.roc = acc

        nhid = args.num_hidden
        dropout = args.dropout
        # self.estimator = GCN(nfeat, 1, dropout) #nhid instead of 1
        self.estimator = GCN(nfeat, nhid, 1, dropout)
        self.GNN = get_model(nfeat, args)
        self.classifier = nn.Linear(nhid, 1)
        self.adv = nn.Linear(nhid, 1)

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
        print("sens: ", sens)
        print("labels: ", labels)
        print("output: ", output)
        print("idx: ", idx)
        val_y = labels[idx].cpu().numpy()
        idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
        idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

        idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

        pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
        print("output[idx] : ", output[idx])

        print("pred_y :", pred_y)
        print("pred_y[idx_s0] :", pred_y[idx_s0])

        print("sub(pred_y[idx_s0]) :", sum(pred_y[idx_s0]))
        print("sum(idx_s0): ", sum(idx_s0))
        print("sum(pred_y[idx_s1]):", sum(pred_y[idx_s1]))
        print("sum(idx_s1): ", sum(idx_s1))
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

    def optimize(self, g, x, labels, idx_train, sens, idx_sens_train, edge_index):
        self.train()

        ### update E, G
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        s = self.estimator(edge_index, x)
        h = self.GNN(edge_index, x)
        y = self.classifier(h)

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
        self.x = features
        self.labels = labels
        self.sens = sens

        self.edge_index = (
            torch.tensor(g.to_dense().nonzero(), dtype=torch.long).t().cuda()
        )
        self.val_loss = 0
        for epoch in tqdm(range(args.epochs)):
            t = time.time()
            self.train()
            self.optimize(
                g, features, labels, idx_train, sens, idx_sens_train, self.edge_index
            )
            self.eval()

            output, s = self(self.edge_index, features)
            acc_val = accuracy(output[idx_val], labels[idx_val])

            parity_val, equality_val = self.fair_metric(sens, labels, output, idx_val)

            # if acc_val > args.acc: #and roc_val > args.roc:
            print("acc_val: ", acc_val)
            print("parity val:", parity_val)
            print("equality_val: ", equality_val)
            print("best fair: ", best_fair)
            if acc_val > args.acc:
                if parity_val + equality_val < best_fair:
                    # if acc_val>best_acc:
                    
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
