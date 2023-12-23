
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, DeepGraphInfomax, JumpingKnowledge

from sklearn.metrics import accuracy_score,roc_auc_score, recall_score,f1_score, precision_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from torch.nn.utils import spectral_norm
from torch_geometric.utils import dropout_adj, convert

import torch.nn.functional as F
import torch.optim as optim

import time
import argparse
import numpy as np
import scipy.sparse as sp

class Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Classifier, self).__init__()

        # Classifier projector
        self.fc1 = spectral_norm(nn.Linear(ft_in, nb_classes))

    def forward(self, seq):
        ret = self.fc1(seq)
        return ret

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5, nheads=1): 
        super(GAT, self).__init__()
        # nfeat = 96, # test
        # nhid = 16, # test
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
        print("CHECK THIS OUT: ", x.size())
        print("EDGE_INDEX: ", edge_index.size())
        x = self.conv1(x, edge_index)
        x = x.flatten(start_dim=1)  # Flatten node features across heads
        x = self.transition(x)
        return x 
    
class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(SAGE, self).__init__()

        # Implemented spectral_norm in the sage main file
        # ~/anaconda3/envs/PYTORCH/lib/python3.7/site-packages/torch_geometric/nn/conv/sage_conv.py
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
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
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        # x = self.conv2(x, edge_index)
        return x
    
class Encoder_DGI(nn.Module):
    def __init__(self, nfeat, nhid):
        super(Encoder_DGI, self).__init__()
        self.hidden_ch = nhid
        self.conv = spectral_norm(GCNConv(nfeat, self.hidden_ch))
        self.activation = nn.PReLU()

    def corruption(self, x, edge_index):
        # corrupted features are obtained by row-wise shuffling of the original features
        # corrupted graph consists of the same nodes but located in different places
        return x[torch.randperm(x.size(0))], edge_index

    def summary(self, z, *args, **kwargs):
        return torch.sigmoid(z.mean(dim=0))

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.activation(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                base_model="gat", k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model
        if self.base_model == 'gat':
            self.conv = GAT(in_channels, out_channels)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv(x, edge_index)
        return x


class GNN(torch.nn.Module):
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
            num_proj_hidden,
            lr,
            weight_decay, 
            sim_coeff,
            encoder="gat", 
            nclass=1, 
            device="cuda"
            ):
        super(GNN, self).__init__()        
        self.device = device
        self.edge_index = convert.from_scipy_sparse_matrix(sp.coo_matrix(adj.to_dense().numpy()))[0]
        
        # self.encoder = Encoder(input_size=features.shape[1], hidden_size=16, output_size = num_hidden).to(device)
        print("features.shape[1]:", features.shape[1])
        print("NUM_HIDDEN:", num_hidden)
        self.encoder = Encoder(in_channels=features.shape[1], out_channels=num_hidden, base_model=encoder).to(device)
        
        self.sim_coeff = sim_coeff
        #self.encoder = encoder
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.sens = sens
        # self.sens_idx = sens_idx
        self.drop_edge_rate_1=self.drop_edge_rate_2=0.5
        self.drop_feature_rate_1=self.drop_feature_rate_2=0.5

        # Projection
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_proj_hidden)),
            nn.BatchNorm1d(num_proj_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            spectral_norm(nn.Linear(num_proj_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden)
        )

        # Prediction
        self.fc3 = nn.Sequential(
            spectral_norm(nn.Linear(num_hidden, num_hidden)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc4 = spectral_norm(nn.Linear(num_hidden, num_hidden))

        # Classifier
        self.c1 = Classifier(ft_in=num_hidden, nb_classes=1)

        for m in self.modules():
            self.weights_init(m)

        par_1 = list(self.encoder.parameters()) + list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(
            self.fc3.parameters()) + list(self.fc4.parameters())
        par_2 = list(self.c1.parameters()) + list(self.encoder.parameters())
        self.optimizer_1 = optim.Adam(par_1, lr=lr, weight_decay=weight_decay)
        self.optimizer_2 = optim.Adam(par_2, lr=lr, weight_decay=weight_decay)
        self = self.to(device)

        self.features = features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.labels = self.labels.to(device)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor,
                    edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        return z

    def prediction(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        return z

    def classifier(self, z):
        return self.c1(z)

    def normalize(self, x):
        val = torch.norm(x, p=2, dim=1).detach()
        x = x.div(val.unsqueeze(dim=1).expand_as(x))
        return x

    def D_entropy(self, x1, x2):
        x2 = x2.detach()
        return (-torch.max(F.softmax(x2), dim=1)[0]*torch.log(torch.max(F.softmax(x1), dim=1)[0])).mean()

    def D(self, x1, x2): # negative cosine similarity
        return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, e_1, e_2, idx):

        # projector
        p1 = self.projection(z1)
        p2 = self.projection(z2)

        # predictor
        h1 = self.prediction(p1)
        h2 = self.prediction(p2)

        # classifier
        c1 = self.classifier(z1)

        l1 = self.D(h1[idx], p2[idx])/2
        l2 = self.D(h2[idx], p1[idx])/2
        l3 = F.cross_entropy(c1[idx], z3[idx].squeeze().long().detach())

        return self.sim_coeff*(l1+l2), l3



    def forwarding_predict(self, emb):

        # classifier
        c1 = self.classifier(emb)

        return c1



    def fit(self, epochs=1000):
        best_loss = 100
        for epoch in range(epochs + 1):
            sim_loss = 0
            self.train() # what is self.train() 
            # print("self.train() ", self.train())
            self.optimizer_2.zero_grad() #what is optimizer ?? 
            edge_index_1 =self.edge_index # what are all of this for?!?
            x_1 = self.features


            # classifier
            z1 = self.forward(x_1, edge_index_1)
            print("X_1:", x_1)
            print("edge_index_1:", edge_index_1)
            c1 = self.classifier(z1)

            # Binary Cross-Entropy
            cl_loss = F.binary_cross_entropy_with_logits(c1[self.idx_train],
                                                    self.labels[self.idx_train].unsqueeze(1).float().to(self.device))

            cl_loss.backward()
            self.optimizer_2.step()


            # Validation
            self.eval()
            z_val = self.forward(self.features, self.edge_index)
            c_val = self.classifier(z_val)
            val_loss = F.binary_cross_entropy_with_logits(c_val[self.idx_val],
                                                    self.labels[self.idx_val].unsqueeze(1).float().to(self.device))
            # print("understand WHY the validation loss is SO bad.")
            # print("c_val[self.idx_val]", c_val[self.idx_val])
            # print("self.labels[self.idx_val].unsqueeze(1).float().to(self.device)", 
            #     self.labels[self.idx_val].unsqueeze(1).float().to(self.device)) #I believe that these should all be the labels
            if epoch % 100 == 0:
                print(f"[Train] Epoch {epoch}: train_c_loss: {cl_loss:.4f} | val_c_loss: {val_loss:.4f}")
            # print("VAL_LOSS: ", val_loss)
            # print("BEST_LOSS: ", best_loss)
            if (val_loss) < best_loss:
                # print("ENTERED!")
                self.val_loss=val_loss.item()

                best_loss = val_loss
                # if self.encoder == "sage": 
                torch.save(self.state_dict(), f'weights_GNN_{"gat"}.pt')
                # else: 
                #     torch.save(self.state_dict(), f'weights_GNN_{self.encoder}.pt')



    def predict(self):
        # if self.encoder == "sage": 
        self.load_state_dict(torch.load(f'weights_GNN_{"gat"}.pt'))
        # else: 
            # self.load_state_dict(torch.load(f'weights_GNN_{self.encoder}.pt'))
        # self.load_state_dict(torch.load(f'weights_GNN_{self.encoder}.pt'))

        self.eval()
        emb = self.forward(self.features.to(self.device), self.edge_index.to(self.device))
        output = self.forwarding_predict(emb)

        output_preds = (output.squeeze() > 0).type_as(self.labels)[self.idx_test].detach().cpu().numpy()
        print("output_preds:", output_preds)
        print("len(output_preds): ", len(output_preds))
        labels = self.labels.detach().cpu().numpy()
        idx_test = self.idx_test

        F1 = f1_score(labels[idx_test], output_preds, average='micro')
        recall = recall_score(labels[idx_test], output_preds, average='micro')
        precision = precision_score(labels[idx_test], output_preds, average='micro')
        cm = confusion_matrix(labels[idx_test], output_preds)

        ACC = accuracy_score(labels[idx_test], output_preds, )
        # AUCROC = roc_auc_score(labels[idx_test], output_preds)

        ACC_sens0, F1_sens0, ACC_sens1, F1_sens1 = self.predict_sens_group(output_preds, idx_test)

        SP, EO = self.fair_metric(output_preds, self.labels[idx_test].detach().cpu().numpy(),
                                  self.sens[idx_test].detach().cpu().numpy())


        return (
                ACC, 
                # AUCROC, 
                F1, 
                recall, 
                precision,
                cm,
                ACC_sens0, 
                # AUCROC_sens0, 
                F1_sens0, 
                ACC_sens1, 
                # AUCROC_sens1, 
                F1_sens1, 
                SP, 
                EO
        )








    def fair_metric(self, pred, labels, sens):

        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
        parity = abs(sum(pred[idx_s0]) / sum(idx_s0) -
                     sum(pred[idx_s1]) / sum(idx_s1))
        equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) -
                       sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
        return parity.item(), equality.item()

    def predict_sens_group(self, output, idx_test):
        #pred = self.lgreg.predict(self.embs[idx_test])
        pred=output
        result=[]
        
        for sens in [0,1]:
            F1 = f1_score(self.labels[idx_test][self.sens[idx_test]==sens].detach().cpu().numpy(), pred[self.sens[idx_test]==sens], average='micro')
            ACC=accuracy_score(self.labels[idx_test][self.sens[idx_test]==sens].detach().cpu().numpy(), pred[self.sens[idx_test]==sens],)
            # AUCROC=roc_auc_score(self.labels[idx_test][self.sens[idx_test]==sens].detach().cpu().numpy(), 
            #                      pred[self.sens[idx_test]==sens], 
            #                      multi_class="ovr")
            result.extend([ACC, 
                        #    AUCROC, 
                           F1])
        print("result in predict_sens_group: ", result)
        return result

def drop_feature(x, drop_prob, sens_idx, sens_flag=True):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob

    x = x.clone()
    drop_mask[sens_idx] = False

    x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    # Flip sensitive attribute
    if sens_flag:
        x[:, sens_idx] = 1-x[:, sens_idx]

    return x