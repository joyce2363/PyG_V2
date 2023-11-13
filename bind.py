import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import ipdb
import torch.nn as nn
from torch_geometric.nn import GCNConv
import scipy.sparse as sp
from torch_geometric.utils import convert

# GCN Module
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        return x
    

    # function returns the train/test/valid split
def load_pokec_renewed(dataset, label_number=1000):  # 1000

    if dataset == 2: #1 means for poekc_z is pokec_1, 2 is for pokec_n
        edges = np.load('/home/joyce/PyG_V2/region_job_2_2_edges.npy')
        features = np.load('/home/joyce/PyG_V2/region_job_2_2_features.npy')
        labels = np.load('/home/joyce/PyG_V2/region_job_2_2_labels.npy')
        sens = np.load('/home/joyce/PyG_V2/region_job_2_2_sens.npy')
    elif dataset == 1: 
        edges = np.load('/home/joyce/PyG_V2/region_job_1_edges.npy')
        features = np.load('/home/joyce/PyG_V2/region_job_1_features.npy')
        labels = np.load('/home/joyce/PyG_V2/region_job_1_labels.npy')
        sens = np.load('/home/joyce/PyG_V2/region_job_1_sens.npy')
   
    from scipy.sparse import csr_matrix
    print("edges.shape: ", edges.shape)
    sm = csr_matrix(edges)
    print("edges:", edges)
    print("Sparse Matrix: ", sm)
    print("try: ", len(set(sm.data)))
    # sm = csr_matrix(edges)
    # sm = np.array(sm)
    # print("sparse matrix: ", sm)
    # column_indices = [entry[1] for entry in sm]
    # unique_vertices = np.unique(column_indices)

    # # Count the number of unique vertices
    # num_vertices = len(unique_vertices)

    # print("Number of vertices:", num_vertices)
    # Convert the sparse matrix to a dense (full) matrix
    # dense_matrix = sm.toarray()

    # print(dense_matrix) 
   

                                  




    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    import random
    random.seed(20) # og random seed is 20
    # print("what are the label: ", labels)
    # print("label length: ", len(labels))
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    print("label_idx_0: ", label_idx_0)
    print("label_idx_1: ", label_idx_1)
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    # split train, val, test set
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])


    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # print("features: ", features)
    # print("idx_train: ", len(idx_train))
    # print("idx_val: ", len(idx_val))
    # print("idx_test: ", len(idx_test))
    return adj, features, labels, idx_train, idx_val, idx_test, sens

adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec_renewed(1)
# print("adj: ", adj)
# print("features: ", features)
# print("labels: ", labels)
edge_index = convert.from_scipy_sparse_matrix(adj)[0]

# calls GCN model (1 without MLP, 1 with MLP)
model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.unique().shape[0]-1, dropout=0.5)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training the model
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index)
    loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
    preds = (output.squeeze() > 0).type_as(labels)
    acc_train = accuracy_new(preds[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not False:
        model.eval()
        output = model(features, edge_index)

    loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
    acc_val = accuracy_new(preds[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item()

def tst():
    model.eval()
    output = model(features, edge_index)
    preds = (output.squeeze() > 0).type_as(labels)
    loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float())
    acc_test = accuracy_new(preds[idx_test], labels[idx_test])

    print("*****************  Cost  ********************")
    print("SP cost:")
    idx_sens_test = sens[idx_test]
    idx_output_test = output[idx_test]
    print(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))

    print("EO cost:")
    idx_sens_test = sens[idx_test][labels[idx_test]==1]
    idx_output_test = output[idx_test][labels[idx_test]==1]
    print(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))
    print("**********************************************")

    parity, equality = fair_metric(preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                   sens[idx_test].numpy())

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    print("Statistical Parity:  " + str(parity))
    print("Equality:  " + str(equality))

def accuracy_new(output, labels):
    correct = output.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

import time

t_total = time.time()
final_epochs = 0
loss_val_global = 1e10
dataset_name = "pokec_n"
starting = time.time()
for epoch in tqdm(range(1000)):
    # train
    loss_mid = train(epoch)
    if loss_mid < loss_val_global:
        loss_val_global = loss_mid
        # torch.save(model, 'gcn_' + dataset_name + '.pth')
        final_epochs = epoch

# torch.save(model, 'gcn_' + dataset_name + '.pth')

ending = time.time()
print("Time:", ending - starting, "s")
# model = torch.load('gcn_' + dataset_name + '.pth')

# test
tst()