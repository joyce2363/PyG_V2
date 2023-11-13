import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import os
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import scipy.sparse as sp
from torch_geometric.utils import convert
from torch_geometric.nn import GCNConv

import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import argparse
import time
import torch
from tqdm import tqdm
from torch_geometric.nn import GCNConv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import zipfile
import io
import gdown

from pygdebias.debiasing import GNN


dataset_name = "pokec_z" # set args later
if dataset_name == "pokec_z" : 
  dataset = "region_job"
  sens_attr = "region"
  predict_attr = "I_am_working_in_field"
  label_number = 500
  sens_number = 200
  seed = 20
  path = "/content/" # /content/region_job_relationship.txt
  test_idx = False

def load_pokec(self, dataset, sens_attr, predict_attr = "I_am_working_in_field", path="/content/", label_number=1000, sens_number=500, seed=19, test_idx=False):

        idx_features_labels = pd.read_csv("/home/joyce/PyG_V2/dataset/pokec_z/region_job.csv")
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        # header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("/home/joyce/PyG_V2/dataset/pokec_z/region_job_relationship.txt", dtype=np.int64)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), 
                         dtype=np.int64).reshape(edges_unordered.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = mx_to_torch_sparse_tensor(adj)

        import random
        random.seed(seed)
        print("labels length: ", len(labels))
        label_idx_0 = np.where(labels == 0)[0]
        print("label_idx_0: ", len(label_idx_0))
        label_idx_1 = np.where(labels == 1)[0]
        print("label_idx_1: ", len(label_idx_1))

        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])


        sens = idx_features_labels[sens_attr]

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)
        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        # random.shuffle(sens_idx)

        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train

adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                       sens_attr, 
                                                                                       predict_attr,
                                                                                       path = path,
                                                                                       label_number = 1000,
                                                                                       sens_number = 500, 
                                                                                       seed = 19, 
                                                                                       test_idx = False, 
                                                                                       )
# print("adj: ", adj)
print("features: ", features)
# print("labels: ", labels)
# model = GNN(adj, features, labels, idx_train, idx_val, idx_test, sens, idx_train) #features.shape[1]