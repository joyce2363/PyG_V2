import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import os
import dgl
import networkx as nx
from typing import Dict, Tuple
import pickle
from os.path import join, dirname, realpath
import csv
import pickle as pkl
import requests
import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import zipfile
import io
import gdown

"""
NotLoaded: LCC, Filmtrust, Lastfm, UNC, oklahoma
"""


import requests


class Dataset(object):
    def __init__(self, seed, root: str = "./dataset") -> None:
        self.adj_ = None
        self.features_ = None
        self.labels_ = None
        self.idx_train_ = None
        self.idx_val_ = None
        self.idx_test_ = None
        self.sens_ = None
        self.sens_idx_ = None
        self.seed_ = seed
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.path_name = ""

    def download(self, url: str, filename: str):
        r = requests.get(url)
        assert r.status_code == 200
        open(os.path.join(self.root, self.path_name, filename), "wb").write(r.content)

    def download_zip(self, url: str):
        r = requests.get(url)
        assert r.status_code == 200
        foofile = zipfile.ZipFile(io.BytesIO(r.content))
        foofile.extractall(os.path.join(self.root, self.path_name))

    def adj(self, datatype: str = "torch.sparse"):
        assert str(type(self.adj_)) == "<class 'torch.Tensor'>"
        if self.adj_ is None:
            return self.adj_
        if datatype == "torch.sparse":
            return self.adj_
        elif datatype == "scipy.sparse":
            return sp.coo_matrix(self.adj.to_dense())
        elif datatype == "np.array":
            return self.adj_.to_dense().numpy()
        else:
            raise ValueError(
                "datatype should be torch.sparse, tf.sparse, np.array, or scipy.sparse"
            )

    def features(self, datatype: str = "torch.tensor"):
        if self.features is None:
            return self.features_
        if datatype == "torch.tensor":
            return self.features_
        elif datatype == "np.array":
            return self.features_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def labels(self, datatype: str = "torch.tensor"):
        if self.labels_ is None:
            return self.labels_
        if datatype == "torch.tensor":
            return self.labels_
        elif datatype == "np.array":
            return self.labels_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_val(self, datatype: str = "torch.tensor"):
        if self.idx_val_ is None:
            return self.idx_val_
        if datatype == "torch.tensor":
            return self.idx_val_
        elif datatype == "np.array":
            return self.idx_val_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_train(self, datatype: str = "torch.tensor"):
        if self.idx_train_ is None:
            return self.idx_train_
        if datatype == "torch.tensor":
            return self.idx_train_
        elif datatype == "np.array":
            return self.idx_train_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def idx_test(self, datatype: str = "torch.tensor"):
        if self.idx_test_ is None:
            return self.idx_test_
        if datatype == "torch.tensor":
            return self.idx_test_
        elif datatype == "np.array":
            return self.idx_test_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def sens(self, datatype: str = "torch.tensor"):
        if self.sens_ is None:
            return self.sens_
        if datatype == "torch.tensor":
            return self.sens_
        elif datatype == "np.array":
            return self.sens_.numpy()
        else:
            raise ValueError("datatype should be torch.tensor, tf.tensor, or np.array")

    def sens_idx(self):
        return self.sens_idx_
    
    def seed(self, datatype: int):
        if self.seed_:
            return self.seed_
def mx_to_torch_sparse_tensor(sparse_mx, is_sparse=False, return_tensor_sparse=True):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not is_sparse:
        sparse_mx = sp.coo_matrix(sparse_mx)
    else:
        sparse_mx = sparse_mx.tocoo()
    if not return_tensor_sparse:
        return sparse_mx

    sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class Nba(Dataset):
    def __init__(
        self, 
        seed,
        dataset_name="nba", 
        predict_attr_specify=None, 
        return_tensor_sparse=True
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        super(Nba, self).__init__(seed)
        (
            adj, 
            features, 
            labels, 
            idx_train, 
            idx_val, 
            idx_test, 
            sens, 
            idx_sens_train, 
            sens_idx
        ) = self.load_pokec(
            dataset = "nba",
            sens_attr = "country",
            predict_attr = "SALARY",
            path= "./dataset/NBA",
            label_number=100,
            sens_number=50,
            seed=seed,
            test_idx = True,
        )

        # adj=adj.todense()
        adj = mx_to_torch_sparse_tensor(
            adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse
        )

        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = -1

    def load_pokec(
        self,
        dataset,
        sens_attr,
        predict_attr,
        label_number,
        sens_number,
        seed,
        path="../dataset/pokec/",
        test_idx=False,
    ):
        """Load data"""

        self.path_name = "nba"
        # if not os.path.exists(os.path.join(self.root, self.path_name)):
        #     os.makedirs(os.path.join(self.root, self.path_name))
        # if not os.path.exists(os.path.join(self.root, self.path_name, "nba.csv")):
        #     url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/NBA/nba.csv"
        #     filename = "nba.csv"
        #     self.download(url, filename)
        # if not os.path.exists(
        #     os.path.join(self.root, self.path_name, "nba_relationship.txt")
        # ):
        #     url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/NBA/nba_relationship.txt"
        #     filename = "nba_relationship.txt"
        #     self.download(url, filename)

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "nba.csv")
        )
        # print("IDX_FEATURES_LABELS NBA: ", idx_features_labels) # this should be the same
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(sens_attr)
        header.remove(predict_attr)

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, "nba_relationship.txt"),
            dtype=np.int64,
        )

        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = mx_to_torch_sparse_tensor(adj)

        random.seed(seed)
        label_idx = np.where(labels >= 0)[0]
        random.shuffle(label_idx)

        # with open("label_idx" + dataset + "_" + str(seed)+ ".pickle", "wb") as handle:
        #     pickle.dump(label_idx, handle)

        with open("label_idx" + dataset + "_" + str(seed)+ ".pickle", "rb") as handle:
            label_idx = pickle.load(handle) 
        # print("label_idx: ", label_idx)

        idx_train = label_idx[: min(int(0.5 * len(label_idx)), label_number)]
        print("LENGTH OF IDX_TRAIN:", len(idx_train))
        idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
        if test_idx:
            idx_test = label_idx[label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.75 * len(label_idx)) :]

        sens = idx_features_labels[sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        random.seed(seed)
        random.shuffle(idx_sens_train)

        # with open("idx_sens_train_" + dataset + "_" + str(seed)+ ".pickle", "wb") as handle:
        #     pickle.dump(idx_sens_train, handle)

        with open("idx_sens_train_" + dataset + "_" + str(seed)+ ".pickle", "rb") as handle:
            idx_sens_train = pickle.load(handle) 
        # print("idx_sens_train: ", idx_sens_train)

        idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        features = torch.cat([features, sens.unsqueeze(-1)], -1)
        print("IDX_SENS_TRAIN: ", idx_sens_train)
        print("IDX_VAL: ", idx_val)
        print("IDX_TEST: ", idx_test)
        return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train, sens_idx

class Pokec_z(Dataset):
    def __init__(
        self,
        seed,
        dataset_name="pokec_z",
        predict_attr_specify=None,
        return_tensor_sparse=True,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
        super(Pokec_z, self).__init__(seed)
        (
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test,
            sens,
            # idx_sens_train,
        ) = self.load_pokec(
            dataset="pokec_z",
            sens_attr="region",
            predict_attr="I_am_working_in_field",
            path="./dataset/pokec/",
            label_number= 1000, #bail
            sens_number= 200,
            seed=seed,
            test_idx=False,
        )

        # adj=adj.todense(
        adj = mx_to_torch_sparse_tensor(
            adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse
        )

        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens

    def load_pokec(
        self,
        dataset,
        seed,
        sens_attr,
        predict_attr,
        path="../dataset/pokec/",
        label_number=1000, #500
        sens_number=200,
        test_idx=False,
    ):
        """Load data"""
        edges = np.load('/home/joyce/region_job_1_edges.npy')
        features = np.load('/home/joyce/region_job_1_features.npy')
        labels = np.load('/home/joyce/region_job_1_labels.npy')
        sens = np.load('/home/joyce/region_job_1_sens.npy')
    
        # idx_features_labels = pd.read_csv(
        #     os.path.join(self.root, self.path_name, "region_job.csv")
        # )
        # header = list(idx_features_labels.columns)
        # header.remove("user_id")

        # header.remove(sens_attr)
        # header.remove(predict_attr)

        # features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        # labels = idx_features_labels[predict_attr].values

        # build graph
        # idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        # idx_map = {j: i for i, j in enumerate(idx)}
        # edges_unordered = np.genfromtxt(
        #     os.path.join(self.root, self.path_name, "region_job_relationship.txt"),
        #     dtype=np.int64,
        # )

        # edges = np.array(
        #     list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
        # ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        # features = torch.FloatTensor(np.array(features.todense()))
        # labels = torch.LongTensor(labels)
        # adj = mx_to_torch_sparse_tensor(adj)
        features = torch.FloatTensor(np.array(features))
        labels = torch.LongTensor(labels)
        sens = torch.LongTensor(sens)

        random.seed(seed)
        # label_idx = np.where(labels >= 0)[0]
        # random.shuffle(label_idx)

        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        dataset = str(1)
        with open("label_idx_0_" + dataset + "_" + str(seed)+ ".pickle", "rb") as handle:
            label_idx_0 = pickle.load(handle)
        with open("label_idx_1_" + dataset + "_" + str(seed)+ ".pickle", "rb") as handle:
            label_idx_1 = pickle.load(handle) 
        print("label_idx_0: ", label_idx_0)
        print("label_idx_1: ", label_idx_1)

        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        print("len(idx_train):", len(idx_train))
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        print("len(idx_val):", len(idx_val))
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
        print("len(idx_test):", len(idx_test))


        # idx_train = label_idx[: min(int(0.5 * len(label_idx)), label_number)]
        # idx_val = label_idx[int(0.5 * len(label_idx)) : int(0.75 * len(label_idx))]
        # if test_idx:
        #     idx_test = label_idx[label_number:]
        #     idx_val = idx_test
        # else:
        #     idx_test = label_idx[int(0.75 * len(label_idx)) :]

        # sens = idx_features_labels[sens_attr].values

        # sens_idx = set(np.where(sens >= 0)[0])
        # idx_test = np.asarray(list(sens_idx & set(idx_test)))

        # idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        # random.seed(seed)
        # random.shuffle(idx_sens_train)
        # idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        # features = torch.cat([features, sens.unsqueeze(-1)], -1)
        # random.shuffle(sens_idx)

        return adj, features, labels, idx_train, idx_val, idx_test, sens
    # , idx_sens_train


class Pokec_n(Dataset):
    def __init__(
        self,
        seed,
        dataset_name="pokec_n",
        predict_attr_specify=None,
        return_tensor_sparse=True,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        super(Pokec_n, self).__init__(seed)
        (adj, 
        features, 
        labels, 
        idx_train, 
        idx_val, 
        idx_test, 
        sens, 
        # idx_sens_train,
        ) = self.load_pokec( 
            dataset = "region_job_2",
            sens_attr = "region",
            predict_attr = "I_am_working_in_field",
            path = 'none',
            label_number=1000, #500
            sens_number = 200,
            seed = seed,
            test_idx= False,
        )

        adj = mx_to_torch_sparse_tensor(
            adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse
        )

        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens

    def load_pokec(
        self,
        dataset,
        seed,
        sens_attr,
        predict_attr,
        path="../dataset/pokec/",
        label_number=1000, #500
        sens_number=200,
        test_idx=False,
    ):
        """Load data"""
        edges = np.load('/home/joyce/region_job_2_2_edges.npy')
        features = np.load('/home/joyce/region_job_2_2_features.npy')
        labels = np.load('/home/joyce/region_job_2_2_labels.npy')
        sens = np.load('/home/joyce/region_job_2_2_sens.npy')

        # self.path_name = "pokec_n"
        # self.destination = os.path.join(self.root, self.path_name, "pokec_n.zip")
        # idx_features_labels = pd.read_csv(
        #     os.path.join(self.root, self.path_name, "region_job_2.csv")
        # )
        # header = list(idx_features_labels.columns)
        # header.remove("user_id")
        # header.remove(sens_attr)
        # header.remove(predict_attr)

        # features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        # labels = idx_features_labels[predict_attr].values

        # build graph
        # idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
        # idx_map = {j: i for i, j in enumerate(idx)}
        # edges_unordered = np.genfromtxt(
        #     os.path.join(self.root, self.path_name, "region_job_2_relationship.txt"),
        #     dtype=np.int64,
        # )

        # edges = np.array(
        #     list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
        # ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        # features = torch.FloatTensor(np.array(features.todense()))
        features = torch.FloatTensor(np.array(features))
        labels = torch.LongTensor(labels)
        sens = torch.LongTensor(sens)

        # adj = mx_to_torch_sparse_tensor(adj)

        import random

        random.seed(seed)
        label_idx_0 = np.where(labels == 0)[0]
        # print("label_idx_0: ", len(label_idx_0))
        label_idx_1 = np.where(labels == 1)[0]
        # print("label_idx_1: ", len(label_idx_1))

        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        dataset = str(2)
        with open("label_idx_0_" + dataset + "_" + str(seed)+ ".pickle", "rb") as handle:
            label_idx_0 = pickle.load(handle)
        with open("label_idx_1_" + dataset + "_" + str(seed)+ ".pickle", "rb") as handle:
            label_idx_1 = pickle.load(handle) 
        print("LENGTH OF label_idx_0: ", len(label_idx_0))
        print("LENGTH OF label_idx_1: ", len(label_idx_1))

        # idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
        #                   label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        # idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        #                 label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        # idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        print("(int(0.5 * len(label_idx_0)):", (int(0.5 * len(label_idx_0))))
        print("len(idx_train):", len(idx_train))
        curr_min = min(int(0.5 * len(label_idx_0)), label_number // 2)
        if curr_min == int(0.5 * len(label_idx_0)): 
            print("MIN: int(0.5 * len(label_idx_0)) :", len(int(0.5 * len(label_idx_0))))
        elif curr_min == label_number // 2:
            print("MIN: label_number // 2 :", (len(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)])))
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        print("len(idx_val):", len(idx_val))
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
        print("len(idx_test):", len(idx_test))
        # sens = idx_features_labels[sens_attr].values
        print("length of all splits combined: ", len(idx_test) + len(idx_train) + len(idx_val))
        sens_idx = set(np.where(sens >= 0)[0])
        # idx_test = np.asarray(list(sens_idx & set(idx_test)))
        # sens = torch.FloatTensor(sens)
        # idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
        # random.seed(seed)
        # random.shuffle(idx_sens_train)
        # idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
        # features = torch.cat([features, sens.unsqueeze(-1)], -1)
        # features = torch.FloatTensor(np.array(features))
        # labels = torch.LongTensor(labels)
        # sens = torch.LongTensor(sens)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        # random.shuffle(sens_idx)
        print("features: ", features)
        return adj, features, labels, idx_train, idx_val, idx_test, sens
    # , idx_sens_train

class Bail(Dataset):
    def __init__(self,
                 seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        super(Bail, self).__init__(seed)
        (
            adj,
            features,
            labels,
            edges,
            sens,
            idx_train,
            idx_val,
            idx_test,
            sens_idx,
        ) = self.load_bail("bail", seed)
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        node_num = features.shape[0]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        features = self.feature_norm(features)
        adj = mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = sens_idx

    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    def load_bail(
        self,
        dataset,
        seed,
        sens_attr="WHITE",
        predict_attr="RECID",
        path="./dataset/bail/",
        label_number=1000,
    ):
        # print('Loading {} dataset from {}'.format(dataset, path))
        self.path_name = "bail"
        # if not os.path.exists(os.path.join(self.root, self.path_name)):
        #     os.makedirs(os.path.join(self.root, self.path_name))

        # if not os.path.exists(os.path.join(self.root, self.path_name, "bail.csv")):
        #     url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/bail/bail.csv"
        #     file_name = "bail.csv"
        #     self.download(url, file_name)
        # if not os.path.exists(
        #     os.path.join(self.root, self.path_name, "bail_edges.txt")
        # ):
        #     url = "https://raw.githubusercontent.com/PyGDebias-Team/data/main/2023-7-26/bail/bail_edges.txt"
        #     file_name = "bail_edges.txt"
        #     self.download(url, file_name)

        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "{}.csv".format(dataset))
        )
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)

        # build relationship

        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, f"{dataset}_edges.txt")
        ).astype("int")

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=int
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        import random

        random.seed(seed)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        with open("label_idx_0_" + dataset + "_" + str(seed)+ ".pickle", "rb") as handle:
            label_idx_0 = pickle.load(handle)
        with open("label_idx_1_" + dataset + "_" + str(seed)+ ".pickle", "rb") as handle:
            label_idx_1 = pickle.load(handle) 
        print("label_idx_0: ", label_idx_0)
        print("label_idx_1: ", label_idx_1)

        idx_train = np.append(
            label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
            label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
        )
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
        )
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 0
    


class Income(Dataset):
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        super(Income, self).__init__(seed)
        (
            adj,
            features,
            labels,
            edges,
            sens,
            idx_train,
            idx_val,
            idx_test,
            sens_idx,
        ) = self.load_income("income", seed)
        seed = seed
        
        node_num = features.shape[0]
        self.seed = seed
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        features = self.feature_norm(features)
        adj = mx_to_torch_sparse_tensor(adj, is_sparse=True)
        self.adj_ = adj
        self.features_ = features
        self.labels_ = labels
        self.idx_train_ = idx_train
        self.idx_val_ = idx_val
        self.idx_test_ = idx_test
        self.sens_ = sens
        self.sens_idx_ = sens_idx
    def feature_norm(self, features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2 * (features - min_values).div(max_values - min_values) - 1

    def load_income(
        self,
        dataset,
        seed,
        sens_attr="race",
        predict_attr="income",
        path="empty",
        label_number=1000,
    ):
        print('Loading {} dataset from {}'.format(dataset, path))
        self.path_name = "income"
        idx_features_labels = pd.read_csv(
            os.path.join(self.root, self.path_name, "{}.csv".format(dataset))
        )
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)

        # build relationship

        edges_unordered = np.genfromtxt(
            os.path.join(self.root, self.path_name, f"{dataset}_edges.txt")
        ).astype("int")

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=int
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)

        # import random
        # label_idx_0 = np.where(labels == 0)[0]
        # label_idx_1 = np.where(labels == 1)[0]
        # random.shuffle(label_idx_0)
        # random.shuffle(label_idx_1)
        # /home/joyce/PyG_V2/pygdebias/datasets/label_idx_0_1_1.pickle
        print("seed: ", seed)
        # with open("label_idx_0_" + dataset + "_" + str(seed) + ".pickle", "wb") as handle:
        #     pickle.dump(label_idx_0, handle)
        # with open("label_idx_1_" + dataset + "_" + str(seed) + ".pickle", "wb") as handle:
        #     pickle.dump(label_idx_1, handle)

        with open("label_idx_0_" + dataset + "_" + str(seed)+ ".pickle", "rb") as handle:
            label_idx_0 = pickle.load(handle)
        with open("label_idx_1_" + dataset + "_" + str(seed)+ ".pickle", "rb") as handle:
            label_idx_1 = pickle.load(handle) 
        print("label_idx_0: ", label_idx_0)
        print("label_idx_1: ", label_idx_1)
        idx_train = np.append(
            label_idx_0[: min(int(0.5 * len(label_idx_0)), label_number // 2)],
            label_idx_1[: min(int(0.5 * len(label_idx_1)), label_number // 2)],
        )
        idx_val = np.append(
            label_idx_0[int(0.5 * len(label_idx_0)) : int(0.75 * len(label_idx_0))],
            label_idx_1[int(0.5 * len(label_idx_1)) : int(0.75 * len(label_idx_1))],
        )
        idx_test = np.append(
            label_idx_0[int(0.75 * len(label_idx_0)) :],
            label_idx_1[int(0.75 * len(label_idx_1)) :],
        )

        sens = idx_features_labels[sens_attr].values.astype(int)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test, 0
