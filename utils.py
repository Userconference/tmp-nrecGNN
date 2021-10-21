import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
from normalization import fetch_normalization, row_normalize
import sys
import networkx as nx
import os
import scipy.io as scio
import pickle
import math
import json
from networkx.readwrite import json_graph
import pdb
sys.setrecursionlimit(99999)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def preprocess_citation_graph(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj2 = features * features.T
    adj2 = adj_normalizer(adj2)
    features = row_normalize(features)
    return adj, features, adj2


def preprocess_citation_bigraph(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj_cn = features.T
    features = row_normalize(features)
    adj_cn = row_normalize(adj_cn)
    adj_nc = features
    return adj, features, adj_nc, adj_cn

def load_citation_gac(dataset_str="cora", semi=1):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        features = row_normalize(features)

    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        features = row_normalize(features)

    # porting to pytorch
    # features = torch.FloatTensor(np.array(features.todense())).float()

    features = sparse_mx_to_torch_sparse_tensor(features).float()

    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    if semi == 0:
        idx_all = list(range(labels.shape[0]))
        used_all = set(idx_train).union(set(idx_val)).union(set(idx_test))
        unused_all = list(set(idx_all).difference(used_all))
        idx_train = list(idx_train) + unused_all

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))

    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def create_sparse_eye_tensor(shape):
    row = np.array(range(shape[0])).astype(np.int64)
    col = np.array(range(shape[0])).astype(np.int64)
    value_ = np.ones(shape[0]).astype(float)
    indices = torch.from_numpy(np.vstack((row, col)))
    values = torch.from_numpy(value_)
    shape = torch.Size(shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citationANEmat_gac(dataset_str="BlogCatalog", semi_rate=0.1):
    data_file = 'data/{}/{}'.format('social', dataset_str) + '.mat'
    data = scio.loadmat(data_file)
    if dataset_str == 'ACM':
        features = data['Features']
    else:
        features = data['Attributes']
    labels = data['Label'].reshape(-1)
    adj = data['Network']
    # adj, features = preprocess_citation(adj, features, normalization)
    features = row_normalize(features)

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1

    train_idx_file = 'data/' + 'social' + '/' + dataset_str + '_train_{}'.format(semi_rate) + '.pickle'
    valid_idx_file = 'data/' + 'social' + '/' + dataset_str + '_valid_{}'.format(semi_rate) + '.pickle'
    test_idx_file = 'data/' + 'social' + '/' + dataset_str + '_test_{}'.format(semi_rate) + '.pickle'
    if os.path.isfile(train_idx_file):
        with open(test_idx_file, 'rb') as f:
            idx_test = pickle.load(f)
        with open(valid_idx_file, 'rb') as f:
            idx_val = pickle.load(f)
        with open(train_idx_file, 'rb') as f:
            idx_train = pickle.load(f)
    else:
        mask = np.unique(labels)
        label_count = [np.sum(labels == v) for v in mask]
        idx_train = []
        idx_val = []
        idx_test = []
        for i, v in enumerate(mask):
            cnt = label_count[i]
            idx_all = np.where(labels == v)[0]
            np.random.shuffle(idx_all)
            idx_all = idx_all.tolist()
            test_len = math.ceil(cnt * 0.2)
            valid_len = math.ceil(cnt * 0.2)
            train_len = math.ceil(cnt * semi_rate)
            idx_test.extend(idx_all[-test_len:])
            idx_val.extend(idx_all[-(test_len + valid_len):-test_len])
            train_len_ = min(train_len, cnt - test_len - valid_len)
            idx_train.extend(idx_all[:train_len_])

        idx_train = np.array(idx_train)
        idx_val = np.array(idx_val)
        idx_test = np.array(idx_test)

        with open(train_idx_file, 'wb') as pfile:
            pickle.dump(idx_train, pfile, pickle.HIGHEST_PROTOCOL)
        with open(test_idx_file, 'wb') as pfile:
            pickle.dump(idx_test, pfile, pickle.HIGHEST_PROTOCOL)
        with open(valid_idx_file, 'wb') as pfile:
            pickle.dump(idx_val, pfile, pickle.HIGHEST_PROTOCOL)
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    features = sparse_mx_to_torch_sparse_tensor(features).float()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(labels)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_webANEmat_gac(dataset_str="texas", semi=1, semi_rate=0.1):
    data_file = 'data/{}/{}'.format('web', dataset_str) + '.mat'
    file_train = 'data/{}/{}_train'.format('web', dataset_str) + '.pickle'
    file_valid = 'data/{}/{}_valid'.format('web', dataset_str) + '.pickle'
    file_test = 'data/{}/{}_test'.format('web', dataset_str) + '.pickle'
    data = scio.loadmat(data_file)
    features = data['Attributes']
    labels = data['Label'].reshape(-1)
    adj = data['Network']
    features = row_normalize(features)

    if (adj != adj.T).sum() != 0:
        adj = adj + adj.T
    if np.any(np.unique(adj[adj.nonzero()].A1) != 1):
        adj.data = np.ones_like(adj.data)

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1

    with open(file_test, 'rb') as f:
        idx_test = pickle.load(f)
    with open(file_valid, 'rb') as f:
        idx_val = pickle.load(f)
    with open(file_train, 'rb') as f:
        idx_train = pickle.load(f)
    if semi == 1:
        train_idx_file = 'data/' + 'web' + '/' + dataset_str + '_train_{}'.format(semi_rate) + '.pickle'
        valid_idx_file = 'data/' + 'web' + '/' + dataset_str + '_valid_{}'.format(semi_rate) + '.pickle'
        test_idx_file = 'data/' + 'web' + '/' + dataset_str + '_test_{}'.format(semi_rate) + '.pickle'
        if os.path.isfile(train_idx_file):
            with open(test_idx_file, 'rb') as f:
                idx_test = pickle.load(f)
            with open(valid_idx_file, 'rb') as f:
                idx_val = pickle.load(f)
            with open(train_idx_file, 'rb') as f:
                idx_train = pickle.load(f)
        else:
            mask = np.unique(labels)
            label_count = [np.sum(labels == v) for v in mask]
            idx_train = []
            idx_val = []
            idx_test = []
            for i, v in enumerate(mask):
                cnt = label_count[i]
                idx_all = np.where(labels == v)[0]
                np.random.shuffle(idx_all)
                idx_all = idx_all.tolist()
                test_len = math.ceil(cnt * 0.2)
                valid_len = math.ceil(cnt * 0.2)
                train_len = math.ceil(cnt * semi_rate)
                idx_test.extend(idx_all[-test_len:])
                idx_val.extend(idx_all[-(test_len + valid_len):-test_len])
                train_len_ = min(train_len, cnt - test_len - valid_len)
                idx_train.extend(idx_all[:train_len_])

            idx_train = np.array(idx_train)
            idx_val = np.array(idx_val)
            idx_test = np.array(idx_test)

            with open(train_idx_file, 'wb') as pfile:
                pickle.dump(idx_train, pfile, pickle.HIGHEST_PROTOCOL)
            with open(test_idx_file, 'wb') as pfile:
                pickle.dump(idx_test, pfile, pickle.HIGHEST_PROTOCOL)
            with open(valid_idx_file, 'wb') as pfile:
                pickle.dump(idx_val, pfile, pickle.HIGHEST_PROTOCOL)

    features = sparse_mx_to_torch_sparse_tensor(features).float()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(labels)

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_adj_with_ratio(adj, ratio):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    random_index = np.arange(0, edges.shape[0], 1)
    np.random.shuffle(random_index)
    if ratio == 0.0:
        train_edges = edges
    else:
        mask_num = int(edges.shape[0] * ratio)
        pre_index = random_index[0:-mask_num]
        train_edges = edges[pre_index]

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()