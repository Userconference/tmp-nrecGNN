from __future__ import division
from __future__ import print_function
from sklearn.metrics import f1_score
import time
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from normalization import row_normalize
from utils import sparse_mx_to_torch_sparse_tensor, accuracy, load_webANEmat_gac, load_citation_gac, load_citationANEmat_gac
from models import nrecGNN
import os
import pickle

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='6', help='specify cuda devices')
parser.add_argument('--model_type', type=str, default='nrecgnn')
parser.add_argument('--dataset', type=str, default="citeseer",
                    help='squirrel.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--semi', type=int, default=0, help='1|0 supervsied/semi-supervised')
parser.add_argument('--semi_rate', type=float, default=0.6, help='semi ratio for supervised learning')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--share_attn', type=int, default=1,
                    help='indicator for whether share weight for all layers or not')
parser.add_argument('--times', type=int, default=5)
parser.add_argument('--hidden_state', type=int, default=1024)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--nlayer', type=int, default=2,
                    help='Number of layers')
parser.add_argument('--act', type=str, default='relu', help='activation function relu|prelu')
parser.add_argument('--dropnode', type=float, default=.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--drop_edge', type=float, default=.6,
                    help='Dropout rate (1 - keep probability).')


def evaluate_acc(model, features, adj, idx):
    model.eval()
    output = model(features, adj, idx)
    acc_test = accuracy(output, labels[idx])

    preds = output.max(1)[1].type_as(labels)
    preds = preds.data.cpu().numpy().reshape(-1)
    test_y = labels[idx].data.cpu().numpy().reshape(-1)
    micro = f1_score(test_y, preds, average='micro')
    macro = f1_score(test_y, preds, average='macro')
    return acc_test.item(), micro, macro


def train(features, adj, labels, idx_train, idx_val, idx_test, args, save_path, device):
    model = nrecGNN(nfeat=features.shape[1],
                n_hops=args.nlayer,
                act=args.act,
                hidden_state=args.hidden_state,
                share_attn=args.share_attn,
                nclass=labels.max().item() + 1,
                dropnode=args.dropnode,
                drop_edge=args.drop_edge,
                     )
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    model.to(device)
    features = features.to_dense().to(device)
    adj_valid = adj_return(adj, idx_val, device)
    adj_test = adj_return(adj, idx_test, device)
    adj_train = adj_return(adj, idx_train, device)
    labels = labels.to(device)
    idx_train_gpu = idx_train.to(device)
    idx_val_gpu = idx_val.to(device)
    idx_test_gpu = idx_test.to(device)

    max_acc = 0.
    best_epoch = 0
    cnt_wait = 0
    for epoch in range(args.epochs):
        t = time.time()
        model.train()

        output = model(features, adj_train, idx_train_gpu)
        loss_train = F.nll_loss(output, labels[idx_train_gpu])
        acc_train = accuracy(output, labels[idx_train_gpu])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        acc_val, f1_mic_val, f1_mac_val = evaluate_acc(model, features, adj_valid, idx_val_gpu)

        if max_acc < acc_val:
            max_acc = acc_val
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            cnt_wait += 1

        print('Epoch %d / %d' % (epoch, args.epochs),
              'current_best_epoch: %d' % best_epoch,
              'train_loss: %.4f' % (loss_train.item()),
              'train_acc: %.4f' % (acc_train.item()),
              'acc_val: %.4f' % acc_val,
              'f1_mic_val: %.4f' % f1_mic_val,
              'f1_mac_val: %.4f' % f1_mac_val,
              'time: {:.4f}s'.format(time.time() - t))

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    print('---> Train ends with best epoch: %d and best valid_acc: %.4f' % (best_epoch, max_acc))
    model.load_state_dict(torch.load(save_path))
    acc_test, f1_mic_test, f1_mac_test = evaluate_acc(model, features, adj_test, idx_test_gpu)

    print('!!! Training finished',
          'best_epoch: %d' % best_epoch,
          'test_acc: %.4f' % acc_test,
          'test_f1_mic: %.4f' % f1_mic_test,
          'test_acc: %.4f' % f1_mac_test,
          )
    return acc_test, f1_mic_test, f1_mac_test

# def get_hops(adj, n_hops):
#     # adj is csr_matrix, n_hops
#     n_node, _ = adj.shape
#     adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape, dtype=float)
#     adj_orig = adj
#     adj_result = []
#     for i in range(n_hops):
#         if i == 0:
#             adj_ = adj_orig.tocoo()
#         else:
#             adj = sp.csr_matrix(adj.dot(adj_orig.toarray().T))
#             adj_ = adj
#         print('---> Sparse rate of %d is : %.4f' % (i + 1, adj_.nnz / n_node / n_node))
#         adj_ = row_normalize(adj_)
#         adj_result.append(adj_)
#     return adj_result
def get_hops(adj, n_hops, args):
    # adj is csr_matrix, n_hops
    hop_file = 'hop_adj/' + args.dataset + '_hop_{}'.format(args.nlayer) + '.pickle'

    if os.path.isfile(hop_file):
        with open(hop_file, 'rb') as f:
            adj_result = pickle.load(f)
    else:
        n_node, _ = adj.shape
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape, dtype=float)
        adj_orig = adj
        adj_result = []
        for i in range(n_hops):
            if i == 0:
                adj_ = adj_orig.tocoo()
            else:
                adj = sp.csr_matrix(adj.dot(adj_orig.toarray().T))
                adj_ = adj
            print('---> Sparse rate of %d is : %.4f' % (i + 1, adj_.nnz / n_node / n_node))
            adj_ = row_normalize(adj_)
            adj_result.append(adj_)

        with open(hop_file, 'wb') as pfile:
            pickle.dump(adj_result, pfile, pickle.HIGHEST_PROTOCOL)
    return adj_result

def adj_return(adj, idx, device):
    # adj is csr_matrix, n_hops
    adj_result = [sparse_mx_to_torch_sparse_tensor(adj_[idx]).to(device) for adj_ in adj]
    return adj_result


def print_configuration(args):
    print('--> Experiment configuration')
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))


def main(features, adj, labels, idx_train, idx_val, idx_test, args, device):
    result_acc = []
    result_f1_mic = []
    result_f1_mac = []

    for t in range(args.times):
        print('----> Start run %s %d / %d times' % (args.model_type, t, args.times))
        save_path = "./weights/%s-" % args.model_type + args.dataset + '-%s_' % args.act \
                    + '%d_' % args.hidden_state + '%d' % args.nlayer + '_%d' % args.share_attn\
                    + '_{}'.format(args.dropnode) + '_{}_'.format(args.drop_edge) + '%d' % t + '.pth'
        t1 = time.time()
        acc_test, f1_mic_test, f1_mac_test = train(features, adj, labels, idx_train, idx_val, idx_test, args, save_path, device)
        t2 = time.time()
        print('---> finish on run with time {}'.format(t2- t1))
        result_acc.append(acc_test)
        result_f1_mic.append(f1_mic_test)
        result_f1_mac.append(f1_mac_test)

    result_acc = np.array(result_acc)
    result_f1_mic = np.array(result_f1_mic)
    result_f1_mac = np.array(result_f1_mac)

    print('!!! Final result'
          'acc: %.4f std: %.4f' % (np.mean(result_acc), np.std(result_acc)),
          'f1_mic: %.4f std: %.4f' % (np.mean(result_f1_mic), np.std(result_f1_mic)),
          'f1_mac: %.4f std: %.4f' % (np.mean(result_f1_mac), np.std(result_f1_mac)),

          )


if __name__ == '__main__':
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        cuda_name = 'cuda:' + args.cuda
        device = torch.device(cuda_name)
        print('--> Use GPU %s' % args.cuda)
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
        print("--> No GPU")
    print_configuration(args)

    # Load data
    print('---> Start Loading Dataset...')
    if args.dataset == 'BlogCatalog' or args.dataset == 'Flickr':
        adj, features, labels, idx_train, idx_val, idx_test = load_citationANEmat_gac(args.dataset, args.semi_rate)
    elif args.dataset == 'texas' or args.dataset == 'wisconsin' or args.dataset == 'chameleon' or args.dataset == 'cornell' or args.dataset == 'film' or args.dataset == 'squirrel':
        adj, features, labels, idx_train, idx_val, idx_test = load_webANEmat_gac(args.dataset, args.semi, args.semi_rate)
    else:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_gac(args.dataset, args.semi)

    # adj_norm = get_hops(adj, args.nlayer)
    adj_norm = get_hops(adj, args.nlayer, args)

    print('---> Start Training...')
    main(features, adj_norm, labels, idx_train, idx_val, idx_test, args, device)
    print("Optimization Finished!")
