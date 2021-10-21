import torch.nn as nn
import torch.nn.functional as F
from layers import LayerConvol, AttenMlp, LayerConvolSelf, LayerConvolAll, AttenMlpFinal
import torch
import math
from torch.nn.parameter import Parameter


class nrecGNN(nn.Module):
    def __init__(self, nfeat, n_hops, act, hidden_state,
                 share_attn, nclass, dropnode=0.0, drop_edge=0.6):
        super(nrecGNN, self).__init__()
        self.input_dim = nfeat
        self.n_hops = n_hops
        self.nclass = nclass
        self.share_attn = share_attn
        self.hidden_state = hidden_state
        self.dropnode = dropnode
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()
        self.dropout = [drop_edge for i in range(n_hops)]
        self.__init()

    def __init(self):
        if self.share_attn:
            self.gc1 = LayerConvol(self.input_dim, self.input_dim)
        else:
            self.gc1 = [LayerConvol(self.input_dim, self.input_dim) for i in range(self.n_hops)]
            # self.gc1 = LayerConvolSelf(self.input_dim, self.hidden_state)
        self.atten = AttenMlp(self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.hidden_state, bias=True)
        self.linear2 = nn.Linear(self.hidden_state, self.nclass, bias=True)

    def __dropout_x(self, x, idx):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        if len(self.dropout) == 1:
            dropt = self.dropout[-1]
        else:
            dropt = self.dropout[idx]

        random_index = torch.rand(len(values)) + dropt
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def forward(self, x, adj, idx):
        seq_emb = []
        seq_emb.append(F.normalize(x[idx], p=2, dim=1))
        xx_anchor = seq_emb[0]
        if self.training:
            adj = [self.__dropout_x(adj_, i) for i, adj_ in enumerate(adj)]

        for i, adj_ in enumerate(adj):
            if self.share_attn:
                x_ = self.gc1(xx_anchor, x, adj_)
            else:
                x_ = self.gc1[i](xx_anchor, x, adj_)

            seq_emb.append(F.normalize(x_, p=2, dim=1))

        seq_emb = torch.stack(seq_emb, dim=1)
        output = self.atten(xx_anchor, seq_emb, seq_emb)
        output = F.dropout(output, self.dropnode, training=self.training)
        output = self.linear(output)
        output = self.act(output)
        output = self.linear2(output)
        return F.log_softmax(output, dim=1)


class nrecGNN_all(nn.Module):
    def __init__(self, nfeat, n_hops, act, hidden_state,
                 share_attn, nclass, dropnode=0.0, drop_edge=0.6):
        super(nrecGNN_all, self).__init__()
        self.input_dim = nfeat
        self.n_hops = n_hops
        self.nclass = nclass
        self.share_attn = share_attn
        self.hidden_state = hidden_state
        self.dropnode = dropnode
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()
        self.dropout = [drop_edge for i in range(n_hops)]
        self.__init()

    def __init(self):
        if self.share_attn:
            self.gc1 = LayerConvol(self.input_dim, self.input_dim)
        else:
            self.gc1 = [LayerConvol(self.input_dim, self.input_dim) for i in range(self.n_hops)]
            # self.gc1 = LayerConvolSelf(self.input_dim, self.hidden_state)
        self.atten = AttenMlp(self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.hidden_state, bias=True)
        self.linear2 = nn.Linear(self.hidden_state, self.nclass, bias=True)

    def __dropout_x(self, x, idx):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        if len(self.dropout) == 1:
            dropt = self.dropout[-1]
        else:
            dropt = self.dropout[idx]

        random_index = torch.rand(len(values)) + dropt
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def forward(self, x, adj, idx):
        seq_emb = []
        seq_emb.append(F.normalize(x[idx], p=2, dim=1))
        xx_anchor = seq_emb[0]
        if self.training:
            adj = [self.__dropout_x(adj_, i) for i, adj_ in enumerate(adj)]

        for i, adj_ in enumerate(adj):
            if self.share_attn:
                x_ = self.gc1(xx_anchor, x, adj_)
            else:
                x_ = self.gc1[i](xx_anchor, x, adj_)

            seq_emb.append(F.normalize(x_, p=2, dim=1))

        seq_emb = torch.stack(seq_emb, dim=1)
        output = self.atten(xx_anchor, seq_emb, seq_emb)
        output = F.dropout(output, self.dropnode, training=self.training)
        output = self.linear(output)
        output = self.act(output)
        output = self.linear2(output)
        return F.log_softmax(output, dim=1)


class nrecGNN_large(nn.Module):
    def __init__(self, nfeat, n_hops, act, hidden_state,
                 share_attn, nclass, dropnode=0.0, drop_edge=0.6):
        super(nrecGNN_large, self).__init__()
        self.input_dim = nfeat
        self.n_hops = n_hops
        self.nclass = nclass
        self.share_attn = share_attn
        self.hidden_state = hidden_state
        self.dropnode = dropnode
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()
        self.dropout = [drop_edge for i in range(n_hops)]
        self.__init()

    def __init(self):
        if self.share_attn:
            self.gc1 = LayerConvol(self.input_dim, self.input_dim)
        else:
            self.gc1 = [LayerConvol(self.input_dim, self.input_dim) for i in range(self.n_hops)]
            # self.gc1 = LayerConvolSelf(self.input_dim, self.hidden_state)
        self.atten = AttenMlp(self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.hidden_state, bias=True)
        self.linear2 = nn.Linear(self.hidden_state, self.nclass, bias=True)

    def __dropout_x(self, x, idx):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        if len(self.dropout) == 1:
            dropt = self.dropout[-1]
        else:
            dropt = self.dropout[idx]

        random_index = torch.rand(len(values)) + dropt
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def extract_hop_features(self, x, adj, idx):
        xx_anchor = F.normalize(x[idx], p=2, dim=1)
        if self.training:
            adj = self.__dropout_x(adj, 0)
        x_hop = self.gc1(xx_anchor, x, adj)
        return x_hop

    def forward(self, x, hop_feat, idx):
        seq_emb = []
        seq_emb.append(F.normalize(x[idx], p=2, dim=1))
        xx_anchor = seq_emb[0]
        seq_emb += hop_feat
        seq_emb = torch.stack(seq_emb, dim=1)
        output = self.atten(xx_anchor, seq_emb, seq_emb)
        output = F.dropout(output, self.dropnode, training=self.training)
        output = self.linear(output)
        output = self.act(output)
        output = self.linear2(output)
        return F.log_softmax(output, dim=1)


class nrecGNN_final(nn.Module):
    def __init__(self, nfeat, n_hops, act, hidden_state,
                 share_attn, nclass, omega='atten', phi='atten', dropnode=0.0, drop_edge=0.6):
        super(nrecGNN_final, self).__init__()
        self.input_dim = nfeat
        self.n_hops = n_hops
        self.nclass = nclass
        self.share_attn = share_attn
        self.hidden_state = hidden_state
        self.dropnode = dropnode
        self.omega = omega
        self.phi = phi
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()
        self.dropout = [drop_edge for i in range(n_hops)]
        self.__init()

    def __init(self):
        self.trans_layer = nn.Linear(self.input_dim, self.hidden_state)
        if self.share_attn:
            self.gc1 = LayerConvolAll(self.input_dim, self.hidden_state, self.omega, self.phi)
        else:
            self.gc1 = [LayerConvolAll(self.input_dim, self.hidden_state, self.omega, self.phi) for i in range(self.n_hops)]
        self.atten = AttenMlpFinal(self.hidden_state, self.n_hops, phi=self.phi)
        self.linear = nn.Linear(self.hidden_state, 128, bias=True)
        self.linear2 = nn.Linear(128, self.nclass, bias=True)

    def __dropout_x(self, x, idx):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        if len(self.dropout) == 1:
            dropt = self.dropout[-1]
        else:
            dropt = self.dropout[idx]

        random_index = torch.rand(len(values)) + dropt
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def trans_x(self, xx):
        xx = self.trans_layer(xx)
        xx = self.act(xx)
        return xx

    def extract_hop_features(self, x, adj, idx):
        xx_anchor = F.normalize(x[idx], p=2, dim=1)
        if self.training:
            adj = self.__dropout_x(adj, 0)
        x_hop = self.gc1(xx_anchor, x, adj)
        return x_hop

    def forward(self, x, hop_feat, idx):
        seq_emb = []
        seq_emb.append(F.normalize(x[idx], p=2, dim=1))
        xx_anchor = seq_emb[0]
        seq_emb += hop_feat
        seq_emb = torch.stack(seq_emb, dim=1)
        output = self.atten(xx_anchor, seq_emb, seq_emb)
        output = F.dropout(output, self.dropnode, training=self.training)
        output = self.linear(output)
        output = self.act(output)
        output = self.linear2(output)
        return F.log_softmax(output, dim=1)


class nrecGNN_test(nn.Module):
    def __init__(self, nfeat, n_hops, act, hidden_state,
                 share_attn, nclass, dropnode=0.0, drop_edge=0.6):
        super(nrecGNN_test, self).__init__()
        self.input_dim = nfeat
        self.n_hops = n_hops
        self.nclass = nclass
        self.share_attn = share_attn
        self.hidden_state = hidden_state
        self.dropnode = dropnode
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()
        self.dropout = [drop_edge for i in range(n_hops)]
        self.__init()

    def __init(self):
        self.trans = nn.Linear(self.input_dim, self.hidden_state, bias=True)
        if self.share_attn:
            self.gc1 = LayerConvol(self.hidden_state, self.hidden_state)
        else:
            # self.gc1 = [LayerConvol(self.input_dim, self.input_dim) for i in range(self.n_hops)]
            self.gc1 = LayerConvolSelf(self.input_dim, self.hidden_state)
        self.atten = AttenMlp(self.hidden_state)
        self.linear = nn.Linear(self.hidden_state, self.hidden_state, bias=True)
        self.linear2 = nn.Linear(self.hidden_state, self.nclass, bias=True)

    def __dropout_x(self, x, idx):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        if len(self.dropout) == 1:
            dropt = self.dropout[-1]
        else:
            dropt = self.dropout[idx]

        random_index = torch.rand(len(values)) + dropt
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def forward(self, x, adj, idx):
        x = F.relu(self.trans(x))
        seq_emb = []
        seq_emb.append(F.normalize(x[idx], p=2, dim=1))
        xx_anchor = seq_emb[0]
        if self.training:
            adj = [self.__dropout_x(adj_, i) for i, adj_ in enumerate(adj)]

        for i, adj_ in enumerate(adj):
            if self.share_attn:
                x_ = self.gc1(xx_anchor, x, adj_)
            else:
                x_ = self.gc1(xx_anchor, x, adj_)
            seq_emb.append(F.normalize(x_, p=2, dim=1))

        seq_emb = torch.stack(seq_emb, dim=1)
        output = self.atten(xx_anchor, seq_emb, seq_emb)
        output = F.dropout(output, self.dropnode, training=self.training)
        output = self.linear(output)
        output = self.act(output)
        output = self.linear2(output)
        return F.log_softmax(output, dim=1)



class nrecGNN_prop(nn.Module):
    def __init__(self, nfeat, n_hops, act, hidden_state, n_hidden,
                 share_attn, nclass, dropnode=0.0, drop_edge=0.6):
        super(nrecGNN_prop, self).__init__()
        self.input_dim = nfeat
        self.n_hops = n_hops
        self.nclass = nclass
        self.nclass_hidden = n_hidden
        self.share_attn = share_attn
        self.hidden_state = hidden_state
        self.dropnode = dropnode
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()
        self.dropout = [drop_edge for i in range(n_hops)]
        self.__init()

    def __init(self):
        self.linear1 = nn.Linear(self.input_dim, self.hidden_state, bias=False)
        self.linear2 = nn.Linear(self.hidden_state, self.nclass, bias=True)
        if self.share_attn:
            self.gc1 = LayerConvol(self.nclass, self.nclass)
        else:
            layers = [LayerConvol(self.nclass, self.nclass) for i in range(self.n_hops)]
            self.gc1 = torch.nn.Module(layers)
        self.atten = AttenMlp(self.nclass, hidden=self.nclass_hidden)

    def __dropout_x(self, x, idx):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        if len(self.dropout) == 1:
            dropt = self.dropout[-1]
        else:
            dropt = self.dropout[idx]

        random_index = torch.rand(len(values)) + dropt
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def forward(self, x, adj, idx):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        seq_emb = []
        seq_emb.append(x[idx])
        xx_anchor = seq_emb[0]
        if self.training:
            adj = [self.__dropout_x(adj_, i) for i, adj_ in enumerate(adj)]

        for i, adj_ in enumerate(adj):
            if self.share_attn:
                x_ = self.gc1(xx_anchor, x, adj_)
            else:
                x_ = self.gc1[i](xx_anchor, x, adj_)
            seq_emb.append(x_)

        seq_emb = torch.stack(seq_emb, dim=1)
        output = self.atten(xx_anchor, seq_emb, seq_emb)
        output = F.dropout(output, self.dropnode, training=self.training)
        return F.log_softmax(output, dim=1)

