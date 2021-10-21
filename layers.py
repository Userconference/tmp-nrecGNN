import math
import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class LayerConvol(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, temperature=0.07, bias=False):
        super(LayerConvol, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, xx_anchor, input, adj):
        adj = adj.to_dense()
        anchor_adj = torch.mm(xx_anchor, self.weight)
        anchor_adj = torch.mm(anchor_adj, input.T)  # [batch_node, num_node]

        paddings = torch.ones_like(anchor_adj) * (-2 ** 32 + 1)

        item_att_w = torch.where(adj > 0, anchor_adj, paddings)
        atten = torch.softmax(item_att_w / self.temperature, dim=1)
        output = torch.mm(atten, input)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class LayerConvolAll(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, omega='atten', phi='atten', temperature=0.07, bias=False):
        super(LayerConvolAll, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.omega = omega
        self.phi = phi
        if self.omega == 'atten':
            self.weight = Parameter(torch.FloatTensor(out_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)
        if self.omega == 'atten':
            torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, xx_anchor, input, adj):
        if self.omega == 'atten':
            adj = adj.to_dense()
            anchor_adj = torch.mm(xx_anchor, self.weight)
            anchor_adj = torch.mm(anchor_adj, input.T)  # [batch_node, num_node]
            paddings = torch.ones_like(anchor_adj) * (-2 ** 32 + 1)
            item_att_w = torch.where(adj > 0, anchor_adj, paddings)
            atten = torch.softmax(item_att_w / self.temperature, dim=1)
            output = torch.mm(atten, input)
        elif self.omega == 'sum':
            output = torch.sparse.mm(adj, input)
        elif self.omega == 'mean':
            output = torch.sparse.mm(adj, input)
            div_output = torch.sparse.sum(adj, dim=1).to_dense().view(-1, 1)
            div_output = torch.where(div_output == 0, torch.ones_like(div_output), div_output)
            output /= div_output
        else: # max
            m, n = adj.shape
            _, d = input.shape
            adj = adj.to_dense().view(m, n, 1)
            input = input.view(1, n, d)
            adj = torch.mul(adj, input)
            adj = torch.max(adj, dim=1)[0]
            output = adj

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LayerConvolSelf(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, temperature=0.07, bias=False):
        super(LayerConvolSelf, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.act = nn.ReLU()

        self.linear1 = nn.Linear(self.in_features, self.out_features, bias=False)
        self.linear2 = nn.Linear(self.out_features, 1, bias=False)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, xx_anchor, input, adj):
        adj = adj.to_dense()
        m, d = input.shape
        input = input.view(1, m, d)
        q_vec = xx_anchor.view(-1, 1, d).repeat(1, m, 1)
        xx = torch.cat((q_vec, input.view(1, m, d)), dim=-1)
        atten = self.linear1(xx)
        atten = self.act(atten)
        atten = self.linear2(atten)  # bsz, m, 1
        atten = torch.softmax(atten, dim=1)
        v_vec = torch.mul(input, atten)
        v_vec = torch.sum(v_vec, dim=1)
        return v_vec

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AttenMlp(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, hidden=512, temperature=0.07, bias=False):
        super(AttenMlp, self).__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.temperature = temperature
        self.act = nn.ReLU()
        self.linear1 = nn.Linear(self.in_features * 2, self.hidden, bias=False)
        self.linear2 = nn.Linear(self.hidden, 1, bias=False)

    def reset_parameters(self):
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.q_weight)
        torch.nn.init.xavier_uniform_(self.k_weight)
        torch.nn.init.xavier_uniform_(self.v_weight)

    def forward(self, q_vec, k_vec, v_vec):
        bsz, m, d = k_vec.shape
        q_vec = q_vec.view(-1, 1, self.in_features).repeat(1, m, 1)
        xx = torch.cat((q_vec, k_vec), dim=-1)
        xx = self.linear1(xx)
        xx = self.act(xx)
        xx = self.linear2(xx)   # bsz, m, 1
        xx = torch.softmax(xx, dim=1)
        v_vec = torch.mul(v_vec, xx)
        v_vec = torch.sum(v_vec, dim=1)
        return v_vec

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class AttenMlpFinal(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, hop, hidden=512, temperature=0.07, phi='atten', bias=False):
        super(AttenMlpFinal, self).__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.temperature = temperature
        self.act = nn.ReLU()
        self.hop = hop
        self.phi = phi
        if self.phi == 'atten':
            self.linear1 = nn.Linear(self.in_features * 2, self.hidden, bias=False)
            self.linear2 = nn.Linear(self.hidden, 1, bias=False)
        else:
            self.linear3 = nn.Linear(self.in_features * (self.hop + 1), self.in_features, bias=False)

    def forward(self, q_vec, k_vec, v_vec):

        if self.phi == 'atten':
            bsz, m, d = k_vec.shape
            q_vec = q_vec.view(-1, 1, self.in_features).repeat(1, m, 1)
            xx = torch.cat((q_vec, k_vec), dim=-1)
            xx = self.linear1(xx)
            xx = self.act(xx)
            xx = self.linear2(xx)   # bsz, m, 1
            xx = torch.softmax(xx, dim=1)
            v_vec = torch.mul(v_vec, xx)
            v_vec = torch.sum(v_vec, dim=1)
        else:
            m, seq_len, d = k_vec.shape
            v_vec = self.linear3(k_vec.view(m, -1))
            v_vec = torch.nn.functional.relu(v_vec)
        return v_vec

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
