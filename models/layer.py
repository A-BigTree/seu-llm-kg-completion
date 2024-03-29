import torch
import torch.nn as nn
import torch.nn.functional as F
from common.config import TRAIN_CONFIG


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region back prop-attention layer."""

    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        temp = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(temp, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = temp._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices
            if TRAIN_CONFIG["cuda"] is not None and TRAIN_CONFIG["cuda"] >= 0:
                edge_sources = edge_sources.to(f'cuda:{TRAIN_CONFIG["cuda"]}')
            grad_values = grad_output[edge_sources]
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903"""

    def __init__(self, num_nodes, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(out_features, 2 * in_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leak_y_relu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input_, edge):
        N = input_.size()[0]

        edge_h = torch.cat((input_[edge[0, :], :], input_[edge[1, :], :]), dim=1).t()
        # edge_h: (2*in_dim) x E

        edge_m = self.W.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        edge_e = torch.exp(-self.leak_y_relu(self.a.mm(edge_m).squeeze())).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_row_sum = self.special_spmm_final(edge, edge_e, N, edge_e.shape[0], 1)
        e_row_sum[e_row_sum == 0.0] = 1e-12

        e_row_sum = e_row_sum
        # e_row_sum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(edge, edge_w, N, edge_w.shape[0], self.out_features)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_row_sum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGAT(nn.Module):
    def __init__(self, num_nodes, n_feat, n_hid, dropout, alpha, n_heads):
        """Sparse version of GAT"""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, n_feat,
                                                 n_hid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(n_heads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(n_feat, n_heads * n_hid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, n_heads * n_hid,
                                             n_heads * n_hid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, entity_embeddings, relation_embeddings, edge_list):
        x = entity_embeddings
        x = torch.cat([att(x, edge_list) for att in self.attentions], dim=1)
        x = self.dropout_layer(x)
        x_rel = relation_embeddings.mm(self.W)
        x = F.elu(self.out_att(x, edge_list))
        return x, x_rel


class MutanLayer(nn.Module):

    def __init__(self, dim, multi):
        super(MutanLayer, self).__init__()

        self.dim = dim
        self.multi = multi

        modal1 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

    def forward(self, modal1_emb, modal2_emb):
        bs = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_mm.append(torch.mul(x_modal1, x_modal2))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(bs, self.dim)
        x_mm = torch.relu(x_mm)
        return x_mm


class TuckERLayer(nn.Module):
    def __init__(self, dim, r_dim):
        super(TuckERLayer, self).__init__()

        self.W = nn.Parameter(torch.rand(r_dim, dim, dim))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.out_drop = nn.Dropout(0.5)

    def forward(self, e_embed, r_embed):
        x = self.bn0(e_embed)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))

        r = torch.mm(r_embed, self.W.view(r_embed.size(1), -1))
        r = r.view(-1, x.size(2), x.size(2))
        r = self.hidden_drop(r)

        x = torch.bmm(x, r)
        x = x.view(-1, x.size(2))
        x = self.bn1(x)
        x = self.out_drop(x)
        return x
