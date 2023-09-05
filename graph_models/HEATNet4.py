import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn import edge_softmax
from dgl.nn.pytorch.glob import GlobalAttentionPooling

from pooling import AvgPooling, SumPooling, MaxPooling


"""
HEATNet (Heterogeneous Edge Attribute Transformer)
Maybe can consider initialize an equal dimension (1024) vector filled by scalar edge weight
"""

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv1d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        l = torch.unsqueeze(l, dim=-1)
        g = torch.unsqueeze(g, dim=-1)
        N, C, W = l.size()
        
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,1)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)

        # return c.view(N,1,W,H), g
        return g


def apply_weights(edges):
    return {'t': edges.data['t'] * edges.data['v']}


class HEATLayer(nn.Module):
    def __init__(self, in_size, out_size, node_dict, n_heads, dropout=0.2):
        super(HEATLayer, self).__init__()

        # W_r for each relation
        self.weight = nn.Linear(in_size, out_size)

        self.in_size = in_size
        self.out_size = out_size

        # Initialize node dicts
        self.node_dict = node_dict
        self.num_node_types = len(node_dict)

        self.n_heads = n_heads
        self.d_k = out_size // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        # Initialize source and target layers
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()

        # Edge feature transformation
        self.e_linear = nn.Linear(1, 1)

        self.skip = nn.Parameter(torch.ones(self.num_node_types))
        self.drop = nn.Dropout(dropout)

        for t in range(self.num_node_types):
            self.k_linears.append(nn.Linear(in_size, out_size))
            self.q_linears.append(nn.Linear(in_size, out_size))
            self.v_linears.append(nn.Linear(in_size, out_size))
            self.a_linears.append(nn.Linear(out_size, out_size))

    def forward(self, G: dgl.DGLGraph, feat_dict):
        # The input is a dictionary of node features for each type
        new_feat_dict = {k: [] for k in feat_dict.keys()}

        node_dict = self.node_dict

        for srctype, etype, dsttype in G.canonical_etypes:
            sub_graph = G[srctype, etype, dsttype]

            # Initialize transformation networks
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]]
            q_linear = self.q_linears[node_dict[dsttype]]

            # Transform features with linear layers
            k = k_linear(feat_dict[srctype]).view(-1, self.n_heads, self.d_k)
            v = v_linear(feat_dict[srctype]).view(-1, self.n_heads, self.d_k)
            q = q_linear(feat_dict[dsttype]).view(-1, self.n_heads, self.d_k)
            ea = self.e_linear(sub_graph.edata["v"].view(-1, 1).type(torch.float32))

            sub_graph.srcdata['k'] = k
            sub_graph.dstdata['q'] = q
            sub_graph.srcdata['v'] = v

            sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
            # Apply edge weights to the features
            attn_score = sub_graph.edata['t'].sum(-1) * ea / self.sqrt_dk
            # attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
            attn_score = edge_softmax(sub_graph, attn_score)
            sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            # sub_graph.apply_edges(apply_weights)

        G.multi_update_all({etype: (fn.u_mul_e('v', 't', 'm'), fn.sum('m', 't'))
                            for etype in G.canonical_etypes}, cross_reducer='mean')

        new_h = {}
        for ntype in G.ntypes:
            '''
                Step 3: Target-specific Aggregation
                x = norm( W[node_type] * gelu( Agg(x) ) + x )
            '''
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            try:
                t = G.nodes[ntype].data['t'].view(-1, self.out_size)
            except KeyError:
                new_h[ntype] = feat_dict[ntype]
                continue
            trans_out = self.drop(self.a_linears[n_id](t))
            trans_out = trans_out * alpha + feat_dict[ntype] * (1 - alpha)
            new_h[ntype] = trans_out

        return new_h


class HEATNet4(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, n_heads, node_dict, dropuout, graph_pooling_type='mean'):
        super(HEATNet4, self).__init__()
        self.node_dict = node_dict # {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        self.gcs = nn.ModuleList()
        self.n_inp = in_dim # 1024
        self.n_hid = hidden_dim # 512
        self.n_out = out_dim # 4
        self.n_layers = n_layers # 2
        self.n_heads = n_heads # 4
        self.adapt_ws = nn.ModuleList()

        self.pools = nn.ModuleList()

        self.linears_prediction = nn.ModuleDict(
            {
                k: nn.Linear(hidden_dim, 256)
                for k, _ in node_dict.items()
             }
        )

        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(in_dim, hidden_dim)) 
            # 对每种节点设置一个projector

        for _ in range(n_layers):
            self.gcs.append(HEATLayer(hidden_dim, hidden_dim, node_dict, n_heads, dropuout)) # dropout is 0.2

        self.attn = nn.ModuleDict(
            {
                a: LinearAttentionBlock(in_features=256, normalize_attn=True)
                for a, _ in node_dict.items()
            }
        )

        # Define pooling readout layers
        for layer in range(n_layers + 1):
            if graph_pooling_type == 'sum':
                self.pools.append(SumPooling())
            elif graph_pooling_type == 'mean':
                self.pools.append(AvgPooling())
            elif graph_pooling_type == 'max':
                self.pools.append(MaxPooling())
            elif graph_pooling_type == 'att':
                if layer == 0:
                    gate_nn = torch.nn.Linear(in_dim, 1)
                else:
                    gate_nn = torch.nn.Linear(hidden_dim, 1)
                self.pools.append(GlobalAttentionPooling(gate_nn))
            else:
                raise NotImplementedError
        
        # self.head_2 = nn.Linear(256*len(node_dict), 256)
        self.head_1 = nn.Linear(256, 64)
        self.head = nn.Linear(64, out_dim)

    def forward(self, G, h=None):

        # Read features   初始化, 取出feature
        if h is None:
            h = {}
            for ntype in G.ntypes:  # ['0', '1', '2', '3', '4', '5']
                n_id = self.node_dict[ntype] # 从 ['0', '1', '2', '3', '4', '5'] 取出对应type..  0~5
                h[ntype] = self.adapt_ws[n_id](G.nodes[ntype].data['feat']) #  (G.nodes[ntype].data['feat']) 对于没有该类型的节点的特征,是空的...
        else:
            for ntype in G.ntypes:
                n_id = self.node_dict[ntype]
                h[ntype] = self.adapt_ws[n_id](h[ntype])

        # Scale and Broadcast features   初始化 边
        ea = G.edata['sim'] # 边相似度   src_node_type -->  edge_type --> tar_node_type
        G.edata['v'] = ea

        # Propagate and Collect features for pooling

        # Transformer
        for i in range(self.n_layers): # 2
            h = self.gcs[i](G, h)

        out_h = {}
        for k, v in h.items():
            if h[k].shape[0] > 0: # 如果该类型的节点数量不为0
                out_h[k] = self.linears_prediction[k](self.pools[0](G, h, ntype=k)) # 对每个类型的节点都有一个output_linear
            else:
                out_h[k] = h[k]

        # out_h is a dict, 每个模态的tensor shape is [1,256]

        # Taking the sum of predictions scores and compute attention score for each node type features
        attn_g = {}
        with G.local_scope():
            hg = 0
            count = 0
            # for h in h_list:
            for ntype in G.ntypes:
                if h[ntype].shape[0] > 0:
                    hg = hg + out_h[ntype]  # (1,256)
                    count += 1

            '''
            out_h['gene']   (1,256)
            out_h['image']   (n_patch,256)
            out_h['text'] (1,256)

            hg.shape [1,256]
            '''
            for a, v in h.items():  # dict of node features
                if out_h[a].shape[0] > 0:
                    attn_g[a] = self.attn[a](out_h[a], hg)
                else:
                    attn_g[a].append(torch.zeros(1,256).cuda())

            # g = torch.cat(attn_g_list, dim=1)  # [1,256] + [1,256] + [1,256] => [1, 768]
            # g = self.head_2(g) # [1, 256]
            g = self.head_1(attn_g['image']) # [1,256] => [1, 64]
            g = self.head(g) # [1, 64] => [1, 3]

        return g
