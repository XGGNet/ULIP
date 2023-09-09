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

        for t in range(self.num_node_types): # 对每种节点设置 一套transformer
            self.k_linears.append(nn.Linear(in_size, out_size))
            self.q_linears.append(nn.Linear(in_size, out_size))
            self.v_linears.append(nn.Linear(in_size, out_size))
            self.a_linears.append(nn.Linear(out_size, out_size))

    def forward(self, G: dgl.DGLGraph, feat_dict):
        # The input is a dictionary of node features for each type
        new_feat_dict = {k: [] for k in feat_dict.keys()}   #  {'gene': [], 'image': [], 'text': []}

        node_dict = self.node_dict  #  {'gene': 0, 'image': 1, 'text': 2}

        for srctype, etype, dsttype in G.canonical_etypes:   # [('gene', 'gene_text', 'text'), ('image', 'image_image', 'image')]
            sub_graph = G[srctype, etype, dsttype] # 按边类型取出子图

            # Initialize transformation networks
            ## 对 src_node 施加 linear
            k_linear = self.k_linears[node_dict[srctype]]  # dim 256 -> 256
            v_linear = self.v_linears[node_dict[srctype]]
            q_linear = self.q_linears[node_dict[dsttype]]

            # Transform features with linear layers
            k = k_linear(feat_dict[srctype]).view(-1, self.n_heads, self.d_k) 
            v = v_linear(feat_dict[srctype]).view(-1, self.n_heads, self.d_k)
            q = q_linear(feat_dict[dsttype]).view(-1, self.n_heads, self.d_k) # [15,256] ==> [15,256] ==> [15, 4, 64] 
            ea = self.e_linear(sub_graph.edata["v"].view(-1, 1).type(torch.float32)) # [15] ==> [15,1] ==> [15, 1]

            sub_graph.srcdata['k'] = k
            sub_graph.dstdata['q'] = q
            sub_graph.srcdata['v'] = v

            sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
            # Apply edge weights to the features
            attn_score = sub_graph.edata['t'].sum(-1) * ea / self.sqrt_dk
            # attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
            attn_score = edge_softmax(sub_graph, attn_score)
            sub_graph.edata['t'] = attn_score.unsqueeze(-1) # [324, 4, 1]

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

#         'gene':
# tensor([[ 4.3952e-02, -4.1549e-02,  9.0665e-03,  4.4403e-02, -2.3402e-02,
#           1.9232e-02, -3.0891e-02, -4.4806e-02, -1.9469e-02, -6.0532e-02,
#          -1.5127e-02,  1.2001e-02, -7.1054e-03,  1.4469e-02, -3.1353e-03,
#          -6.7181e-02,  7.0736e-03,  2.2007e-02,  8.9750e-03, -8.8043e-03,
#          -6.2119e-02, -4.1153e-02, -3.0704e-02,  2.2448e-02, -4.2256e-02,
#           2.0755e-02, -4.6418e-02,  2.8682e-02,  2.6858e-02,  1.5333e-02,
#          -2.7897e-03, -9.3331e-03,  1.3102e-02, -2.3931e-02, -2.1103e-03,
#           3.0314e-03, -1.7797e-02,  3.8931e-02, -1.1157e-02, -2.8593e-03,
#           1.9658e-02,  1.4612e-02,  1.1456e-02, -2.1339e-02,  9.2565e-03,
#          -4.2042e-02, -1.3499e-03,  3.9616e-02, -1.7786e-02, -1.8375e-02,
#           3.0988e-02,  3.6729e-02,  3.1422e-02, -3.6406e-02,  2.7630e-02,
#           2.7841e-02,  3.2257e-03, -8.0709e-04,  1.0647e-02, -7.8662e-03,
#          -3.1033e-02, -2.9313e-02,  4.5702e-02,  1.8528e-02, -3.4076e-02,
#           7.4554e-03, -5.6271e-02, -6.6575e-02,  6.9137e-02,  ...
# 'image':
# tensor([[-0.0724,  0.0109, -0.0608,  ..., -0.0683,  0.0276,  0.0417],
#         [-0.0688,  0.0101, -0.0608,  ..., -0.0676,  0.0277,  0.0452],
#         [-0.0700,  0.0106, -0.0634,  ..., -0.0699,  0.0268,  0.0426],
#         ...,
#         [-0.0659,  0.0106, -0.0648,  ..., -0.0687,  0.0270,  0.0434],
#         [-0.0694,  0.0124, -0.0621,  ..., -0.0684,  0.0289,  0.0412],
#         [-0.0734,  0.0121, -0.0611,  ..., -0.0687,  0.0279,  0.0426]],
#        device='cuda:0')
# 'text':
# tensor([[ 0.0078, -0.0461, -0.0315,  ..., -0.0005,  0.0090, -0.0323],


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
        # self.head_1 = nn.Linear(256, 64)
        # self.head = nn.Linear(64, out_dim)

        # self.head_1 = nn.Linear(256, 64)
        # self.head_h = nn.Linear(512, 64)
        self.head_1 = nn.Linear(256, 512)
        self.head = nn.Linear(512, out_dim)

    def forward(self, G, h=None):

        image_features = G.nodes['image'].data['feat']

        # Read features   初始化, 取出feature
        if h is None: # enter here
            h = {}
            for ntype in G.ntypes:  # ['0', '1', '2', '3', '4', '5']
                n_id = self.node_dict[ntype] # 从 ['0', '1', '2', '3', '4', '5'] 取出对应type..  0~5
                h[ntype] = self.adapt_ws[n_id](G.nodes[ntype].data['feat']) # a MLP for each modal, projecting each modality features to a embedding
                # dim: 512 -> 256  还没聚合
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
        
        # 还没有聚合..

        out_h = {}
        for k, v in h.items():
            if h[k].shape[0] > 0: # 如果该类型的节点数量不为0
                out_h[k] = self.linears_prediction[k](self.pools[0](G, h, ntype=k)) # 对每个类型的节点都有一个output_linear
            else:
                out_h[k] = h[k]

        # out_h is a dict, 每个模态的tensor shape is [1,256]

        # 聚合后的特征..

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
            # hg是全部node特征的ensemble

            for a, v in h.items():  # dict of node features
                if out_h[a].shape[0] > 0:
                    attn_g[a] = self.attn[a](out_h[a], hg) # [1, 256] ==> [1, 256]
                else:
                    attn_g[a].append(torch.zeros(1,256).cuda())

            # g = torch.cat(attn_g_list, dim=1)  # [1,256] + [1,256] + [1,256] => [1, 768]
            # g = self.head_2(g) # [1, 256]
            
            g = self.head_1(attn_g['image']) # [1,256] => [1, 64]
            # g_ori = self.head_h(image_features)
            g = self.head(g+image_features) # [n_patch, 64] => [n_patch, 3]

        return g


class HEATNet4_v1(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, n_heads, node_dict, dropuout, graph_pooling_type='mean'):
        super(HEATNet4_v1, self).__init__()
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
        # self.head_1 = nn.Linear(256, 64)
        # self.head = nn.Linear(64, out_dim)

        # self.head_1 = nn.Linear(256, 64)
        # self.head_h = nn.Linear(512, 64)
        self.head_1 = nn.Linear(256, 64)
        self.head = nn.Linear(64, out_dim)

    def forward(self, G, h=None):

        image_features = G.nodes['image'].data['feat']
        n_patch = image_features.shape[0]

        # Read features   初始化, 取出feature
        if h is None: # enter here
            h = {}
            for ntype in G.ntypes:  # ['0', '1', '2', '3', '4', '5']
                n_id = self.node_dict[ntype] # 从 ['0', '1', '2', '3', '4', '5'] 取出对应type..  0~5
                h[ntype] = self.adapt_ws[n_id](G.nodes[ntype].data['feat']) # a MLP for each modal, projecting each modality features to a embedding
                # dim: 512 -> 256  还没聚合
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
        
        # 还没有聚合..

        out_h = {}
        for k, v in h.items():
            if h[k].shape[0] > 0: # 如果该类型的节点数量不为0
                out_h[k] = self.linears_prediction[k](self.pools[0](G, h, ntype=k)) # 对每个类型的节点都有一个output_linear
            else:
                out_h[k] = h[k]

        # out_h is a dict, 每个模态的tensor shape is [1,256]

        # 聚合后的特征..

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
            # hg是全部node特征的ensemble

            for a, v in h.items():  # dict of node features
                if out_h[a].shape[0] > 0:
                    attn_g[a] = self.attn[a](out_h[a], hg) # [1, 256] ==> [1, 256]
                else:
                    attn_g[a].append(torch.zeros(1,256).cuda())

            # g = torch.cat(attn_g_list, dim=1)  # [1,256] + [1,256] + [1,256] => [1, 768]
            # g = self.head_2(g) # [1, 256]
            
            g = self.head_1( attn_g['image'] ) # [1,256] => [1, 64]
            # g_ori = self.head_h(image_features)
            # g = self.head(g+image_features) # [n_patch, 64] => [n_patch, 3]
            g = self.head(g)

            # repeat g in X times in dim 0
            g = g.tile( (n_patch,1) )
         

        return g



class HEATNet4_v2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, n_heads, node_dict, dropuout, graph_pooling_type='mean'):
        super(HEATNet4_v2, self).__init__()
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
        # self.head_1 = nn.Linear(256, 64)
        # self.head = nn.Linear(64, out_dim)

        self.head_1 = nn.Linear(256, 64)
        self.head_h = nn.Linear(512, 64)
        # self.head_1 = nn.Linear(256, 512)
        self.head = nn.Linear(64, out_dim)

    def forward(self, G, h=None):

        image_features = G.nodes['image'].data['feat']

        # Read features   初始化, 取出feature
        if h is None: # enter here
            h = {}
            for ntype in G.ntypes:  # ['0', '1', '2', '3', '4', '5']
                n_id = self.node_dict[ntype] # 从 ['0', '1', '2', '3', '4', '5'] 取出对应type..  0~5
                h[ntype] = self.adapt_ws[n_id](G.nodes[ntype].data['feat']) # a MLP for each modal, projecting each modality features to a embedding
                # dim: 512 -> 256  还没聚合
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
        
        # 还没有聚合..

        out_h = {}
        for k, v in h.items():
            if h[k].shape[0] > 0: # 如果该类型的节点数量不为0
                out_h[k] = self.linears_prediction[k](self.pools[0](G, h, ntype=k)) # 对每个类型的节点都有一个output_linear
            else:
                out_h[k] = h[k]

        # out_h is a dict, 每个模态的tensor shape is [1,256]

        # 聚合后的特征..

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
            # hg是全部node特征的ensemble

            for a, v in h.items():  # dict of node features
                if out_h[a].shape[0] > 0:
                    attn_g[a] = self.attn[a](out_h[a], hg) # [1, 256] ==> [1, 256]
                else:
                    attn_g[a].append(torch.zeros(1,256).cuda())

            # g = torch.cat(attn_g_list, dim=1)  # [1,256] + [1,256] + [1,256] => [1, 768]
            # g = self.head_2(g) # [1, 256]
            
            g = self.head_1(attn_g['image']) # [1,256] => [1, 64]
            g_ori = self.head_h(image_features)
            g = self.head(g+g_ori) # [n_patch, 64] => [n_patch, 3]

        return g




from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class my_gcn(nn.Module):
    def __init__(self ):
        super(my_gcn, self).__init__()
        # self.layers = nn.ModuleList()
        # # input layer
        # self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # # hidden layers
        # for i in range(n_layers - 1):
        #     self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # # output layer
        # self.layers.append(GraphConv(n_hidden, n_classes))
        # self.dropout = nn.Dropout(p=dropout)

        self.W1 = nn.Linear(512, 256)
        self.W2 = nn.Linear(256, 3)

    def forward(self, X, H):
        # h = features
        # for i, layer in enumerate(self.layers):
        #     if i != 0:
        #         h = self.dropout(h)
        #     h = layer(g, h)

        X = self.W1(X)
        X = torch.mm(H, X)/(torch.sum(H, dim=1, keepdim=True)+0.01) # [E, C]

        X = self.W2(X)
        X = torch.mm(H, X)/(torch.sum(H, dim=1, keepdim=True)+0.01) # [E, C]

        return X