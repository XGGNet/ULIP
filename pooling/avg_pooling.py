from torch import nn

from dgl.readout import mean_nodes


class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()


    def forward(self, graph, feat, ntype=None):
        with graph.local_scope():
            try:
                graph.ndata['h'] = feat
                if ntype is None:
                    readout = mean_nodes(graph, 'h')
                else:
                    readout = mean_nodes(graph, 'h', ntype=ntype)
            except:
                readout = feat[ntype].mean(0).unsqueeze(0)

            # readout 就是对应类型的节点特征 的平均 (或者约等于)

            return readout  # [1, 256]