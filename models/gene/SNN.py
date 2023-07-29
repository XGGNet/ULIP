import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
import torch

import math

# 输入数据, 输出activated class prob

class SNN(nn.Module):
    """
    This is a template for defining your customized 3D backbone and use it for pre-training in ULIP framework.
    The expected input is Batch_size x num_points x 3, and the expected output is Batch_size x point_cloud_feat_dim
    """
    def __init__(self):
        pass

    def forward(self, xyz):
        pass


class SNN(nn.Module):
    def __init__(self, input_dim=80, omic_dim=128, return_grad = 'False', dropout_rate=0.1, act=nn.LogSoftmax(dim=1), label_dim=3, init_max=True):
        super(SNN, self).__init__()
        hidden = [64, 48, 32, 32]
        self.act = act
        self.return_grad = return_grad

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            # nn.BatchNorm1d(hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            # nn.BatchNorm1d(hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            # nn.BatchNorm1d(hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            # nn.BatchNorm1d(omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        
        # self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.relu = nn.ReLU(inplace=False)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max: init_max_weights(self)

        # self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        # self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        features = self.relu(self.encoder(x))
        out = self.classifier(features)

        # if self.return_grad == "True":
        #     omic_grads = get_grad_embedding(out, features).detach().cpu().numpy()
        # else:
        omic_grads = None

        if self.act is not None:
            pred = self.act(out)

            if isinstance(self.act, nn.Sigmoid): # not enter
                pred = pred * self.output_range + self.output_shift

        # if self.return_grad == "True":
        #     y_c = torch.sum(out)
        #     features.grad = None
        #     features.retain_grad()
        #     y_c.backward(retain_graph=True)
        #     omic_grads = features.grad.detach().cpu().numpy()
        #     omic_grad_norm = np.linalg.norm(omic_grads, axis=1)
        #     # print("gradient magnitude of the omic feature:", omic_grad_norm)
        #     # print("predicted hazard of the omic modality:", np.reshape(out.detach().cpu().numpy(), (-1)))
        # else:
        #     omic_grads = None

        return features, out, pred, omic_grads


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
