import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.network import Classifier_1fc, DimReduction



class AttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super(AttentionLayer, self).__init__()
        self.dim = dim

    def forward(self, features, W_1, b_1):
        out_c = F.linear(features, W_1, b_1)
        out = out_c - out_c.max()
        out = out.exp()
        out = out.sum(1, keepdim=True)
        alpha = out / out.sum(0)

        alpha01 = features.size(0) * alpha.expand_as(features)
        context = torch.mul(features, alpha01)

        return context, out_c, torch.squeeze(alpha)

class LBMIL(nn.Module):
    def __init__(self, conf, droprate=0):
        super(LBMIL, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = AttentionLayer(conf.D_inner)
        self.classifier = nn.Linear(conf.D_inner, conf.n_class)

    def forward(self, x): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        out, out_c, alpha = self.attention(med_feat, self.classifier.weight, self.classifier.bias)
        out = out.mean(0, keepdim=True)

        y = self.classifier(out)
        return y, out_c, alpha




