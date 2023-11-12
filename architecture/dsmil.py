import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# vpt
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from functools import reduce
from operator import mul


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, conf, dropout_v=0.0, nonlinear=True, passing_v=False,
                 confounder_path=False):  # K, L, N
        super(BClassifier, self).__init__()
        self.n_masked_patch = conf.n_masked_patch
        input_size=conf.D_feat
        output_class=conf.n_class
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, conf.D_inner), nn.ReLU(), nn.Linear(conf.D_inner, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, conf.D_inner)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        self.confounder_path = None
        if confounder_path:
            self.confounder_path = confounder_path
            conf_list = []
            for i in confounder_path:
                conf_list.append(torch.from_numpy(np.load(i)).float())
            conf_tensor = torch.cat(conf_list,
                                    0)  # [ k, C, K] k-means, c classes , K-dimension, should concatenate at centers k
            conf_tensor_dim = conf_tensor.shape[-1]
            self.register_buffer("confounder_feat", conf_tensor)
            joint_space_dim = 128
            dropout_v = 0.1
            self.confounder_W_q = nn.Linear(input_size, joint_space_dim)
            self.confounder_W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size + conf_tensor_dim)
            # self.classifier =  nn.Linear(self.L*self.K+in_size, out_size)
            self.dropout = nn.Dropout(dropout_v)


    def forward(self, feats, c, is_train=True):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0,
                                  descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        # print(m_indices.shape)
        m_feats = torch.index_select(feats, dim=0,
                                     index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0,
                                        1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device))  # normalize attention scores, A in shape N x C,
        A = A.transpose(0, 1)

        if self.n_masked_patch > 0 and is_train:
            # Get the indices of the top-k largest values
            q, c = A.shape
            n_masked_patch = min(self.n_masked_patch, c)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            indices = indices.reshape(q, -1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :int(n_masked_patch * 0.5)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(q, c).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask.reshape(q, -1) == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=-1)
        B = torch.mm(A, V)  # compute bag representation, B in shape C x V
        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        # cls-specific confounder
        if self.confounder_path:
            if 'agnostic' in self.confounder_path[0]:
                device = B.device
                bag_q = self.confounder_W_q(B.squeeze(0))  # bs x C x V -- C x V
                conf_k = self.confounder_W_k(self.dropout(self.confounder_feat))  # k x V
                A = torch.mm(conf_k, bag_q.transpose(0, 1))  # k * C
                A = F.softmax(A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)),
                              0)  # normalize attention scores, A in shape N x C,
                conf_feats = torch.mm(A.transpose(0, 1),
                                      self.confounder_feat)  # compute bag representation, B in shape C x V
                B = torch.cat((B, conf_feats.unsqueeze(0)), dim=-1)
            elif self.confounder_path:  #### cls-agnostic
                device = B.device
                bag_q = self.confounder_W_q(B.squeeze(0)).unsqueeze(0)  # bs x C x V --- C x V ----bs x C x Q
                conf_k = self.confounder_W_k(self.confounder_feat.view(-1, B.shape[-1]))  # k x C x K  ---- C*k x K
                conf_k = conf_k.view(self.confounder_feat.shape[0], self.confounder_feat.shape[1],
                                     bag_q.shape[-1])  # C*k x K ---k x C x Q
                A = torch.einsum('kcq, bcq -> kcb ', conf_k, bag_q)
                # A = torch.mm(conf_k, bag_q.transpose(0, 1))
                A = F.softmax(A / torch.sqrt(torch.tensor(conf_k.shape[-1], dtype=torch.float32, device=device)), 0)  #
                # conf_feats = torch.mm(A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
                conf_feats = torch.einsum(' kcb ,kcq-> bcq ', A, self.confounder_feat)
                B = torch.cat((B, conf_feats), dim=2)
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A_out, B


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x, is_train=True):
        feats, classes = self.i_classifier(x[0])
        # print(feats)
        prediction_bag, A, B = self.b_classifier(feats, classes, is_train=is_train)
        return classes, prediction_bag, A
