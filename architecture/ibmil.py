import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from architecture.network import Classifier_1fc, DimReduction

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN


        return A  ### K x N


class IBMIL(nn.Module):
    def __init__(self, conf, confounder_dim=128, confounder_merge='cat'):
        super(IBMIL, self).__init__()
        self.confounder_merge = confounder_merge
        assert confounder_merge in ['cat', 'add', 'sub']
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, 128, 1)
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0)
        self.confounder_path = None
        if conf.c_path:
            print('deconfounding')
            self.confounder_path = conf.c_path
            conf_list = []
            for i in conf.c_path:
                conf_list.append(torch.from_numpy(np.load(i)).view(-1, conf.D_inner).float())
            conf_tensor = torch.cat(conf_list, 0)
            conf_tensor_dim = conf_tensor.shape[-1]
            if conf.c_learn:
                self.confounder_feat = nn.Parameter(conf_tensor, requires_grad=True)
            else:
                self.register_buffer("confounder_feat", conf_tensor)
            joint_space_dim = confounder_dim
            dropout_v = 0.5
            self.W_q = nn.Linear(conf.D_inner, joint_space_dim)
            self.W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            if confounder_merge == 'cat':
                self.classifier = nn.Linear(conf.D_inner + conf_tensor_dim, conf.n_class)
            elif confounder_merge == 'add' or 'sub':
                self.classifier = nn.Linear(conf.D_inner, conf.n_class)
            self.dropout = nn.Dropout(dropout_v)

    def forward(self, x):
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, x) ## K x L
        # x = x.squeeze(0)

        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        # H = self.feature_extractor_part2(H)  # NxL

        # A = self.attention_1(x)
        # A = self.attention_2(A)  # NxK
        # A = self.attention(x)  # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        # print('norm')
        # A = F.softmax(A/ torch.sqrt(torch.tensor(x.shape[1])), dim=1)  # For Vis

        # M = torch.mm(A, x)  # KxL
        if self.confounder_path:
            device = M.device
            # bag_q = self.confounder_W_q(M)
            # conf_k = self.confounder_W_k(self.confounder_feat)
            bag_q = self.W_q(M)
            conf_k = self.W_k(self.confounder_feat)
            deconf_A = torch.mm(conf_k, bag_q.transpose(0, 1))
            deconf_A = F.softmax(
                deconf_A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)),
                0)  # normalize attention scores, A in shape N x C,
            conf_feats = torch.mm(deconf_A.transpose(0, 1),
                                  self.confounder_feat)  # compute bag representation, B in shape C x V
            if self.confounder_merge == 'cat':
                M = torch.cat((M, conf_feats), dim=1)
            elif self.confounder_merge == 'add':
                M = M + conf_feats
            elif self.confounder_merge == 'sub':
                M = M - conf_feats
        Y_prob = self.classifier(M)
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        if self.confounder_path:
            return Y_prob, M, deconf_A
        else:
            return Y_prob, M, A

    # # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     _, Y_hat, _ = self.forward(X)
    #     error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
    #
    #     return error, Y_hat
    #
    # def calculate_objective(self, X, Y):
    #     Y = Y.float()
    #     Y_prob, _, A = self.forward(X)
    #     Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
    #     neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
    #
    #     return neg_log_likelihood, A

