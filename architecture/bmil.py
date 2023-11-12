import math
import numbers
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.utils import initialize_weights
from architecture.linear_vdo import LinearVDO, Conv2dVDO
import numpy as np
from torch.distributions import kl

EPS_1 = 1e-16
# EPS_2 = 1e-28

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        ard_init = -1.
        self.attention_a = [
            LinearVDO(L, D, ard_init=ard_init),
            nn.Tanh()]

        self.attention_b = [LinearVDO(L, D, ard_init=ard_init),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = LinearVDO(D, n_classes, ard_init=ard_init)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class DAttn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(DAttn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        # print(x.shape)
        return A, x


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        # return self.conv(input, weight=self.weight, groups=self.groups, dilation=2)
        return self.conv(input, weight=self.weight, groups=self.groups)


class probabilistic_MIL_Bayes_vis(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, n_classes=2, top_k=1):
        super(probabilistic_MIL_Bayes_vis, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=2)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-3.)
        self.n_classes = n_classes
        self.print_sample_trigger = False
        self.num_samples = 16
        self.temperature = torch.tensor([1.0])
        self.fixed_b = torch.tensor([5.], requires_grad=False)

        initialize_weights(self)
        self.top_k = top_k

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.temperature = self.temperature.to(device)

    def forward(self, h, validation=False):
        device = h.device
        # *-*# A, h = self.attention_net(h)  # NxK

        A, h = self.attention_net(h)

        mu = A[:, 0]
        logvar = A[:, 1]
        gaus_samples = self.reparameterize(mu, logvar)
        beta_samples = F.sigmoid(gaus_samples)
        A = beta_samples.unsqueeze(0)
        # print('gaus   max: {0:.4f}, gaus   min: {1:.4f}.'.format(torch.max(gaus_samples), torch.min(gaus_samples)))
        # print('sample max: {0:.4f}, sample min: {1:.4f}.'.format(torch.max(A), torch.min(A)))

        M = torch.mm(A, h) / A.sum()
        logits = self.classifiers(M)
        y_probs = F.softmax(logits, dim=1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1, )
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim=1)[1]
        Y_prob = F.softmax(top_instance, dim=1)
        # results_dict = {}

        # if return_features:
        #     top_features = torch.index_select(h, dim=0, index=top_instance_idx)
        #     results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, A


class probabilistic_MIL_Bayes_enc(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, n_classes=2, top_k=1):
        super(probabilistic_MIL_Bayes_enc, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        first_transform = nn.Linear(size[0], size[1])
        fc1 = [first_transform, nn.ReLU()]

        if dropout:
            fc1.append(nn.Dropout(0.25))

        if gate:
            postr_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=2)
        else:
            postr_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)

        fc1.append(postr_net)

        self.postr_net = nn.Sequential(*fc1)
        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-3.)

        self.n_classes = n_classes
        self.print_sample_trigger = False
        self.num_samples = 16
        self.temperature = torch.tensor([1.0])
        self.prior_mu = torch.tensor([-5., 0.])
        self.prior_logvar = torch.tensor([-1., 3.])

        initialize_weights(self)
        self.top_k = top_k

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.attention_net = self.attention_net.to(device)
        self.postr_net = self.postr_net.to(device)
        # self.prior_net = self.prior_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.temperature = self.temperature.to(device)
        self.prior_mu = self.prior_mu.to(device)
        self.prior_logvar = self.prior_logvar.to(device)

    def kl_logistic_normal(self, mu_pr, mu_pos, logvar_pr, logvar_pos):
        return (logvar_pr - logvar_pos) / 2. + (logvar_pos ** 2 + (mu_pr - mu_pos) ** 2) / (2. * logvar_pr ** 2) - 0.5

    def forward(self, h, return_features=False, slide_label=None, validation=False):
        device = h.device
        # *-*# A, h = self.attention_net(h)  # NxK

        param, h = self.postr_net(h)

        mu = param[:, 0]
        logvar = param[:, 1]
        gaus_samples = self.reparameterize(mu, logvar)
        beta_samples = F.sigmoid(gaus_samples)
        A = beta_samples.unsqueeze(0)

        if not validation:
            mu_pr = self.prior_mu[slide_label.item()].expand(h.shape[0])
            logvar_pr = self.prior_logvar[slide_label.item()]
            kl_div = self.kl_logistic_normal(mu_pr, mu, logvar_pr, logvar)
        else:
            kl_div = None

        M = torch.mm(A, h) / A.sum()

        logits = self.classifiers(M)

        y_probs = F.softmax(logits, dim=1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1, )
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim=1)[1]
        Y_prob = F.softmax(top_instance, dim=1)
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        if not validation:
            return top_instance, Y_prob, Y_hat, kl_div, y_probs, A
        else:
            return top_instance, Y_prob, Y_hat, y_probs, A



class probabilistic_MIL_Bayes_spvis(nn.Module):
    def __init__(self, conf, size_arg="small", top_k=1):
        super(probabilistic_MIL_Bayes_spvis, self).__init__()

        # self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict = {"small": [conf.feat_d, 512, 256], "big": [conf.feat_d, 512, 384]}
        size = self.size_dict[size_arg]

        ard_init = -4.
        self.linear1 = nn.Linear(size[0], size[1])
        self.linear2a = LinearVDO(size[1], size[2], ard_init=ard_init)
        self.linear2b = LinearVDO(size[1], size[2], ard_init=ard_init)
        self.linear3 = LinearVDO(size[2], 2, ard_init=ard_init)

        self.gaus_smoothing = GaussianSmoothing(1, 3, 0.5)

        self.classifiers = LinearVDO(size[1], conf.n_class, ard_init=-3.)

        self.dp_0 = nn.Dropout(0.25)
        self.dp_a = nn.Dropout(0.25)
        self.dp_b = nn.Dropout(0.25)

        self.prior_mu = torch.tensor([-5., 0.])
        self.prior_logvar = torch.tensor([-1., 3.])

        initialize_weights(self)
        self.top_k = top_k
        self.patch_size = conf.patch_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_logistic_normal(self, mu_pr, mu_pos, logvar_pr, logvar_pos):
        return (logvar_pr - logvar_pos) / 2. + (logvar_pos ** 2 + (mu_pr - mu_pos) ** 2) / (2. * logvar_pr ** 2) - 0.5

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.linear1 = self.linear1.to(device)
        self.linear2a = self.linear2a.to(device)
        self.linear2b = self.linear2b.to(device)
        self.linear3 = self.linear3.to(device)

        self.dp_0 = self.dp_0.to(device)
        self.dp_a = self.dp_a.to(device)
        self.dp_b = self.dp_b.to(device)
        self.gaus_smoothing = self.gaus_smoothing.to(device)

        self.prior_mu = self.prior_mu.to(device)
        self.prior_logvar = self.prior_logvar.to(device)

        self.classifiers = self.classifiers.to(device)

    def forward(self, h, coords, height, width, slide_label=None, validation=False):
        h = h[0]
        device = h.device
        h = F.relu(self.dp_0(self.linear1(h)))

        feat_a = self.dp_a(torch.sigmoid(self.linear2a(h)))
        feat_b = self.dp_b(torch.tanh(self.linear2b(h)))
        feat = feat_a.mul(feat_b)
        params = self.linear3(feat)

        coords = coords // self.patch_size
        asign = lambda coord: coord[:, 0] + coord[:, 1] * (width // self.patch_size)
        coords = asign(coords)
        coords = torch.from_numpy(coords).to(device)

        mu = torch.zeros([1, (height // self.patch_size + 1) * (width // self.patch_size + 1)]).to(device)
        logvar = torch.zeros([1, (height // self.patch_size + 1) * (width // self.patch_size + 1)]).to(device)

        mu[:, coords.long()] = params[:, 0]
        logvar[:, coords.long()] = params[:, 1]

        mu = mu.view(1, height // self.patch_size + 1, width // self.patch_size + 1)
        logvar = logvar.view(1, height // self.patch_size + 1, width // self.patch_size + 1)

        if not validation:
            mu_pr = self.prior_mu[slide_label.item()].expand_as(mu)
            logvar_pr = self.prior_logvar[slide_label.item()]
            kl_div = self.kl_logistic_normal(mu_pr, mu, logvar_pr, logvar)
        else:
            kl_div = None

        # # no branch
        mu = F.pad(mu, (1, 1, 1, 1), mode='constant', value=0)
        mu = torch.unsqueeze(mu, dim=0)
        mu = self.gaus_smoothing(mu)

        gaus_samples = self.reparameterize(mu, logvar)
        gaus_samples = torch.squeeze(gaus_samples, dim=0)

        A = F.sigmoid(gaus_samples)
        A = A.view(1, -1)

        patch_A = torch.index_select(A, dim=1, index=coords)
        M = torch.mm(patch_A, h) / patch_A.sum()

        logits = self.classifiers(M)

        y_probs = F.softmax(logits, dim=1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1, )
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim=1)[1]
        Y_prob = F.softmax(top_instance, dim=1)

        if not validation:
            return top_instance, Y_prob, Y_hat, kl_div, y_probs, patch_A.view((1, -1))
        else:
            return top_instance, Y_prob, Y_hat, y_probs, patch_A.view((1, -1))


def get_ard_reg_vdo(module, reg=0):
    """
    :param module: model to evaluate ard regularization for
    :param reg: auxilary cumulative variable for recursion
    :return: total regularization for module
    """
    if isinstance(module, LinearVDO) or isinstance(module, Conv2dVDO): return reg + module.get_reg()
    if hasattr(module, 'children'): return reg + sum([get_ard_reg_vdo(submodule) for submodule in module.children()])
    return reg


bMIL_model_dict = {
    'vis': probabilistic_MIL_Bayes_vis,
    'enc': probabilistic_MIL_Bayes_enc,
    'spvis': probabilistic_MIL_Bayes_spvis,
}