import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from functools import reduce
import operator

eps = 1e-8

class LinearVDO(nn.Module):
    """
    Dense layer implementation with weights ARD-prior (arxiv:1701.05369)
    """

    def __init__(self, in_features, out_features, bias=True, thresh=3, ard_init=-8.):
        super(LinearVDO, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.thresh = thresh
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.ard_init = ard_init
        self.log_alp = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)

        self.reset_parameters()

    def forward(self, input):
        """
        Forward with all regularized connections and random activations (Beyesian mode). Typically used for train
        """
        # if self.training == False: return F.linear(input, self.weights_clipped, self.bias)

        W = self.weight
        mu = input.matmul(W.t())

        eps = 1e-8
        log_alp = self.log_alp

        in2 = input * input
        exp_ = torch.exp(log_alp)
        w2 = self.weight * self.weight

        var = in2.matmul(((exp_ * w2) + eps).t())

        si = torch.sqrt(var)

        activation = mu + torch.normal(torch.zeros_like(mu), torch.ones_like(mu)) * si
        return activation + self.bias

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def reset_parameters(self):
        self.weight.data.normal_(std=0.01)
        if self.bias is not None:
            self.bias.data.uniform_(0, 0)
        self.log_alp.data = self.ard_init * torch.ones_like(self.log_alp)

    @staticmethod
    def clip(tensor, to=10.):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    @staticmethod
    def clip_alp(tensor, lwrb=20.):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -lwrb, -eps)

    def get_clip_mask(self):
        log_alp = self.clip_alp(self.log_alp)
        return torch.ge(log_alp, self.thresh)

    def train(self, mode):
        self.training = mode
        super(LinearVDO, self).train(mode)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        # a flexible reparameterization of variance

        k1 = 0.6134
        k2 = 0.2026
        k3 = 0.7126

        log_alp = self.log_alp

        element_wise_kl = -.5 * torch.log(1 + 1. / (torch.exp(log_alp))) \
                          + k1 * torch.exp(-(k2 + k3 * log_alp) ** 2)

        sum_kl = element_wise_kl.mean(dim=(1,))

        return - sum_kl.sum()
        # return -torch.mean(minus_kl)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (with log alpha greater than "thresh" parameter)
        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        eps = 1e-8
        return self.log_sigma2 - 2 * torch.log(torch.abs(self.weight) + eps)


class Conv2dVDO(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, ard_init=-1., thresh=3, weight_prob_fwd=True):
        bias = False  # Goes to nan if bias = True
        super(Conv2dVDO, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)
        self.bias = None
        self.thresh = thresh
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ard_init = ard_init
        # self.log_sigma2 = Parameter(ard_init * torch.ones_like(self.weight))
        self.log_alp = Parameter(ard_init * torch.ones_like(self.weight), requires_grad=True)
        self.weight_prob_fwd = weight_prob_fwd

        # self.log_sigma2 = Parameter(2 * torch.log(torch.abs(self.weight) + eps).clone().detach()+ard_init*torch.ones_like(self.weight))

    @staticmethod
    def clip(tensor, to=8):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -to, to)

    @staticmethod
    def clip_alp(tensor, lwrb=10.):
        """
        Shrink all tensor's values to range [-to,to]
        """
        return torch.clamp(tensor, -lwrb, -eps)

    def set_weight_prob_fwd(self, weight_prob_fwd):
        assert type(weight_prob_fwd) is bool
        self.weight_prob_fwd = weight_prob_fwd

    def forward(self, input):
        """
        Forward with all regularized connections and random activations (Beyesian mode). Typically used for train
        """
        if self.training == False and self.weight_prob_fwd == False:
            return F.conv2d(input, self.weights_clipped,
                            self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        eps = 1e-8
        W = self.weight
        zeros = torch.zeros_like(W)
        clip_mask = self.get_clip_mask()
        conved_mu = F.conv2d(input, W, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)

        log_alpha = self.log_alp
        # log_alpha = self.log_alpha

        conved_si = torch.sqrt(eps + F.conv2d(input * input,
                                              torch.exp(log_alpha) * W * W, self.bias, self.stride,
                                              self.padding, self.dilation, self.groups))

        conved = conved_mu + \
                 conved_si * torch.normal(torch.zeros_like(conved_mu), torch.ones_like(conved_mu))
        return conved

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def get_clip_mask(self):
        log_alp = self.clip_alp(self.log_alp)
        # log_alp = self.clip_alp(self.log_alpha)

        return torch.ge(log_alp, self.thresh)

    def train(self, mode):
        self.training = mode
        super(Conv2dVDO, self).train(mode)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """

        # param 1
        # k1 = 0.792
        # k2 = -0.4826
        # k3 = 0.3451

        # param 2
        k1 = 0.6134
        k2 = 0.2026
        k3 = 0.7126

        log_alp = self.log_alp

        element_wise_kl = -.5 * torch.log(1 + 1./(torch.exp(log_alp))) \
                          + k1 * torch.exp(-(k2 + k3 * log_alp) ** 2)

        sum_kl = element_wise_kl.mean(dim=(1, 2, 3))
        return - sum_kl.sum()

        # log_alp = self.clip_alp(self.log_alp)
        # # log_alp = self.clip_alp(self.log_alpha)
        # # mdkl = k1 * torch.sigmoid(k2 + k3 * log_alp2) - 0.5 * torch.log1p(torch.exp(-log_alp2)) + C
        # minus_kl = .5 * log_alp \
        #            + 1.16145124 * torch.exp(log_alp) \
        #            - 1.50204118 * torch.exp(log_alp)**2 \
        #            + 0.58629921 * torch.exp(log_alp)**3
        #
        # return -torch.sum(minus_kl)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (greater than "thresh" parameter)
        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        eps = 1e-8
        return self.log_sigma2 - 2 * torch.log(torch.abs(self.weight) + eps)
