import torch
import numpy as np
from torch import nn
from modules.datten import *
import torch.nn.functional as F
from modules.satten import *


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self, temp_t=1., temp_s=1.):
        super(SoftTargetCrossEntropy_v2, self).__init__()
        self.temp_t = temp_t
        self.temp_s = temp_s

    def forward(self, x: torch.Tensor, target: torch.Tensor, mean: bool = True) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target / self.temp_t, dim=-1) * F.log_softmax(x / self.temp_s, dim=-1), dim=-1)
        if mean:
            return loss.mean()
        else:
            return loss


class MHIM(nn.Module):
    def __init__(self, mlp_dim=512, mask_ratio=0, n_classes=2, temp_t=1., temp_s=1., dropout=0.25, act='relu',
                 select_mask=True, select_inv=False, msa_fusion='vote', mask_ratio_h=0., mrh_sche=None,
                 mask_ratio_hr=0., mask_ratio_l=0., da_act='gelu', baseline='selfattn', head=8, attn_layer=0, feat_dim=1024):
        super(MHIM, self).__init__()

        self.mask_ratio = mask_ratio
        self.mask_ratio_h = mask_ratio_h
        self.mask_ratio_hr = mask_ratio_hr
        self.mask_ratio_l = mask_ratio_l
        self.select_mask = select_mask
        self.select_inv = select_inv
        self.msa_fusion = msa_fusion
        self.mrh_sche = mrh_sche
        self.attn_layer = attn_layer

        self.patch_to_emb = [nn.Linear(feat_dim, mlp_dim)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        if baseline == 'selfattn':
            self.online_encoder = SAttention(mlp_dim=mlp_dim, head=head)
        elif baseline == 'attn':
            self.online_encoder = DAttention(mlp_dim, da_act)

        self.predictor = nn.Linear(mlp_dim, n_classes)

        self.temp_t = temp_t
        self.temp_s = temp_s

        self.cl_loss = SoftTargetCrossEntropy_v2(self.temp_t, self.temp_s)

        self.predictor_cl = nn.Identity()
        self.target_predictor = nn.Identity()

        self.apply(initialize_weights)

    def select_mask_fn(self, ps, attn, largest, mask_ratio, mask_ids_other=None, len_keep_other=None,
                       cls_attn_topk_idx_other=None, random_ratio=1., select_inv=False):
        ps_tmp = ps
        mask_ratio_ori = mask_ratio
        mask_ratio = mask_ratio / random_ratio
        if mask_ratio > 1:
            random_ratio = mask_ratio_ori
            mask_ratio = 1.

        # print(attn.size())
        if mask_ids_other is not None:
            if cls_attn_topk_idx_other is None:
                cls_attn_topk_idx_other = mask_ids_other[:, len_keep_other:].squeeze()
                ps_tmp = ps - cls_attn_topk_idx_other.size(0)
        if len(attn.size()) > 2:
            if self.msa_fusion == 'mean':
                _, cls_attn_topk_idx = torch.topk(attn, int(np.ceil((ps_tmp * mask_ratio)) // attn.size(1)),
                                                  largest=largest)
                cls_attn_topk_idx = torch.unique(cls_attn_topk_idx.flatten(-3, -1))
            elif self.msa_fusion == 'vote':
                vote = attn.clone()
                vote[:] = 0

                _, idx = torch.topk(attn, k=int(np.ceil((ps_tmp * mask_ratio))), sorted=False, largest=largest)
                mask = vote.clone()
                mask = mask.scatter_(2, idx, 1) == 1
                vote[mask] = 1
                vote = vote.sum(dim=1)
                _, cls_attn_topk_idx = torch.topk(vote, k=int(np.ceil((ps_tmp * mask_ratio))), sorted=False)
                # print(cls_attn_topk_idx.size())
                cls_attn_topk_idx = cls_attn_topk_idx[0]
        else:
            k = int(np.ceil((ps_tmp * mask_ratio)))
            _, cls_attn_topk_idx = torch.topk(attn, k, largest=largest)
            cls_attn_topk_idx = cls_attn_topk_idx.squeeze(0)

        # randomly
        if random_ratio < 1.:
            random_idx = torch.randperm(cls_attn_topk_idx.size(0), device=cls_attn_topk_idx.device)

            cls_attn_topk_idx = torch.gather(cls_attn_topk_idx, dim=0, index=random_idx[:int(
                np.ceil((cls_attn_topk_idx.size(0) * random_ratio)))])

        # concat other masking idx
        if mask_ids_other is not None:
            cls_attn_topk_idx = torch.cat([cls_attn_topk_idx, cls_attn_topk_idx_other]).unique()

        # if cls_attn_topk_idx is not None:
        len_keep = ps - cls_attn_topk_idx.size(0)
        a = set(cls_attn_topk_idx.tolist())
        b = set(list(range(ps)))
        mask_ids = torch.tensor(list(b.difference(a)), device=attn.device)
        if select_inv:
            mask_ids = torch.cat([cls_attn_topk_idx, mask_ids]).unsqueeze(0)
            len_keep = ps - len_keep
        else:
            mask_ids = torch.cat([mask_ids, cls_attn_topk_idx]).unsqueeze(0)

        return len_keep, mask_ids

    def get_mask(self, ps, i, attn, mrh=None):
        if attn is not None and isinstance(attn, (list, tuple)):
            if self.attn_layer == -1:
                attn = attn[1]
            else:
                attn = attn[self.attn_layer]
        else:
            attn = attn

        # random mask
        if attn is not None and self.mask_ratio > 0.:
            len_keep, mask_ids = self.select_mask_fn(ps, attn, False, self.mask_ratio, select_inv=self.select_inv,
                                                     random_ratio=0.001)
        else:
            len_keep, mask_ids = ps, None

        # low attention mask
        if attn is not None and self.mask_ratio_l > 0.:
            if mask_ids is None:
                len_keep, mask_ids = self.select_mask_fn(ps, attn, False, self.mask_ratio_l, select_inv=self.select_inv)
            else:
                cls_attn_topk_idx_other = mask_ids[:, :len_keep].squeeze() if self.select_inv else mask_ids[:,
                                                                                                   len_keep:].squeeze()
                len_keep, mask_ids = self.select_mask_fn(ps, attn, False, self.mask_ratio_l, select_inv=self.select_inv,
                                                         mask_ids_other=mask_ids, len_keep_other=ps,
                                                         cls_attn_topk_idx_other=cls_attn_topk_idx_other)

        # high attention mask
        mask_ratio_h = self.mask_ratio_h
        if self.mrh_sche is not None:
            mask_ratio_h = self.mrh_sche[i]
        if mrh is not None:
            mask_ratio_h = mrh
        if mask_ratio_h > 0.:
            # mask high conf patch
            if mask_ids is None:
                len_keep, mask_ids = self.select_mask_fn(ps, attn, largest=True, mask_ratio=mask_ratio_h,
                                                         len_keep_other=ps, random_ratio=self.mask_ratio_hr,
                                                         select_inv=self.select_inv)
            else:
                cls_attn_topk_idx_other = mask_ids[:, :len_keep].squeeze() if self.select_inv else mask_ids[:,
                                                                                                   len_keep:].squeeze()

                len_keep, mask_ids = self.select_mask_fn(ps, attn, largest=True, mask_ratio=mask_ratio_h,
                                                         mask_ids_other=mask_ids, len_keep_other=ps,
                                                         cls_attn_topk_idx_other=cls_attn_topk_idx_other,
                                                         random_ratio=self.mask_ratio_hr, select_inv=self.select_inv)

        return len_keep, mask_ids

    @torch.no_grad()
    def forward_teacher(self, x, return_attn=False):

        x = self.patch_to_emb(x)
        x = self.dp(x)

        if return_attn:
            x, attn = self.online_encoder(x, return_attn=True)
        else:
            x = self.online_encoder(x)
            attn = None

        return x, attn

    @torch.no_grad()
    def forward_test(self, x, return_attn=False, no_norm=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)

        if return_attn:
            x, a = self.online_encoder(x, return_attn=True, no_norm=no_norm)
        else:
            x = self.online_encoder(x)
        x = self.predictor(x)

        if return_attn:
            return x, a
        else:
            return x

    def pure(self, x, return_attn=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        ps = x.size(1)

        if return_attn:
            x, attn = self.online_encoder(x, return_attn=True)
        else:
            x = self.online_encoder(x)

        x = self.predictor(x)

        if self.training:
            if return_attn:
                return x, 0, ps, ps, attn
            else:
                return x, 0, ps, ps
        else:
            if return_attn:
                return x, attn
            else:
                return x

    def forward_loss(self, student_cls_feat, teacher_cls_feat):
        if teacher_cls_feat is not None:
            cls_loss = self.cl_loss(student_cls_feat, teacher_cls_feat.detach())
        else:
            cls_loss = 0.

        return cls_loss

    def forward(self, x, attn=None, teacher_cls_feat=None, i=None):
        x = self.patch_to_emb(x)
        x = self.dp(x)

        ps = x.size(1)

        # get mask
        if self.select_mask:
            len_keep, mask_ids = self.get_mask(ps, i, attn)
        else:
            len_keep, mask_ids = ps, None

        # forward online network
        student_cls_feat = self.online_encoder(x, len_keep=len_keep, mask_ids=mask_ids, mask_enable=True)

        # prediction
        student_logit = self.predictor(student_cls_feat)

        # cl loss
        cls_loss = self.forward_loss(student_cls_feat=student_cls_feat, teacher_cls_feat=teacher_cls_feat)

        return student_logit, cls_loss, ps, len_keep
