import torch
import torch.nn as nn
from einops import repeat
from .nystrom_attention import NystromAttention
from modules.emb_position import *

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = head,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x, need_attn=False):
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
            x = x+z
            return x,attn
        else:
            x = x + self.attn(self.norm(x))
            return x  

class SAttention(nn.Module):

    def __init__(self,mlp_dim=512,pos_pos=0,pos='ppeg',peg_k=7,head=8):
        super(SAttention, self).__init__()
        self.norm = nn.LayerNorm(mlp_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))

        self.layer1 = TransLayer(dim=mlp_dim,head=head)
        self.layer2 = TransLayer(dim=mlp_dim,head=head)

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(512,k=peg_k)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

    # Modified by MAE@Meta
    def masking(self, x, ids_shuffle=None,len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        assert ids_shuffle is not None

        _,ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False,mask_enable=False):
        batch, num_patches, C = x.shape 
        
        attn = []

        if self.pos_pos == -2:
            x = self.pos_embedding(x)
        
        # masking
        if mask_enable and mask_ids is not None:
            x, _, _ = self.masking(x,mask_ids,len_keep)

        # cls_token
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = batch)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_pos == -1:
            x = self.pos_embedding(x)

        # translayer1
        if return_attn:
            x,_attn = self.layer1(x,True)
            attn.append(_attn.clone())
        else:
            x = self.layer1(x)

        # add pos embedding
        if self.pos_pos == 0:
            x[:,1:,:] = self.pos_embedding(x[:,1:,:])
        
        # translayer2
        if return_attn:
            x,_attn = self.layer2(x,True)
            attn.append(_attn.clone())
        else:
            x = self.layer2(x)

        #---->cls_token
        x = self.norm(x)

        logits = x[:,0,:]
 
        if return_attn:
            _a = attn
            return logits ,_a
        else:
            return logits
    