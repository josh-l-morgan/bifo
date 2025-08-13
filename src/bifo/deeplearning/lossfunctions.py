# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 13:55:54 2025

@author: jlmorgan
"""

import torch
from torch import nn

class MaskedWeightedBCE(nn.Module):
    def __init__(self, pos_weight=1.0, eps=1e-6, reduction='mean'):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, target, valid_mask=None):
        """
        logits: (N, ...) raw scores (pre-sigmoid)
        target: (N, ...) in {0,1} or ∈[0,1]
        valid_mask: (N, ...) boolean or 0/1; optional

        returns scalar loss if reduction='mean' or 'sum',
        else per-element loss if reduction='none'
        """
        # BCE with logits (stable)
        # log(1+exp(-|x|)) style for stability:
        maxv = torch.clamp_min(-logits, 0)                # = relu(-x)
        # -y*x + (1-y)*softplus(x)  ==> custom pos_weight via (y * w)
        loss = (1 - target) * (maxv + torch.log1p(torch.exp(-maxv - logits))) \
             + target * (maxv + torch.log1p(torch.exp(-maxv + logits)) - logits)

        if self.pos_weight != 1.0:
            loss = torch.where(target > 0, loss * self.pos_weight, loss)

        if valid_mask is not None:
            loss = loss * valid_mask

        if self.reduction == 'mean':
            denom = (valid_mask.sum() if valid_mask is not None else loss.numel()).clamp_min(1)
            return loss.sum() / denom
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss