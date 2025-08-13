# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 13:55:54 2025

@author: jlmorgan
"""

import torch
from torch import nn
from scipy import ndimage
import numpy as np


class Generic_ChatGPT_MaskedWeightedBCE(nn.Module):
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
        
class one_pt_cpu_test(nn.Module):
    def __init__(self, pos_weight=1.0, eps=1e-6, reduction='mean'):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, batch_pts, valid_mask=None):
        """
        logits: (N, ...) raw scores (pre-sigmoid)
        target: list of pts
        valid_mask: (N, ...) boolean or 0/1; optional

        returns scalar loss if reduction='mean' or 'sum',
        else per-element loss if reduction='none'
        """
        
        mask_tensor = logits > logits.mean()
        loss = []
        for bi, pts in enumerate(batch_pts):
            mask_np = mask_tensor[bi,0,:].cpu().numpy()
            labels_np, num = ndimage.label(mask_np)
            subs = np.floor(pts).astype(int)
            hits = labels_np[subs[:,0], subs[:,1], subs[:,2]]
            bins = np.arange(-0.5, num + 1, 1.0)
            hist, bin_edges = np.histogram(hits, bins)
            target = np.ones(num + 1)
            target[0] = 0
            wrong = np.absolute(target - hist)
            right = (target==hist).sum()
            rms = (wrong ** 2).mean() ** (1/2)
            loss.append(torch.from_numpy(rms))
       
        # if self.pos_weight != 1.0:
        #     loss = torch.where(target > 0, loss * self.pos_weight, loss)

        # if valid_mask is not None:
        #     loss = loss * valid_mask

        # if self.reduction == 'mean':
        #     denom = (valid_mask.sum() if valid_mask is not None else loss.numel()).clamp_min(1)
        #     return loss.sum() / denom
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # else:
        #     return loss


def _make_grid_zyx(shape, device, dtype):
    Z, Y, X = shape
    zz = torch.arange(Z, device=device, dtype=dtype).view(Z,1,1)
    yy = torch.arange(Y, device=device, dtype=dtype).view(1,Y,1)
    xx = torch.arange(X, device=device, dtype=dtype).view(1,1,X)
    return zz, yy, xx  # broadcastable (Z,Y,X)

class SoftAssignPointLoss(nn.Module):
    """
    Per-point soft-assignment loss.
    - logits: [N,1,Z,Y,X] (or [N,1,H,W] for 2D; works if Z=1)
    - batch_pts: list of length N; each is tensor [P_i, 3] in (z,y,x) (floats OK)
    - sigma: spatial scale of the kernel in voxels
    - alpha_bg: background weight >0 to allow some mass unassigned (stability)
    """
    def __init__(self, sigma=2.0, alpha_bg=0.0, reduction="mean", eps=1e-8):
        super().__init__()
        self.sigma = float(sigma)
        self.alpha_bg = float(alpha_bg)
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, batch_pts):
        assert logits.dim() in (4,5), "expect [N,1,H,W] or [N,1,Z,Y,X]"
        if logits.dim() == 4:
            logits = logits.unsqueeze(2)  # [N,1,1,H,W] -> treat as Z=1

        N, C, Z, Y, X = logits.shape
        assert C == 1
        device, dtype = logits.device, logits.dtype
        s = torch.sigmoid(logits)  # [N,1,Z,Y,X]

        losses = []
        for i in range(N):
            pts = batch_pts[i]
            if pts.numel() == 0:
                # No points: optional—penalize any mass (false positives)
                # Here, just encourage s to be small.
                loss_empty = s[i].mean()
                losses.append(loss_empty)
                continue

            pts = pts.to(device=device, dtype=dtype)  # [P,3]
            P = pts.shape[0]
            zz, yy, xx = _make_grid_zyx((Z, Y, X), device=device, dtype=dtype)

            # squared distances per point to each voxel: [P,Z,Y,X]
            dz2 = (zz.unsqueeze(0) - pts[:, 0].view(P,1,1,1))**2
            dy2 = (yy.unsqueeze(0) - pts[:, 1].view(P,1,1,1))**2
            dx2 = (xx.unsqueeze(0) - pts[:, 2].view(P,1,1,1))**2
            d2  = dz2 + dy2 + dx2

            K = torch.exp(-d2 / (2 * self.sigma**2))  # [P,Z,Y,X]

            # soft responsibilities r_{p,v} (normalize over points + optional background)
            denom = K.sum(dim=0, keepdim=False)  # [Z,Y,X]
            if self.alpha_bg > 0:
                denom = denom + self.alpha_bg

            r = K / (denom.clamp_min(self.eps))  # [P,Z,Y,X]

            # mass collected by each point: m_p = sum_v r_{p,v} * s_v
            s_i = s[i, 0]  # [Z,Y,X]
            m = (r * s_i).sum(dim=(1,2,3))  # [P]

            # encourage each m_p to be close to 1 (or at least >0). Use -log for a stable push
            point_term = -torch.log(m.clamp_min(self.eps)).mean()

            # optional: penalize leftover mass (not assigned to any point)
            if self.alpha_bg > 0:
                leftover = (self.alpha_bg / (denom.clamp_min(self.eps))) * s_i  # fraction to background
                bg_term = leftover.mean()
                loss_i = point_term + 0.5 * bg_term
            else:
                loss_i = point_term

            losses.append(loss_i)

        loss = torch.stack(losses).mean() if self.reduction == "mean" else torch.stack(losses).sum()
        return loss
        
        
        
        
        
        
        
        
        
        
        
        
        