
import numpy as np
import copy

import torch

def baseline_interpolate(s2: torch.Tensor, masks: torch.Tensor, doy: torch.Tensor):
    """Linear interpolation between unmasked pixels."""
    s2 = s2.numpy()
    masks = masks.numpy()
    doy = doy.numpy()

    T, C, H, W = s2.shape

    s2_interp = copy.deepcopy(s2)

    for c in range(C):
        for h in range(H):
            for w in range(W):

                mask = masks[:,h,w]
                xp = doy[~mask]
                fp = s2[~mask,c,h,w]
                x = doy[mask]

                if len(xp) == 0:
                    s2_interp[mask,c,h,w] = 1
                else:
                    s2_interp[mask,c,h,w] = np.interp(x, xp, fp)

    return torch.from_numpy(s2_interp)
    

def baseline_nearest(s2: torch.Tensor, masks: torch.Tensor, doy: torch.Tensor):
    """Masked pixels take the values of the nearest (day-wise) unmasked pixels."""
    s2 = s2.numpy()
    masks = masks.numpy()
    doy = doy.numpy()

    T, C, H, W = s2.shape

    s2_nearest = copy.deepcopy(s2)

    for c in range(C):
        for h in range(H):
            for w in range(W):
                
                mask = masks[:,h,w]
                xp = doy[~mask]
                fp = s2[~mask,c,h,w]
                x = doy[mask]

                if len(xp) == 0:
                    s2_nearest[mask,c,h,w] = 1
                else:
                    diff = np.abs(x[:, np.newaxis] - xp) # len(x) x len(xp)
                    idx = np.argmin(diff, axis=1)
                    
                    s2_nearest[mask,c,h,w] = fp[idx]
    
    return torch.from_numpy(s2_nearest)

def metric_mae(gt: torch.Tensor, pred: torch.Tensor):
    gt = gt.flatten()
    pred = pred.flatten()

    return torch.abs(gt - pred).mean()

def metric_rmse(gt: torch.Tensor, pred: torch.Tensor):
    gt = gt.flatten()
    pred = pred.flatten()

    return torch.sqrt(torch.mean((gt - pred) ** 2))

def metric_psnr(gt: torch.Tensor, pred: torch.Tensor):
    return 20 * torch.log10(1/metric_rmse(gt, pred))

def metric_sam(gt: torch.Tensor, pred: torch.Tensor, ch_dim=1):
    dot_product = (gt * pred).sum(dim=ch_dim)
    gt_mag = gt.norm(dim=ch_dim)
    pred_mag = pred.norm(dim=ch_dim)

    sam = dot_product / (gt_mag * pred_mag)
    return sam.acos().mean().rad2deg()



