
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np

from typing import List


def plot_gt_pred_heatmap(gt: torch.Tensor, pred: torch.Tensor):
    gt = gt.numpy()
    pred = pred.numpy()

    fig, ax = plt.subplots()
    
    _, _, _, im = ax.hist2d(gt, pred, bins=(200, 200), range=((0,1), (0,1)), norm=mpl.colors.LogNorm(), cmap='Oranges')
    ax.plot((0,1), (0,1), linewidth=0.5, color='k', linestyle='dashed')
    ax.grid(linestyle='dashed')
    
    fig.colorbar(im, ax=ax, orientation='vertical')
    
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Prediction')

    return ax

def plot_attn_timestamp(attn: torch.Tensor, timestamp: int):
    attn = attn.numpy()
    N, T, _, H, W = attn.shape

    fig, axes = plt.subplots(N, T)
    axes = axes.reshape(N, T)

    for n in range(N):
        for t in range(T):
            ax = axes[n,t]
            ax.set_xticks([])
            ax.set_yticks([])

            ax.imshow(attn[n, timestamp, t], vmin=0, vmax=1, cmap='magma')
            
            if t == timestamp:
                ax.spines[:].set_color('r')

            if t == 0:
                ax.set_ylabel(f'head n°{n+1}')
    
    return axes

def plot_attn_head(attn: torch.Tensor, head: int):
    attn = attn.numpy()
    N, T, _, H, W = attn.shape

    fig, axes = plt.subplots(T, T)
    axes = axes.reshape(T, T)

    for t1 in range(T):
        for t2 in range(T):
            ax = axes[t1,t2]
            ax.set_xticks([])
            ax.set_yticks([])

            ax.imshow(attn[head, t1, t2], vmin=0, vmax=1, cmap='magma')

            if t1 == t2:
                ax.spines[:].set_color('r')

    return axes

def plot_timeseries(s2: torch.Tensor|List[torch.Tensor], channels: List[int]=[2, 1, 0], bright_coef: float=3.33):
    if not isinstance(s2, List):
        s2 = [s2]
    
    N = len(s2)
    T = s2[0].shape[0]

    fig, axes = plt.subplots(N, T, figsize=(30, 7))
    axes = axes.reshape(N, T)

    for n in range(N):
        for t in range(T):
            ax = axes[n, t]
            ax.set_xticks([])
            ax.set_yticks([])
            
            img = np.clip(s2[n][t, channels].numpy() * bright_coef, 0, 1)
            img = np.moveaxis(img, 0, -1)
            ax.imshow(img)
    
    return axes

