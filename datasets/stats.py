
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import matplotlib.patches as patches

from typing import List

def get_mask_percentage(dataset: Dataset, rep=3):
    acc = 0
    count = 0
    for _ in range(rep):
        for i in range(len(dataset)):
            _, masks, _ = dataset[i][:3]
            d_seq = masks.shape[0]
            acc += d_seq * 100 * masks.sum() / masks.numel()
            count += d_seq

    return acc / count

def get_doy_distribution(dataset: Dataset, rep=3):
    doys = []
    for _ in range(rep):
        for i in range(len(dataset)):
            _, _, doy = dataset[i][:3]
            doys.append(doy.tolist())
    return doys

def plot_time_serie(s2, s2_occl, doy, s2_out=None, rgb: List[int]=[2, 1, 0]):
    fig, axes = plt.subplots(2 if s2_out is None else 3, s2.shape[0])

    for i, axs in enumerate(axes.T):
        ax1, ax2 = axs[:2]

        ax1.axis('off')
        img = torch.movedim(s2_occl[i][rgb], 0, -1)
        img = torch.clip(img * 3.33, 0, 1)
        ax1.imshow(img)
        ax1.set_title(doy[i].item())

        ax2.axis('off')
        img = torch.movedim(s2[i][rgb], 0, -1)
        img = torch.clip(img * 3.33, 0, 1)
        ax2.imshow(img)

        if s2_out is not None:
            ax3 = axs[2]
            ax3.axis('off')
            img = torch.movedim(s2_out[i][rgb], 0, -1)
            img = torch.clip(img * 3.33, 0, 1)
            ax3.imshow(img)

    plt.show()


def plot_attn(s2_occl, attn, i_seq, rgb: List[int]=[2,1,0]):
    n_heads, d_seq, _, _, _ = attn.shape
    _, _, h, w = s2_occl.shape

    fig, axes = plt.subplots(n_heads+1, d_seq)

    for ax in axes.flatten():
        ax.axis('off')

    for i, ax in enumerate(axes[0]):
        img = torch.movedim(s2_occl[i][rgb], 0, -1)
        img = torch.clip(img * 3.33, 0, 1)

        ax.imshow(img)
        if i == i_seq:
            rect = patches.Rectangle((-1, -1), h, w, linewidth=4, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    for j, axs in enumerate(axes[1:]):
        for i, ax in enumerate(axs):
            A = attn[j, i_seq, i]
            ax.imshow(A, vmin=0, vmax=1)
    
    plt.show()
