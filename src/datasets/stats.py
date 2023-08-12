
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from typing import List

def get_mask_percentage(dataset: Dataset, rep=3):
    acc = 0
    count = 0
    for _ in range(rep):
        for i in range(len(dataset)):
            _, masks, _ = dataset[i]
            d_seq = masks.shape[0]
            acc += d_seq * 100 * masks.sum() / masks.numel()
            count += d_seq

    return acc / count

def get_doy_distribution(dataset: Dataset, rep=3):
    doys = []
    for _ in range(rep):
        for i in range(len(dataset)):
            _, _, doy = dataset[i]
            doys += doy.tolist()
    return doys

def plot_time_serie(dataset: Dataset, index: int, rgb: List[int]=[2, 1, 0]):
    s2, masks, doy = dataset[index][:3]
    fig, axes = plt.subplots(2, s2.shape[0])

    for i, axs in enumerate(axes.T):
        ax1, ax2 = axs

        ax1.axis('off')
        ax1.imshow(masks[i], vmin=0, vmax=1)
        ax1.set_title(doy[i].item())

        ax2.axis('off')
        img = torch.movedim(s2[i][rgb], 0, -1)
        img = torch.clip(img * 3.33, 0, 1)
        ax2.imshow(img)

    plt.show()