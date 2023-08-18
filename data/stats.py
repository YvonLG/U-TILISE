
from torch.utils.data import Dataset

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