
import torch
from models import UTILISE
from datasets import TimeSeriesDS, get_many_path_dicts, plot_time_serie, plot_attn
from pathlib import Path

import random

utilise = UTILISE(
    input_nc=3,
    output_nc=3,
    nch=64,
    n_depth=3,
    activ='relu',
    norm='batch',
    padding='reflect',
    n_groups=4,
    upsample='bilinear',
    n_heads=4,
    d_key=4,
    mlp_nc=[256],
    attn_dropout=0.,
    pos_encoding=True
)

path = Path('../data/BIHAR/')
path_dicts = get_many_path_dicts(path, sanity_check=False)
seed = random.randint(0, 1000)
seed = 995
dataset = TimeSeriesDS(
    path_dicts,
    dmax_seq=10,
    dmin_seq=5,
    occl_range=3,
    cloud_treshold=0.5,
    cloudfree_treshold=0.01,
    fullcover_treshold=0.99,
    use_quadrants=True,
    s2_nch=4,
    s2_ch=[0, 1, 2],
    use_padding=True,
    use_transforms=True,
    sample_cloudfree=False,
    seed=seed,
    preprocess_file=Path('./prepross.pickle')
)

utilise.load_state_dict(torch.load('./model4.pt'))

utilise.eval()
with torch.no_grad():
    s2, masks, doy, pad_mask = dataset[22]
    doy = torch.arange(1, 200, 20)


    s2 = s2.unsqueeze(0)
    masks = masks.unsqueeze(0)
    doy = doy.unsqueeze(0)
    pad_mask = pad_mask.unsqueeze(0)

    masks_broadcast = masks.unsqueeze(2).expand_as(s2)
    s2_occl = torch.where(masks_broadcast, 1., s2)

    s2_out, attn = utilise(s2_occl, doy, pad_mask, return_attn=True)
    s2_out = s2_out.squeeze().cpu()

print(f'seed={seed}')
plot_time_serie(s2.squeeze().cpu(), s2_occl.squeeze().cpu(), doy.squeeze().cpu(), s2_out)
plot_attn(s2_occl.squeeze().cpu(), attn.squeeze().cpu(), 0)