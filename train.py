
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from datasets import TimeSeriesDS, get_many_path_dicts, plot_time_serie
from models import UTILISE, initialize_weights

from pathlib import Path



path = Path('../data/BIHAR/')
path_dicts = get_many_path_dicts(path, sanity_check=True)

dataset = TimeSeriesDS(
    path_dicts,
    dmax_seq=10,
    dmin_seq=5,
    occl_range=(3, 5),
    cloud_treshold=0.5,
    cloudfree_treshold=0.01,
    fullcover_treshold=0.9,
    use_quadrants=True,
    s2_nch=4,
    s2_ch=[0, 1, 2],
    use_padding=True,
    use_transforms=True,
    sample_cloudfree=False,
    seed=None,
    preprocess_file=None
)

utilise = UTILISE(
    input_nc=3,
    output_nc=3,
    nch=64,
    n_depth=3,
    activ='relu',
    norm='batch',
    padding='reflect',
    n_groups=8,
    upsample='bilinear',
    n_heads=4,
    d_key=4,
    mlp_nc=[256],
    attn_dropout=0.1,
    pos_encoding=False
)

beta1 = 0.9
lr = 2e-4
batch_size = 1
epochs = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

utilise = utilise.to(device)
initialize_weights(utilise, 'kaiming')
#utilise.load_state_dict(torch.load('./model4.pt'))

dataloader = DataLoader(dataset, batch_size)

optim = Adam(utilise.parameters(), lr, (beta1, 0.999))
criterion = nn.L1Loss().to(device)

for epoch in range(1, epochs+1):
    
    n = len(dataloader)
    losses = []
    count = 0
    pbar = tqdm(enumerate(dataloader), total=n)
    for i, data in pbar:

        if len(data) == 4:
            s2, masks, doy, pad_mask = data
        else:
            s2, masks, doy = data
            pad_mask = torch.zeros((s2.size(0), s2.size(1)), dtype=bool)

        s2 = s2.to(device)
        masks = masks.to(device)
        doy = doy.to(device)
        pad_mask = pad_mask.to(device)

        masks_broadcast = masks.unsqueeze(2).expand_as(s2)
        s2_occl = torch.where(masks_broadcast, 1., s2)

        s2_out = utilise(s2_occl, doy, pad_mask)
        loss = criterion(s2_out[~pad_mask], s2[~pad_mask])

        optim.zero_grad()
        loss.backward()
        optim.step()

        bs = s2.size(0)
        losses.append((loss * bs).detach().cpu().item())
        count += bs

        pbar.set_description(f'{sum(losses)/count:.4f}')

    pbar.close()

    torch.save(utilise.state_dict(), './model4.pt')







