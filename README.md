This repo contains an unofficial implementation of the model U-TILISE from **"U-TILISE: A Sequence-to-sequence Model for Cloud Removal in Optical Satellite Time Series"** (https://arxiv.org/abs/2305.13277). The official implementation is available here https://github.com/prs-eth/U-TILISE.

Use the utilities in data/utils.py to get the prepare the data. Make sure the data on your hard drive is organized with the expected folder structure.

```python
from data.utils import get_many_path_dicts
from data.dataloader import TimeSeriesDS

paths = get_many_path_dicts(Path('/.../data/BIHAR')

ds = TimeSeriesDS(paths, ...)
```

To train the model, change the configurations as desired in train.py and run it. As it is set up, it will require logging to a wandb session.

```python
from model.utilise import UTILISE
from valid.visu import plot_timeseries

# create the model from a config file
model = UTILISE(**configs)

# load the weights
model.load_state_dict(torch.load("weights.pt"))
model.eval()

s2, s1, masks, doy = ds[0]
s2 = s2.unsqueeze(0)
s1 = s1.unsqueeze(0)
masks = masks.unsqueeze(0)
doy = doy.unsqueeze(0)

# the dataset gives pairing of cloud-free s2 and s1 images time series.
# the following step artificially "clouds" part of the s2 time series.
masks_broadcast = masks.unsqueeze(2).expand_as(s2)
s2_occl = torch.where(masks_broadcast, 1., s2)

s2_and_s1 = torch.cat([s2_occl, s1], dim=2)
s2_out = model(s2_and_s1, doy, pad_mask)

plot_timeseries(s2_out)
```

The data can be exported to your **google** drive from **google** earth engine using download_data.ipynb in a **google** colab session. NOTE: the script seems to be broken at the moment, working on it.







