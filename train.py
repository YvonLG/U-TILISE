
import torch
from datasets import TimeSeriesDS, get_many_path_dicts, plot_time_serie

from pathlib import Path

path = Path('../data/BIHAR/')
path_dicts = get_many_path_dicts(path, sanity_check=False)

ds = TimeSeriesDS(
    path_dicts,
    dmax_seq=15,
    dmin_seq=5,
    occl_range=(5, 8),
    cloud_treshold=0.5,
    cloudfree_treshold=0.01,
    fullcover_treshold=0.9,
    use_quadrants=True,
    s2_nch=4,
    s2_ch=None,
    use_padding=True,
    use_transforms=True,
    seed=None,
    preprocess_file=Path('./preprocess.pickle')
)

plot_time_serie(ds, 6)



