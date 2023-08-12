
from pathlib import Path
from typing import *
import logging
from datetime import datetime

import rasterio as rio
import numpy as np
import pandas as pd

PathDicts: TypeAlias = List[Dict[str, Path]]

def get_path_dicts(path: Path, sanity_check: bool=False) -> PathDicts:
    """The subfolder architecture must be:
    path/
    ├── time_serie1/
    │   ├── s1.tif
    │   ├── s2.tif
    │   ├── s2cloudless.tif
    │   ├── s1_propeties.csv
    |   ├── s2_properties.csv
    │   └── tile.json
    ├── time_serie2/
    ..."""
    path_dicts = []

    for dir in path.iterdir():
        if not dir.is_dir():
            continue
        
        path_dict = {
            's1': dir / 's1.tif',
            's2': dir / 's2.tif',
            's2cloudless': dir / 's2cloudless.tif',
            's1_properties': dir / 's1_properties.csv',
            's2_properties': dir / 's2_properties.csv',
            'tile': dir / 'tile.json',
        }

        if sanity_check:
            with rio.open(path_dict['s2'], 'r') as src:
                data = src.read()
            n_null = np.count_nonzero(data == 0)
            if n_null > 10:
                logging.info(f"{path_dict['s2'].as_posix()} is corrupted. {100 * n_null / np.prod(data.shape):2.2f}% of null values.")
                continue

        path_dicts.append(path_dict)
    return path_dicts

def get_many_path_dicts(path: Path, sanity_check: bool=False) -> PathDicts:
    path_dicts = []
    for dir in path.iterdir():
        if not dir.is_dir():
            continue

        path_dicts += get_path_dicts(dir, sanity_check)
    return path_dicts

def read_tif(path: Path, indices: List[int]|None=None, quadrant: int|None=None, full_size: int=256):
    half_size = full_size // 2
    if quadrant is None:
        window = ((0, full_size), (0, full_size))
    elif quadrant == 0:
        window = ((0, half_size), (0, half_size))
    elif quadrant == 1:
        window = ((half_size, full_size), (0, half_size))
    elif quadrant == 2:
        window = ((half_size, full_size), (half_size, full_size))
    elif quadrant == 3:
        window = ((0, half_size), (half_size, full_size))
    else:
        raise ValueError
    
    with rio.open(path) as src:
        return src.read(indices, window=window)

def read_doy(properties_path: Path) -> List[int]:
    """Will only work if the property 'system:time_start' is defined."""
    properties = pd.read_csv(properties_path)
    dates =  properties['system:time_start'].tolist()
    return [datetime.fromtimestamp(d/1000).timetuple().tm_yday-1 for d in dates]

def get_rio_indices(nch, ch, indices):
    return np.concatenate([1+nch*idx+ch for idx in indices]).tolist()
