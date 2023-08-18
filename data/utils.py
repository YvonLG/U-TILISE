
from typing import List, Dict, TypeAlias
from pathlib import Path
import logging

import numpy as np
import rasterio as rio

PathDicts: TypeAlias = List[Dict[str, Path]]
logging.basicConfig(level=logging.INFO)

def get_path_dicts(path: Path, sanity_check: bool=False, sanity_check_treshold: int=100) -> PathDicts:
    """The subfolder architecture must be:
    `path`/
    ├── time_serie1/
    │   ├── s1.tif
    │   ├── s2.tif
    │   ├── s2cloudless.tif
    │   ├── s1_propeties.csv
    |   ├── s2_properties.csv
    │   └── tile.json
    ├── time_serie2/
    ...
    If `sanity_check` then all the s2 files will be opened and checked for an unusually high number of zeros.
    The corrupted files are not returned."""
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
            n_zeros = np.count_nonzero(data == 0)
            if n_zeros > sanity_check_treshold:
                logging.info(f"{path_dict['s2'].as_posix()} is corrupted. {100 * n_zeros / np.prod(data.shape):2.2f}% of zeros.")
                continue

        path_dicts.append(path_dict)
    return path_dicts

def get_many_path_dicts(path: Path, sanity_check: bool=False, sanity_check_treshold: int=100) -> PathDicts:
    """`path` must lead to a directory containing directories matching `get_path_dict` requirement's."""
    path_dicts = []
    for dir in path.iterdir():
        if not dir.is_dir():
            continue

        path_dicts += get_path_dicts(dir, sanity_check, sanity_check_treshold)
    return path_dicts

def split_path_dicts(path_dicts: PathDicts, proportions: List[float]):

    assert abs(sum(proportions) - 1) < 1e-5
    cumsum = [p + sum(proportions[:i]) for i, p in enumerate(proportions)]
    
    n_tot = len(path_dicts)
    all_indices = np.arange(n_tot)
    np.random.shuffle(all_indices)

    sections = [int(p*n_tot) for p in cumsum[:-1]]
    indices_sections = np.split(all_indices, sections)

    return [[path_dicts[i] for i in indices] for indices in indices_sections]

if __name__ == '__main__':
    path_dicts = get_many_path_dicts(Path('../data/BIHAR'), sanity_check=True)

    path_dicts_t, path_dicts_v = split_path_dicts(path_dicts, [0.8, 0.2])
    print(len(path_dicts_t), len(path_dicts_v))


    import pickle
    with open('../data/train.pickle', 'wb') as src:
        pickle.dump(path_dicts_t, src)

    with open('../data/valid.pickle', 'wb') as src:
        pickle.dump(path_dicts_t, src)
