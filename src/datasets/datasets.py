
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np

from .utils import *

class TimeSeriesDS(Dataset):
    def __init__(self, path_dicts: PathDicts, dmax_seq: int=10, dmin_seq: int=5, occl_range: Tuple[int, int]=(4, 8), cloud_treshold: float=0.5,
                 cloudfree_treshold: float=0.01, fullcover_treshold: float=0.99, use_quadrants: bool=True, s2_nch: int=4, s2_ch: List[int]|None=None,
                 use_padding: bool=True, use_transforms: bool=True, seed: int|None=None, preprocess_file: Path|None=None,
                 ):
        self.path_dicts = path_dicts

        self.dmax_seq = dmax_seq
        self.dmin_seq = dmin_seq
        self.occl_range = occl_range
        self.cloud_treshold = cloud_treshold
        self.cloudfree_treshold = cloudfree_treshold
        self.fullcover_treshold = fullcover_treshold
        self.use_quadrants = use_quadrants
        self.s2_nch = s2_nch
        self.s2_ch = np.arange(s2_nch) if s2_ch is None else np.array(s2_ch)
        self.use_padding = use_padding
        self.use_transforms = use_transforms

        self.image_size = 128 if use_quadrants else 256

        self.rng = np.random.default_rng(seed=seed)

        self.preprocess(preprocess_file)

    def preprocess(self, preprocess_file: Path|None):
        
        if preprocess_file is not None and preprocess_file.exists():
            with open(preprocess_file, 'rb') as src:
                self.time_series = pickle.load(src)
                return

        time_series = []
        for idx, path_dict in enumerate(self.path_dicts):
            quadrants = [0, 1, 2, 3] if self.use_quadrants else [None]
            for quadrant in quadrants:
                
                cloudfree, _, _ = self.get_clouds_info(path_dict['s2cloudless'], quadrant)
                n_cloudfree = np.count_nonzero(cloudfree)

                if n_cloudfree < self.dmin_seq:
                    break
                time_series.append((idx, quadrant))

        self.time_series = time_series

        if preprocess_file is not None:
            with open(preprocess_file, 'wb') as src:
                pickle.dump(time_series, src)

    def get_clouds_info(self, path: Path, quadrant=None):
        clouds_proba = read_tif(path, quadrant=quadrant)
        clouds_masks = clouds_proba > self.cloud_treshold * 100
        clouds_agg = clouds_masks.sum(axis=(1, 2)) / self.image_size ** 2

        cloudfree = clouds_agg < self.cloudfree_treshold
        fullcover = clouds_agg > self.fullcover_treshold
        return cloudfree, fullcover, clouds_masks
    
    def transforms(self, s2, masks):
        # 90 deg. rotations
        k1 = self.rng.integers(0, 4)
        k2 = self.rng.integers(0, 4)
        s2 = np.rot90(s2, k1, (1, 2))
        masks = np.rot90(masks, k2, (1, 2))

        # vertical and horizontal flips
        if self.rng.random() > 0.5:
            s2 = np.flip(s2, axis=1)
        if self.rng.random() > 0.5:
            masks = np.flip(masks, axis=1)
        
        if self.rng.random() > 0.5:
            s2 = np.flip(s2, axis=2)
            masks = np.flip(masks, axis=2)

        # copy to avoid negative stride
        return s2.copy(), masks.copy()


    def __getitem__(self, index):
        idx, quadrant = self.time_series[index]
        path_dict = self.path_dicts[idx]

        cloudfree, fullcover, clouds_masks = self.get_clouds_info(path_dict['s2cloudless'], quadrant)

        # the dataset outputs cloudfree time series
        cloudfree_idx = np.where(cloudfree)[0]
        n_cloudfree = len(cloudfree_idx)

        # partially cloudy images from the same time serie are used to get artificial cloud masks
        partialcover = (~fullcover) & (~cloudfree)
        partialcover_idx = np.where(partialcover)[0]

        if n_cloudfree >= self.dmax_seq:
            start = self.rng.integers(0, n_cloudfree-self.dmax_seq+1)
            d_seq = self.dmax_seq
        
        else:
            start = 0
            d_seq = n_cloudfree

        cloudfree_idx = cloudfree_idx[start:start+d_seq]

        # choose the number of cloudfree images to mask
        occl_range = (min(self.occl_range[0], d_seq), (min(self.occl_range[1], 1+d_seq)))
        n_occl = self.rng.integers(*occl_range)

        # choose which images will be masked in the time serie
        occluded_idx = sorted(self.rng.choice(d_seq, n_occl, replace=False))

        # choose which masks to use among partially covered images
        occl_masks_idx = self.rng.choice(partialcover_idx, n_occl, replace=len(partialcover_idx) < n_occl)

        s2_indices = get_rio_indices(self.s2_nch, self.s2_ch, cloudfree_idx)
        s2 = read_tif(path_dict['s2'], s2_indices, quadrant=quadrant) / 10000
        s2 = s2.astype(np.float32)

        masks = np.zeros((d_seq, self.image_size, self.image_size), dtype=bool)
        masks[occluded_idx] = clouds_masks[occl_masks_idx]

        if self.use_transforms:
            s2, masks = self.transforms(s2, masks)

        # broadcast masks to the shape of s2 and mask s2
        masks_broadcast = np.repeat(masks, len(self.s2_ch), axis=0)
        s2[masks_broadcast] = 1.
        s2 = s2.reshape(d_seq, len(self.s2_ch), self.image_size, self.image_size)

        doy = np.array(read_doy(path_dict['s2_properties']), dtype=np.int16)
        doy = doy[cloudfree_idx]

        padding_size = self.dmax_seq - d_seq
        if padding_size > 0 and self.use_padding:
            s2_padding = np.ones((padding_size, len(self.s2_ch), self.image_size, self.image_size))
            s2 = np.concatenate([s2, s2_padding], axis=0)
            
            masks_padding = s2_padding[:,0]
            masks = np.concatenate([masks, masks_padding], axis=0)

            doy = np.concatenate([doy, np.zeros(padding_size)])
    
        s2 = torch.from_numpy(s2)
        masks = torch.from_numpy(masks)
        doy = torch.from_numpy(doy)

        if self.use_padding:
            pad_mask = torch.cat([torch.zeros(d_seq, dtype=bool), torch.ones(padding_size, dtype=bool)])
            return s2, masks, doy, pad_mask
        return s2, masks, doy

    def __len__(self):
        return len(self.time_series)