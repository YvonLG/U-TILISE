
from typing import List, Tuple
import pickle
from pathlib import Path
import csv
from datetime import datetime

from torch.utils.data import Dataset
import torch
import numpy as np
import rasterio as rio

from .utils import PathDicts

class TimeSeriesDS(Dataset):
    """`TimeSeriesDS` elements are cloudfree time series partially occluded with cloud masks from the same time serie.
    Use `save_instance` and `load_instance` to avoid the preprocessing everytime."""
    def __init__(self,
                 path_dicts: PathDicts,
                 desired_len: int=10,
                 min_len: int=5,
                 occl_range: int|Tuple[int, int]=(4, 8),
                 cloud_treshold: float=0.5,
                 cloudfree_treshold: float=0.01,
                 fullcover_treshold: float=0.99,
                 use_quadrants: bool=True,
                 use_padding: bool=True,
                 use_transforms: bool=True,
                 continuous_sample: bool=True,
                 s2_nch: int=4,
                 s2_used_ch: List[int]|None=None,
                 s1_nch: int=2,
                 s1_used_ch: List[int]|None=None,
                 seed: int|None=None
                 ):
        """
        Arguments:
            `path_dicts`: output of `get_path_dict` like
            `desired_len`: the desired time serie length
            `min_len`: time series with less cloudfree images will be discarded
            `occl_range`: if `n` will occlude exactly `n` images,
            if `a, b` will randomly occlude between [a, b[ images
            `cloud_treshold`: above this value a pixel is considered cloudy on s2cloudless
            `cloudfree_treshold`: an image is cloudfree if it's percentage of cloudy pixel is below this value
            `fullcover_treshold`: an image is fullycovered if it's percentage of cloudy pixel is above this value
            `use_quadrants`: if true the images are split in four and the four quadrants time series are considered
            `use_padding`: if true all elements are padded to match `desired_len` and a mask is returned to indicate
            padding indices
            `use_transforms`: if true will randomly rot90 Hflip and Vflip the images
            `continuous_sample`: if cloudfree timesteps are ABCDE if `continuous_sample` you might get BCD otherwise
            you might get ACE
            `s2_nch`: the number of s2 channels in your dataset
            `s2_used_ch`: the s2 channels that will be read and returned, `None` defaults `[0, 1, ..., s2_nch-1]`
            `seed`: random seed
            """
        self.path_dicts = path_dicts
        self.desired_len = desired_len
        self.min_len = min_len
        self.occl_range = occl_range
        self.cloud_treshold = cloud_treshold
        self.cloudfree_treshold = cloudfree_treshold
        self.fullcover_treshold = fullcover_treshold
        self.use_quadrants = use_quadrants
        self.use_padding = use_padding
        self.use_transforms = use_transforms
        self.continuous_sample = continuous_sample
        self.s2_nch = s2_nch
        self.s2_used_ch = np.arange(s2_nch) if s2_used_ch is None else np.array(s2_used_ch)
        self.s1_nch = s1_nch
        self.s1_used_ch = np.arange(s1_nch) if s1_used_ch is None else np.array(s1_used_ch)

        self.rng = np.random.default_rng(seed)

        self.indices = []

        # iterate thru all time series to remove time series with less than `min_len`
        for idx, path_dict in enumerate(path_dicts):
            quadrants = [0, 1, 2, 3] if use_quadrants else [None]
            for quadrant in quadrants:
                
                cloudfree, _, _ = self.get_clouds_info(path_dict['s2cloudless'], quadrant)
                n_cloudfree = np.count_nonzero(cloudfree)

                if n_cloudfree < min_len:
                    break
                self.indices.append((idx, quadrant))
    
    def get_clouds_info(self, path: Path, quadrant: int|None=None):
        cloud_probas = self.read_tif(path, None, quadrant)
        cloud_masks = cloud_probas > self.cloud_treshold * 100
        cloud_agg = cloud_masks.mean(axis=(1, 2))

        cloudfree = cloud_agg < self.cloudfree_treshold
        fullcover = cloud_agg > self.fullcover_treshold
        return cloudfree, fullcover, cloud_masks
    
    def transforms(self, s2, s1, masks):
        # 90 deg. rotations
        k1 = self.rng.integers(0, 4)
        k2 = self.rng.integers(0, 4)
        s2 = np.rot90(s2, k1, (1, 2))
        s1 = np.rot90(s1, k1, (1, 2))
        masks = np.rot90(masks, k2, (1, 2))

        # vertical and horizontal flips
        if self.rng.random() > 0.5:
            s2 = np.flip(s2, axis=1)
            s1 = np.flip(s1, axis=1)
        if self.rng.random() > 0.5:
            masks = np.flip(masks, axis=1)
        
        if self.rng.random() > 0.5:
            s2 = np.flip(s2, axis=2)
            s1 = np.flip(s1, axis=2)
            masks = np.flip(masks, axis=2)

        # copy to avoid negative stride
        return s2.copy(), s1.copy(), masks.copy()

    @staticmethod
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
    
    @staticmethod
    def read_doy(properties_path: Path) -> List[int]:
        with open(properties_path, 'r') as src:
            reader = csv.reader(src)
            header = next(reader)
            i = header.index('system:time_start')
            doy = [int(row[i]) for row in reader]
            doy = [datetime.fromtimestamp(d/1000).timetuple().tm_yday-1 for d in doy]
            return doy

    def save_instance(self, path: Path):
        with open(path, 'wb') as src:
            pickle.dump(self, src)

    @classmethod
    def from_instance(cls, path: Path):
        with open(path, 'rb') as src:
            instance = pickle.load(src)
        if isinstance(instance, TimeSeriesDS):
            return instance
        raise ValueError

    def __getitem__(self, index):
        idx, quadrant = self.indices[index]
        path_dict = self.path_dicts[idx]
        
        cloudfree, fullcover, cloud_masks = self.get_clouds_info(path_dict['s2cloudless'], quadrant)

        cloudfree_idx = np.where(cloudfree)[0]
        n_cloudfree = len(cloudfree_idx)

        partialcover = (~fullcover) & (~cloudfree)
        partialcover_idx = np.where(partialcover)[0]

        if n_cloudfree >= self.desired_len:
            n_images = self.desired_len

            if self.continuous_sample:
                start = self.rng.integers(0, n_cloudfree-self.desired_len+1)
                cloudfree_idx = cloudfree_idx[start:start+n_images]
            else:
                cloudfree_idx = sorted(self.rng.choice(cloudfree_idx, n_images, replace=False))
        else:
            n_images = n_cloudfree

        # number of images to occlude
        if isinstance(self.occl_range, int):
            n_occl = min(self.occl_range, n_images)
        else:
            occl_range = (min(self.occl_range[0], n_images), (min(self.occl_range[1], n_images+1)))
            n_occl = self.rng.integers(*occl_range)
        
        # where to occlude
        occl_idx = self.rng.choice(n_images, n_occl, replace=False)

        # where to get the cloud masks
        masks_idx = self.rng.choice(partialcover_idx, n_occl, replace=len(partialcover_idx) < n_occl)

        s2_idx = np.concatenate([1+self.s2_nch*idx+self.s2_used_ch for idx in cloudfree_idx]).tolist()
        s2 = self.read_tif(path_dict['s2'], s2_idx, quadrant)
        h, w = s2.shape[1:]

        s1_idx = np.concatenate([1+self.s1_nch*idx+self.s1_used_ch for idx in cloudfree_idx]).tolist()
        s1 = self.read_tif(path_dict['s1'], s1_idx, quadrant)

        masks = np.zeros((n_images, h, w), dtype=bool)
        masks[occl_idx] = cloud_masks[masks_idx]

        if self.use_transforms:
            s2, s1, masks = self.transforms(s2, s1, masks)
        
        s2 = s2.reshape(n_images, len(self.s2_used_ch), w, h)
        s1 = s1.reshape(n_images, len(self.s1_used_ch), w, h)

        doy = np.array(self.read_doy(path_dict['s2_properties']))
        doy = doy[cloudfree_idx]

        padding_size = self.desired_len - n_images
        if self.use_padding and padding_size > 0:
            padding = np.ones((padding_size, len(self.s2_used_ch), h, w))
            s2 = np.concatenate([s2, padding], axis=0)

            padding = np.ones((padding_size, len(self.s1_used_ch), h, w))
            s1 = np.concatenate([s1, padding], axis=0)

            masks = np.concatenate([masks, padding[:,0]], axis=0)

            doy = np.concatenate([doy, np.zeros(padding_size)])
        
        s2 = np.clip(s2, 0, 10000) / 10000
        s2 = torch.Tensor(s2).to(dtype=torch.float32)
        s1 = np.clip(s1, 0, 50000) / (-1000)
        s1 = (s1 + 50) / 50
        s1 = torch.Tensor(s1).to(dtype=torch.float32)
        masks = torch.Tensor(masks).to(dtype=torch.bool)
        doy = torch.Tensor(doy).to(dtype=torch.int16)

        if self.use_padding:
            pad_mask = torch.cat([torch.zeros(n_images, dtype=torch.bool), torch.ones(padding_size, dtype=torch.bool)])
            return s2, s1, masks, doy, pad_mask
        return s2, s1, masks, doy 

    def __len__(self):
        return len(self.indices)
