""" Dataset classes for loading the united (Cossad) dataset """

from torch.utils.data import Dataset
import glob
import os
import numpy as np
import logging
import constants
from . import util_dataloaders as util_dload
import re

logger = logging.getLogger(__name__)

def load_npz_as_native_dict(file_path):
    """Read a dictionary of values from .npz file and restores the original data types for the entries"""
    data = np.load(file_path, allow_pickle=True)  # in case of object types
    result = {}
    for key in data.files:
        value = data[key]
        # Automatically extract scalar strings or numbers
        if isinstance(value, np.ndarray) and value.shape == () and value.dtype.kind in {'U', 'S', 'i', 'f', 'b', 'O'}:
            result[key] = value.item()
        else:
            result[key] = value
    return result


class DatasetCossad(Dataset):
    """Loads a 3D point clouds from the unified Cossad dataset.
    This set is fairly small, and typically contains 4 objects per class. All labels are "good"
    """

    def __init__(self, cossad_dir, split, dataset_tpl='.*', class_tpl='.*', anomaly_tpl='.*', index_tpl='.*', normalize=False):
        """
        :param cossad_dir: the path to the Cossad dataset
        :param split: the split (i.e. 'template' or 'train')
        :param class_name: the name of the class (plane, car, ....  - see constants.py)
        """
        assert (split == 'template' or split == 'train')
        name_regex = fr'^.*/{dataset_tpl}_{class_tpl}_{index_tpl}_{anomaly_tpl}\.npz$'
        name_re_pattern = re.compile(name_regex)
        # self.files_list = glob.glob(os.path.join(cossad_dir, split, f'{dataset_tpl}_{class_tpl}_{index_tpl}_{anomaly_tpl}.npz'))
        all_file_names = glob.glob(os.path.join(cossad_dir, split, f'*.npz'))
        self.files_list = [f for f in all_file_names if name_re_pattern.fullmatch(f)]
        self.files_list.sort()
        self.normalize = normalize

    def __getitem__(self, idx):
        """Returns a tuple consisting of:
        - unordered point cloud as Numpy array
        - mask for each point in the pointcloud (0: good, 1: anomalous)
        - label for the whole object: 0: good; 1: anomalous
        - names of the file PC was loaded from
        """
        data = load_npz_as_native_dict(self.files_list[idx])
        if self.normalize:
            data['np_pointcloud'] = util_dload.normalize_pc(data['np_pointcloud'])
        data['npz_file'] = self.files_list[idx]
        return data

    def __len__(self):
        return len(self.files_list)
