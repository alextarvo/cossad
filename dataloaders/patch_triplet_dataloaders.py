"""Reader to read contrastive patches from the disk"""

from zipfile import ZipFile, BadZipFile
import logging
import re
from typing import Any, List, DefaultDict, Dict, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict
from itertools import chain
import random

from sympy import false
from pathlib import Path

from xarray.ufuncs import positive

import utils.pc_utils as pc_utils
from torch.utils.data._utils.collate import default_collate

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from .cossad_dataloaders import load_npz_as_native_dict

import constants
from .pc_triplet_dataloaders import triplet_dict_collate

logger = logging.getLogger(__name__)

POS_NEG_FILE_FILTER = '[A-Za-z0-9]+_[A-Za-z0-9]+_[0-9]{5}\.npz$'
ALL_POS_FILE_FILTER = '[A-Za-z0-9]+_[A-Za-z0-9]+_[0-9]{1,3}\.npz$'


def get_patch_class_id(object_class: str, start_loc_id: int, loc_id_no: int, patch_source:constants.PatchSource) -> int:
    """Get a 'class' ID for the patch. Class ID is a combination of object class, location, and patch type (good/bad).
        class ID is even if the patch is good, and odd if it is anomalous.

    Args:
        object_class: Name of the object class
        start_loc_id: The initial location of a patch sequence. Each object_class will have its own range of
            locations. These start from 0 if we want to assign locations for the whole PC using FPS algorithm based on
            a template. They start from MIN_DYNAMIC_PATCH_ID if we assign a location for a dynamic range of location
            IDs for a tuple of (bad, good1, ..., goodN) PCs.
        loc_id_no: An index of the location ID within the sequence of locations. Usually we expect this is a natural
            number
        good: True if this is an anomaly-free patch

    Returns:
        class ID for the patch, which unifies all pieces of information
    """
    # object_id = constants.get_class_id(object_class)
    # if patch_source == constants.PatchSource.TEMPLATE:
    #     return object_id + start_loc_id + loc_id_no * 3
    # elif patch_source == constants.PatchSource.GOOD:
    #     return object_id + start_loc_id + loc_id_no * 3 + 1
    # elif patch_source == constants.PatchSource.TEMPLATE:
    #     return object_id + start_loc_id + loc_id_no * 3 + 2
    # else:
    #     raise ValueError(f'Unknown patch source {patch_source}')
    object_id = constants.get_class_id(object_class)
    if patch_source == constants.PatchSource.TEMPLATE or patch_source == constants.PatchSource.GOOD:
        return object_id + start_loc_id + loc_id_no * 2
    elif patch_source == constants.PatchSource.ANOMALOUS:
        return object_id + start_loc_id + loc_id_no * 2 + 1
    else:
        raise ValueError(f'Unknown patch source {patch_source}')


class DatasetPatchTriplet(Dataset):
    """An v0.1 dataset that loads patch triplets consisting of a template, positive and negative patches."""
    def __init__(self, file_list, num_points, centering=constants.TripletCentereing.SEPARATE,
                 transformsPostPP=None, transformsPrePP=None):
        self.file_list = file_list
        self.centering = centering
        self.num_points = num_points
        self.transformsPostPP = transformsPostPP
        self.transformsPrePP = transformsPrePP

    def __len__(self):
        return len(self.file_list)

    def raw_triplet_from_npz(self, file_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Extract the triplet of patches from the data dictionary, loaded from .npz file"""
        data = np.load(self.file_list[file_idx], allow_pickle=True)
        # These PCs have shape [num_pts, 3]
        anchor = torch.from_numpy(data['anchor']).float()
        positive = torch.from_numpy(data['good']).float()
        negative = torch.from_numpy(data['anomalous']).float()
        return anchor, positive, negative

    def __getitem__(self, idx):
        # Load the triplet
        #
        anchor, positive, negative = self.raw_triplet_from_npz(idx)

        if anchor is None or positive is None or negative is None:
            return None
        if anchor.shape[0] == 0 or positive.shape[0] == 0 or negative.shape[0] == 0:
            return None

        if self.transformsPrePP is not None:
            triplet = self.transformsPrePP([anchor, positive, negative])
            anchor, positive, negative = triplet[0], triplet[1], triplet[2]

        anchor = pc_utils.normalize_pc_tensor(anchor, True)
        positive = pc_utils.normalize_pc_tensor(positive, True)
        negative = pc_utils.normalize_pc_tensor(negative, True)

        # Sample fixed number of points
        anchor = pc_utils.resample_pc_tensor(anchor, self.num_points)
        positive = pc_utils.resample_pc_tensor(positive, self.num_points)
        negative = pc_utils.resample_pc_tensor(negative, self.num_points)

        # anchor, positive, negative = pc_utils.center_triplet_tensor(self.centering, anchor, positive, negative)

        if self.transformsPostPP is not None:
            triplet = torch.stack([anchor, positive, negative], dim=0)
            triplet = self.transformsPostPP(triplet)
            anchor, positive, negative = triplet[0], triplet[1], triplet[2]

        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'filename': str(self.file_list[idx])
        }


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None  # None when all filtered


class SupConDatasetPatchTriplet(DatasetPatchTriplet):
    """Returns patch triplets from the SupCon-formatted dataset of patches.

    TODO(alexta): this is a copycat of the  largely for validation. Should not be a permanent code.
    """
    def __init__(self, file_list, num_points, centering=constants.TripletCentereing.SEPARATE,
                 transformsPostPP=None, transformsPrePP=None):

        pattern = re.compile(r"_\d{5}\.npz$")  # matches exactly 5 digits before ".npz"
        filtered_files = [f for f in file_list if pattern.search(f)]

        super(SupConDatasetPatchTriplet, self).__init__(filtered_files*4, num_points, centering, transformsPostPP, transformsPrePP)

    def raw_triplet_from_npz(self, file_idx: int) -> tuple[None, None, None] | tuple[Tensor, Tensor, Tensor]:
        """Extract the triplet of patches from the data dictionary, loaded from .npz file"""
        # These PCs have shape [num_pts, 3]
        data = load_npz_as_native_dict(self.file_list[file_idx])

        # template_patches = pc_utils.unstack_pcs(data['patches_template_idx'], data['patches_template'])
        assert len(data['patches_template']) == 0
        good_patches = pc_utils.unstack_pcs(data['patches_good_idx'], data['patches_good'])
        bad_patches = pc_utils.unstack_pcs(data['patches_bad_idx'], data['patches_bad'])
        assert len(bad_patches) <= 1
        if len(bad_patches) < 1:
            logging.warning(f"No bad patches in the file {data['filename']}")
            return None, None, None
        if len(good_patches) < 2:
            # logging.warning(f"Num. good patches in the file {self.file_list[file_idx]} is {len(good_patches)}")
            return None, None, None
        # Pick anchor and positive randomly
        negative = torch.from_numpy(bad_patches[0]).float()
        good_indices = torch.randperm(len(good_patches))
        anchor = torch.from_numpy(good_patches[good_indices[0]]).float()
        positive = torch.from_numpy(good_patches[good_indices[1]]).float()
        assert anchor is not None and anchor.ndim==2 and anchor.shape[1] == 3
        assert positive is not None and positive.ndim==2 and positive.shape[1] == 3
        assert negative is not None and negative.ndim==2 and negative.shape[1] == 3
        return anchor, positive, negative

#
# class MultiSimilarityDataset(Dataset):
#     def __init__(self, file_list, num_points, centering=constants.TripletCentereing.SEPARATE,
#                  transformsPostPP=None, transformsPrePP=None,
#                  max_good_patches_per_class:int = 3, num_positive_files_for_single_negative=5):
#         self.file_list = file_list
#
#         # Select "positive-to-negative" and "template-to-positive" files list separately
#         pos_to_neg_re = re.compile(r'.*_\d{5}\.npz$')
#         self.pos_to_neg_files  = [f for f in file_list if pos_to_neg_re.match(f)]
#         self.template_to_pos_files = [f for f in file_list if not pos_to_neg_re.match(f)]
#
#         self.centering = centering
#         self.num_points = num_points
#         self.transformsPostPP = transformsPostPP
#         self.transformsPrePP = transformsPrePP
#         self.max_good_patches_per_class = max_good_patches_per_class
#         self.num_positive_files_for_single_negative = num_positive_files_for_single_negative
#
#     def normalize_and_resample_pc_tuple(self, pc_group: List[torch.Tensor]) -> List[torch.Tensor]:
#         """
#         Normalize all the tensors in a tuple (i.e. make them zero-mean and, optionally, unit stddev).
#         Then re-sample them to a given nuymber of points.
#         """
#         for i in range(len(pc_group)):
#             pc_group[i] = pc_utils.normalize_pc_tensor(pc_group[i], True)
#             pc_group[i] = pc_utils.resample_pc_tensor(pc_group[i], self.num_points)
#         return pc_group
#
#     def from_numpy_to_tensors(self, pc_group: List[np.ndarray]) -> List[torch.Tensor]:
#         """Convert a tuple of point clouds from Numpy to Tensors"""
#         tensors = []
#         for np_pc in pc_group:
#             tensors.append(torch.from_numpy(np_pc).float())
#         return tensors
#
#     def _get_simple_item(self, obj_class:str, data: Dict[str, Any]) -> Dict[str, Any] | None:
#         """Load a single  tuple of patches, collected from a same location."""
#         location_id = data['location_id']
#         # Unwrap the patches from a single stacked tensor to the corresponding
#         # These patches have shape [num_pts, 3]; num_pts varies from patch to patch
#         template_patches = pc_utils.unstack_pcs(data['patches_template_idx'], data['patches_template'])
#         good_patches = pc_utils.unstack_pcs(data['patches_good_idx'], data['patches_good'])
#         if len(good_patches) > self.max_good_patches_per_class:
#             good_patches = random.sample(good_patches, self.max_good_patches_per_class)
#         bad_patches = pc_utils.unstack_pcs(data['patches_bad_idx'], data['patches_bad'])
#
#         num_template = len(template_patches)
#         assert num_template  <= 1
#         num_good = len(good_patches)
#         if num_good < 1:
#             # Happens for Airplane, class 10000. Most likely, issues with registration.
#             # logging.warning(f"No good patches in the entry: {data}")
#             return None
#             # assert num_good >= 1
#
#         num_bad = len(bad_patches)
#         #TODO: currently we can at most 1 bad
#         assert num_bad <= 1
#
#         if len(template_patches) == 0 and len(good_patches) == 1:
#             # This file contains a single "good" patch. Too little info for contrastive learning.
#             return None
#
#         template_patches = self.from_numpy_to_tensors(template_patches)
#         good_patches = self.from_numpy_to_tensors(good_patches)
#         bad_patches = self.from_numpy_to_tensors(bad_patches)
#         assert len(template_patches) == num_template and len(good_patches) == num_good and len(bad_patches) == num_bad
#         assert len(template_patches) + len(good_patches) >= 2
#
#         if self.transformsPrePP is not None:
#             # Do a pre-normalization transforms. Such as "cutting" holes in patches
#             # TODO(alexta): some "good" patches may be really, really ugly -  a small chunk of an original patch.
#             # Consider NOT doing any transforms on these.
#             pcs = template_patches + good_patches
#             pcs_transformed = self.transformsPrePP(pcs)
#             template_patches = pcs_transformed[:num_template]
#             good_patches = pcs_transformed[num_template:]
#         assert len(template_patches) == num_template and len(good_patches) == num_good and len(bad_patches) == num_bad
#
#         template_patches_norm = self.normalize_and_resample_pc_tuple(template_patches)
#         good_patches_norm = self.normalize_and_resample_pc_tuple(good_patches)
#         bad_patches_norm = self.normalize_and_resample_pc_tuple(bad_patches)
#         assert len(template_patches_norm) + len(good_patches_norm) == num_template + num_good
#         assert len(bad_patches_norm) == num_bad
#
#         num_positive_patches_norm = len(template_patches_norm) + len(good_patches_norm)
#         num_negative_patches_norm = len(bad_patches_norm)
#
#         batch_all_patches = torch.stack(template_patches_norm+good_patches_norm + bad_patches_norm, dim=0)
#         if self.transformsPostPP is not None:
#             # Post-normalization transforms - random rotations, ... .
#             batch_all_patches = self.transformsPostPP(batch_all_patches)
#             # template_patches = [t for t in transformed_all_patches[:num_template]]
#             # good_patches = [t for t in transformed_all_patches[num_template:num_template + num_good]]
#             # bad_patches = [t for t in transformed_all_patches[num_template + num_good:]]
#             # assert len(template_patches) == num_template and len(good_patches) == num_good and len(bad_patches) == num_bad
#         assert batch_all_patches.shape[0] == num_positive_patches_norm+num_negative_patches_norm
#
#         # if len(template_patches) == 0:
#         #     # For data format consistency, ensure we have at least one "template" patch
#         #     template_patches[0] = good_patches.pop(0)
#         # assert len(template_patches) == 1
#         # assert len(good_patches) == num_good or len(good_patches) == num_good-1
#         # assert len(bad_patches) == num_bad
#
#         # template_class_id = get_patch_class_id(class_id, 0, location_id, constants.PatchSource.TEMPLATE)
#         good_class_id = get_patch_class_id(obj_class, 0, location_id, constants.PatchSource.GOOD)
#         bad_class_id = get_patch_class_id(obj_class, 0, location_id, constants.PatchSource.ANOMALOUS)
#
#         # patches = template_patches + good_patches + bad_patches
#         labels = [good_class_id]*num_positive_patches_norm + [bad_class_id]*num_negative_patches_norm
#         return {
#             'patches': batch_all_patches,
#             'labels': torch.tensor(labels, dtype=torch.long),
#             'num_positive_patches': num_positive_patches_norm,
#             'num_negative_patches': num_negative_patches_norm,
#             'positive_class_id': good_class_id,
#             'negative_class_id': bad_class_id,
#         }
#
#     def __len__(self):
#         return len(self.pos_to_neg_files)
#         # return len(self.file_list)
#
#     def __getitem__(self, idx):
#         """Load a set of patch tuples.
#         One tuple must contain the anomalous patch. Remaining num_positive_files_for_single_negative tuples
#         will contain only "positive" patches.
#         This is necessary because we may have a huge disbalance between anomalous and non-amomalous patches,
#         to the point where there may be no single anomalous patch per minibatch.
#         """
#         #
#         # # Load data from the input .npz file that contains a contrastive set - a negative patch vs. few positives
#         # # This is the original approach. Fails on shapenet - too few anomalies per non-anomaly patch
#         # path = Path(self.file_list[idx])
#         # data = load_npz_as_native_dict(path)
#         # obj_class = path.name.split("_")[1]
#         # data_item = self._get_simple_item(obj_class, data)
#         # return data_item
#
#         # Maintain a specified ratio of anomalous vs anomaly-free patches. However, this returns not a patch - but
#         # a list of patches. As a result, the collate receives not a list of patches, but a list of lists.
#         try:
#             path = Path(self.pos_to_neg_files[idx])
#             obj_class = path.name.split("_")[1]
#             data_list = []
#             data_pos_to_neg = load_npz_as_native_dict(self.pos_to_neg_files[idx])
#             pos_to_neg = self._get_simple_item(obj_class, data_pos_to_neg)
#             if pos_to_neg is not None:
#                 assert pos_to_neg['num_negative_patches'] != 0
#                 data_list.append(pos_to_neg)
#
#             # Now load files that contains no anomalies - a template patch vs. few positives
#             fnames_template_to_pos = random.sample(self.template_to_pos_files, self.num_positive_files_for_single_negative)
#             for fname in fnames_template_to_pos:
#                 data_template_to_pos = load_npz_as_native_dict(fname)
#                 template_to_pos = self._get_simple_item(obj_class, data_template_to_pos)
#                 if template_to_pos is not None:
#                     assert template_to_pos['num_negative_patches'] == 0
#                     data_list.append(template_to_pos)
#             return data_list
#         except BadZipFile:
#             logger.error("Bad zip file, skipping")
#             return None


class MultiSimilarityBaseDataset(Dataset):
    """A base class for all the v.0.3 datasets."""
    def __init__(self, file_list, num_points, centering=constants.TripletCentereing.SEPARATE,
                 transformsPostPP=None, transformsPrePP=None, max_good_patches_per_class=2):
        """
        Args:
            file_list: the list of files to read patches from.
            num_points: the desired number of points in the patch.
            centering: a method to normalize and center the patches.
        """
        re_file_list = re.compile(self.get_file_filter_())
        self.file_list = [f for f in file_list if re_file_list.match(Path(f).name)]
        self.centering = centering
        self.num_points = num_points
        self.transformsPostPP = transformsPostPP
        self.transformsPrePP = transformsPrePP
        self.max_good_patches_per_class = max_good_patches_per_class

    def get_file_filter_(self):
        raise NotImplementedError()

    def sample_positive_patches_(self, patches: List[np.ndarray]):
        raise NotImplementedError()

    @staticmethod
    def normalize_and_resample_pc_tuple(pc_group: List[torch.Tensor], num_points) -> List[torch.Tensor]:
        """
        Normalize all the tensors in a tuple (i.e. make them zero-mean and, optionally, unit stddev).
        Then re-sample them to a given nuymber of points.
        """
        for i in range(len(pc_group)):
            pc_group[i] = pc_utils.normalize_pc_tensor(pc_group[i], True)
            pc_group[i] = pc_utils.resample_pc_tensor(pc_group[i], num_points)
        return pc_group

    @staticmethod
    def from_numpy_to_tensors(pc_group: List[np.ndarray]) -> List[torch.Tensor]:
        """Convert a tuple of point clouds from Numpy to Tensors"""
        tensors = []
        for np_pc in pc_group:
            tensors.append(torch.from_numpy(np_pc).float())
        return tensors


    def get_single_item_(self, obj_class:str, data: Dict[str, Any]) -> Dict[str, Any] | None:
        """Load a single  tuple of patches, collected from a same location."""
        location_id = data['location_id']
        # Unwrap the patches from a single stacked tensor to the corresponding
        # These patches have shape [num_pts, 3]; num_pts varies from patch to patch
        template_patches = pc_utils.unstack_pcs(data['patches_template_idx'], data['patches_template'])
        good_patches = pc_utils.unstack_pcs(data['patches_good_idx'], data['patches_good'])
        good_patches = self.random_sample_positive_patches_(good_patches)
        bad_patches = pc_utils.unstack_pcs(data['patches_bad_idx'], data['patches_bad'])

        num_template = len(template_patches)
        assert num_template  <= 1
        num_good = len(good_patches)
        if num_good < 1:
            # Happens for Airplane, class 10000. Most likely, issues with registration.
            # logging.warning(f"No good patches in the entry: {data}")
            return None

        num_bad = len(bad_patches)
        #TODO: currently we can at most 1 bad
        assert num_bad <= 1

        if len(template_patches) == 0 and len(good_patches) == 1:
            # This file contains a single "good" patch. Too little info for contrastive learning.
            return None

        template_patches = self.from_numpy_to_tensors(template_patches)
        good_patches = self.from_numpy_to_tensors(good_patches)
        bad_patches = self.from_numpy_to_tensors(bad_patches)
        assert len(template_patches) == num_template and len(good_patches) == num_good and len(bad_patches) == num_bad
        assert len(template_patches) + len(good_patches) >= 2

        if self.transformsPrePP is not None:
            # Do a pre-normalization transforms. Such as "cutting" holes in patches
            # TODO(alexta): some "good" patches may be really, really ugly -  a small chunk of an original patch.
            # Consider NOT doing any transforms on these.
            pcs = template_patches + good_patches
            pcs_transformed = self.transformsPrePP(pcs)
            template_patches = pcs_transformed[:num_template]
            good_patches = pcs_transformed[num_template:]
        assert len(template_patches) == num_template and len(good_patches) == num_good and len(bad_patches) == num_bad

        template_patches_norm = self.normalize_and_resample_pc_tuple(template_patches, self.num_points)
        good_patches_norm = self.normalize_and_resample_pc_tuple(good_patches, self.num_points)
        bad_patches_norm = self.normalize_and_resample_pc_tuple(bad_patches, self.num_points)
        assert len(template_patches_norm) + len(good_patches_norm) == num_template + num_good
        assert len(bad_patches_norm) == num_bad

        num_positive_patches_norm = len(template_patches_norm) + len(good_patches_norm)
        num_negative_patches_norm = len(bad_patches_norm)

        batch_all_patches = torch.stack(template_patches_norm+good_patches_norm + bad_patches_norm, dim=0)
        if self.transformsPostPP is not None:
            # Post-normalization transforms - random rotations, ... .
            batch_all_patches = self.transformsPostPP(batch_all_patches)
        assert batch_all_patches.shape[0] == num_positive_patches_norm+num_negative_patches_norm

        # template_class_id = get_patch_class_id(class_id, 0, location_id, constants.PatchSource.TEMPLATE)
        good_class_id = get_patch_class_id(obj_class, 0, location_id, constants.PatchSource.GOOD)
        bad_class_id = get_patch_class_id(obj_class, 0, location_id, constants.PatchSource.ANOMALOUS)

        # patches = template_patches + good_patches + bad_patches
        labels = [good_class_id]*num_positive_patches_norm + [bad_class_id]*num_negative_patches_norm
        return {
            'patches': batch_all_patches,
            'labels': torch.tensor(labels, dtype=torch.long),
            'num_positive_patches': num_positive_patches_norm,
            'num_negative_patches': num_negative_patches_norm,
            'positive_class_id': good_class_id,
            'negative_class_id': bad_class_id,
        }

    def random_sample_positive_patches_(self, positive_patches: List[np.ndarray]):
        if len(positive_patches) > self.max_good_patches_per_class:
            positive_patches = random.sample(positive_patches, self.max_good_patches_per_class)
        return positive_patches

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            path = Path(self.file_list[idx])
            obj_class = path.name.split("_")[1]
            data_raw = load_npz_as_native_dict(self.file_list[idx])
            data = self.get_single_item_(obj_class, data_raw)
            if data is not None:
                data['filename'] = path
            return data
        except BadZipFile:
            logger.error("Bad zip file, skipping")
            return None

class MultiSimilarityAllPositiveDataset(MultiSimilarityBaseDataset):
    """Loads tuples that don't contain the negative, anomalous patches. Form a basis for WA tuples."""
    def __init__(self, file_list, num_points, centering=constants.TripletCentereing.SEPARATE,
                 transformsPostPP=None, transformsPrePP=None, max_good_patches_per_class:int = 3):
        super(MultiSimilarityAllPositiveDataset, self).__init__(
            file_list, num_points, centering, transformsPostPP, transformsPrePP,
            max_good_patches_per_class=max_good_patches_per_class)
        self.max_good_patches_per_class = max_good_patches_per_class

    def __getitem__(self, idx):
        ret = super(MultiSimilarityAllPositiveDataset, self).__getitem__(idx)
        if ret is not None:
            assert ret['num_negative_patches'] == 0
        return ret

    def get_file_filter_(self):
        return ALL_POS_FILE_FILTER

class MultiSimilarityNegTupleDataset(MultiSimilarityBaseDataset):
    """Loads tuples that contain the negative, anomalous patches. These are, essentially FA tuples."""
    def __init__(self, file_list, num_points, centering=constants.TripletCentereing.SEPARATE,
                 transformsPostPP=None, transformsPrePP=None, max_good_patches_per_class:int = 2):
        super(MultiSimilarityNegTupleDataset, self).__init__(
            file_list, num_points, centering, transformsPostPP, transformsPrePP)
        self.max_good_patches_per_class = max_good_patches_per_class

    def __getitem__(self, idx):
        ret = super(MultiSimilarityNegTupleDataset, self).__getitem__(idx)
        if ret is not None:
            assert ret['num_negative_patches'] == 1
        return ret

    def get_file_filter_(self):
        return POS_NEG_FILE_FILTER


class TupleCollator:
    def collate_positives_(self, all_patches, positive_start_index, positives, dtype_idx):
        """Collates a subset of positive patches with a set of all the patches. Updates indices

        Args:
            all_patches: the set of all the patches in the minibatch
            positive_start_index: the global index of the next positive patch that will be added to all_patches
            positives: the subset of all the positive patches in a tuple to be added to the set of all patches in a
                minibatch
            dtype_idx: the datatype of the index
        Returns:
            all_patches: an updated set of all patches in a minibatch
            tuple_positive_pairs: the set of indices for the positive patches that were added, within a global minibatch
            next_start_index: the global index of the next patch that will be added to all_patches minibatch
         """
        # Append actual patches
        all_patches.append(positives)

        # The number of positive patches in this "batch"
        P = positives.shape[0]
        # Create index of mutual matches
        tuple_all_positive_idx = torch.arange(positive_start_index, positive_start_index + P, dtype=dtype_idx)
        # Build a cartesian product of all "good" indices, including the anchor, for this tuple.
        # In this way, every positive is an anchor.
        mutual_positive_pairs_idx = torch.cartesian_prod(tuple_all_positive_idx, tuple_all_positive_idx)
        # Skip pairs with itself
        valid_pairs_mask = mutual_positive_pairs_idx[:, 0] != mutual_positive_pairs_idx[:, 1]
        # List of all valid "good-to-good" pairs in this tuple - skip self-matches
        tuple_positive_pairs = mutual_positive_pairs_idx[valid_pairs_mask]
        # # LLMs say this is more efficient that Cartesian product.
        # first get a set of combinations, and then make them symmetric
        # pos = torch.combinations(anchor_and_positive_idx, r=2)  # [M, 2]
        # mutual_positive_pairs_idx = torch.cat([pos, pos[:, [1, 0]]], dim=0)

        # Get the start index of the next tuple. It could be a next positive tuple, or the negative in this tuple
        next_start_index = positive_start_index + P
        return all_patches, tuple_positive_pairs, next_start_index


class AllPositiveTupleCollator(TupleCollator):
    """Collate tuples that consist of only  positive patches into a minibatch with a format accepted by PML library.

    Args:
        batch: a batch returned by the MultiSimilarityDataset containing N entries.
         Each i-th tuple contains a number M_i of "patches" entries. These should be tensors of a shape [N_points, 3].
         Tuple will also contain "num_positive_patches" = M_i for each entry.
    Returns:
        a dictionary with the fields: 'patches', 'valid indices'
        patches: patches from all tuples, flattened into a tensor (M_1+M_2+...+M_N) x N_points x 3
            For example, if the input batch contains patches (marked as P):
            [[P11, P12, P13], [P21, P22],[P31, P32]]
            The 'patches' tensor should contain them flattened into [P11, P12, P13, P21, P22, P31, P32].
        'valid indices' is a tuple of 4 tensors. first two are a Cartesian product of patch indices. The last two empty.
            The first tuple should contain the first elements of a cartesian product,
            the second - a corresponding 2nd element. In our example:
            [0, 0, 1, 1, 2, 2, 3, 4, 5, 6], [1,2,0,2, 0,1,4,3,6,5],[],[].
    """

    def __call__(self, batch):
        tuple_start_index = 0
        dtype_idx = torch.long

        all_positive_pairs = []
        all_patches = []

        for tuple in batch:
            if tuple is None:
                continue
            assert type(tuple) == dict
            B = tuple["patches"].shape[0]  # total items in this tuple
            assert int(tuple["num_negative_patches"]) == 0
            # The total number of "positive" - which includes anchor (template) and "good" ones - patches
            num_positive = int(tuple["num_positive_patches"])  # includes the template
            # assert that all the patches in this tuple are "positives"
            assert num_positive > 1
            assert num_positive == B
            # Simply add positives to the batch
            all_patches, positive_pairs, tuple_start_index = self.collate_positives_(
                all_patches, tuple_start_index, tuple["patches"], dtype_idx)
            all_positive_pairs.append(positive_pairs)

        if not all_patches:
            raise RuntimeError("AllPositiveTupleCollator: empty batch after filtering None/invalid entries")
        all_patches_tensor = torch.cat(all_patches, dim=0)
        # Now build all (a, p) pairs across the batch.
        all_positive_pairs_tensor = torch.cat(all_positive_pairs, dim=0)
        empty_idx = torch.empty(0, dtype=dtype_idx)
        all_positive_indices_tuple = (all_positive_pairs_tensor[:,0], all_positive_pairs_tensor[:,1],
                                      empty_idx, empty_idx)

        return {'patches':all_patches_tensor, 'valid_indices': all_positive_indices_tuple}

class SingleNegativeTupleCollator(TupleCollator):
    # def __init__(self, device):
    #     super(SingleNegativeTupleCollator, self).__init__(device)

    def __call__(self, batch):
        tuple_start_index = 0
        dtype_idx = torch.long

        idx_ap, idx_p = [], []
        idx_an, idx_n = [], []
        all_patches = []

        for tuple in batch:
            if tuple is None:
                continue
            tuple_patches = tuple["patches"]
            # The total number of "positive" - which includes anchor (template) and "good" ones - patches
            num_positive = int(tuple["num_positive_patches"])  # includes the template
            # assert that all the patches in this tuple are "positives"
            assert num_positive >= 1
            # The total number of "negatives". Should be 1 for now
            # TODO (alexta): in the future we may get more negatives
            num_negative = int(tuple["num_negative_patches"])
            assert num_negative == 1
            assert num_positive + num_negative == tuple_patches.shape[0]    # total items in this tuple
            if num_positive == 1:
                # We can't do any contrastive learning with a single positive instance
                continue
            # add positives to the batch
            all_patches, positive_pairs, negative_start_index = self.collate_positives_(
                all_patches, tuple_start_index, tuple_patches[:num_positive], dtype_idx)
            # Form the index of "anchor-to-negative" pairs
            idx_ap.append(positive_pairs[:,0])
            idx_p.append(positive_pairs[:,1])
            # Process a negative patch. Append it to all the patches. Again, we asssume there's a single neg. patch
            # Make sure it conforms in shape to the rest of the patches, i.e. of a shape [1, N, 3]
            assert tuple_start_index + num_positive == negative_start_index
            negative_patch = tuple_patches[num_positive].unsqueeze(0)
            all_patches.append(negative_patch)

            # idx_an.append(torch.arange(tuple_start_index, negative_start_index))
            # all_neg_indices = torch.full([num_positive], negative_start_index, dtype=dtype_idx)

            # Form negative pairs. These include all the negative pairs in a triplet style,
            # i.e. a separate anchor-negative pair for each anchor-positive pair
            idx_an.append(positive_pairs[:,0])
            all_neg_indices = torch.full([positive_pairs[:,0].shape[0]], negative_start_index, dtype=dtype_idx)
            # Form the index of "anchor-to-negative" pairs
            idx_n.append(all_neg_indices)
            tuple_start_index += tuple_patches.shape[0]

        if not all_patches:
            raise RuntimeError("SingleNegativeTupleCollator: empty batch after filtering None/single-positive tuples")
        all_patches_tensor = torch.cat(all_patches, dim=0)
        all_indices_tuple = (torch.cat(idx_ap, dim=0), torch.cat(idx_p, dim=0),
                             torch.cat(idx_an, dim=0), torch.cat(idx_n, dim=0))
        return {'patches': all_patches_tensor, 'valid_indices': all_indices_tuple}


def collate_ms_batches(mined_patches, mined_indices, fa_patches, fa_indices):
    """This stacks the two batches of data - one from all_positive another from negative loader.

    Patches tensors are merged. Pair indices are adjusted to reflect the relative change in indices.
    """
    triplet_offset = mined_patches.shape[0]
    patches = torch.cat((mined_patches, fa_patches), dim=0)
    for t in fa_indices:
        t += triplet_offset
    return patches, mined_indices, fa_indices

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

def load_all_negatives(dataset_dir:str, num_points:int, train_fraction: float) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load all the negative (anomalous) patches from the whole dataset.
    Intended to be used with a more intelligent miner (not implemented currently)"""

    # Try to load from cache.
    # Give a cache a different extension so it won't interfere with dataset loading.
    cache_path = Path(dataset_dir) / "all_negative_patches_cache.npc"
    if cache_path.exists():
        logger.info(f"Loading cached negative patches from {cache_path}")
        try:
            with np.load(cache_path) as data:
                neg_patches_np = data["neg_patches"]
            neg_patches_tensor = torch.from_numpy(neg_patches_np)
        except Exception as e:
            logger.warning(
                f"Failed to load negative patches cache at {cache_path}, "
                f"recomputing from raw files. Error: {e}"
            )
            return None,None
    else:
        file_list = Path(dataset_dir).glob("*.npz")
        re_file_list = re.compile(POS_NEG_FILE_FILTER)
        file_list_filtered = [f for f in file_list if re_file_list.match(f.name)]
        logger.info(f'Loading all the negative patches:')
        pbar = tqdm(enumerate(file_list_filtered), total=len(file_list_filtered))
        all_neg_patches = []
        for idx, fname in pbar:
            data_raw = load_npz_as_native_dict(fname)
            bad_patches = pc_utils.unstack_pcs(data_raw['patches_bad_idx'], data_raw['patches_bad'])
            if len(bad_patches) != 1:
                logger.warning(f'Multiple negative patches in the file {fname}')
            # do regular pre-processing of the patches. But don't do any augmentations.
            bad_patches = MultiSimilarityBaseDataset.from_numpy_to_tensors(bad_patches)
            bad_patches_norm = MultiSimilarityBaseDataset.normalize_and_resample_pc_tuple(bad_patches, num_points)
            all_neg_patches.extend(bad_patches_norm)
        neg_patches_tensor = torch.stack(all_neg_patches)

    # Randomly shuffle a tensor
    N = neg_patches_tensor.shape[0]
    shuffled_indices = torch.randperm(N)
    neg_patches_tensor = neg_patches_tensor[shuffled_indices]

    # Save cache for future runs
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            neg_patches=neg_patches_tensor.cpu().numpy(),
        )
        logger.warning(f"Saved negative patches cache to {cache_path}")
    except Exception as e:
        logger.error(f"Failed to save negative patches cache at {cache_path}: {e}")

    # split into train/validation and return
    n_train = int(N * train_fraction)
    return neg_patches_tensor[:n_train], neg_patches_tensor[n_train:]

#
# def ms_collate_fn(batch_unflattened):
#     """"""
#     # batch = [chain.from_iterable(subbatch) for subbatch in batch if subbatch is not None]
#     #
#     # # We have a "list of lists" in our batch. We need to flatten these.
#     # # Comprehension is unreadable and can be prone to errors
#     # # batch = [item for sublist in batch_unflattened for item in sublist]
#     # Collate patch tuples manually
#     batch = []
#     for group in batch_unflattened:
#         if group is not None:
#             batch.extend(group)
#     #
#     # # An old way - when we receive a list of patches.
#     # batch = [b for b in batch_unflattened if b is not None]
#
#     if not batch:
#         return None  # or a well-defined “empty” batch you handle upstream
#
#     all_patches = []
#     all_labels = []
#
#     idx_ap = []
#     idx_p = []
#     idx_an = []
#     idx_n = []
#
#     tuple_start_index = 0
#     device = batch[0]["patches"].device
#     dtype_idx = torch.long
#
#     # Indices of all the positive and negative items in the minibatch.
#     # We will use them to build all "positive-to-negative" pairs.
#     all_good_idx = []
#     all_negative_idx = []
#     # These are valid pairs of "positive-to-positive" patches that share the same location. Tensors of shape [N, 2]
#     all_positive_pairs = []
#
#     for subbatch in batch:
#         assert type(subbatch) == dict
#         assert subbatch['patches'].shape[0] == subbatch['labels'].shape[0]
#         B = subbatch["labels"].shape[0]  # total items in this tuple
#         all_patches.append(subbatch['patches'])
#         all_labels.append(subbatch['labels'])
#
#         # The total number of "positive" - which includes anchor (template) and "good" ones - patches
#         num_pos = int(subbatch["num_positive_patches"])  # includes the template
#         num_neg = int(subbatch["num_negative_patches"])
#         num_goods = num_pos - 1
#         assert num_goods >= 1
#
#         # For this tuple, these are the indices of anchor (i.e. a first good), positives and negatives (we expect one)
#         anchor_index = tuple_start_index
#         # goods_start = anchor_index + 1
#         negs_start = anchor_index + num_pos
#
#         # This is an index of anomaly-free elements in the current tuple
#         anchor_and_positive_idx = torch.arange(anchor_index, anchor_index + num_pos, dtype=dtype_idx, device=device)
#         all_good_idx.append(anchor_and_positive_idx)
#
#         # Build a cartesian product of all "good" indices, including the anchor, for this tuple.
#         # In this way, every positive is an anchor.
#         mutual_positive_pairs_idx = torch.cartesian_prod(anchor_and_positive_idx, anchor_and_positive_idx)
#         # Skip pairs with itself
#         valid_pairs_mask = mutual_positive_pairs_idx[:,0] != mutual_positive_pairs_idx[:,1]
#         # List of all valid "good-to-good" pairs in this tuple - skip self-matches
#         mutual_positive_pairs_idx = mutual_positive_pairs_idx[valid_pairs_mask]
#         # # LLMs say this is more efficient that Cartesian product.
#         # first get a set of combinations, and then make them symmetric
#         # pos = torch.combinations(anchor_and_positive_idx, r=2)  # [M, 2]
#         # mutual_positive_pairs_idx = torch.cat([pos, pos[:, [1, 0]]], dim=0)
#         all_positive_pairs.append(mutual_positive_pairs_idx)
#
#         # 0th entry of patches is a template - i.e. an anchor. In our full batch, its index should be anchor_index
#         # "positive" entries - skip template.  Assign rest of the patches to "positive"
#         # positive_idx = torch.arange(goods_start, goods_start + num_goods, dtype=dtype_idx, device=device)
#
#         if num_neg > 0:
#             # In the input set, we should always have template as a 0th element of the subbatch, negative - the last one
#
#             # we are creating a set of indices for (anchor, positive, negative) triplets.
#             # Each tuple contain indices of valid pairs / triplets (template_idx, good_idx, template_idx, bad_idx).
#
#             # Last entries are negative; assign their indices.
#             # Note: as of 11/2025 we expect a single negative
#             neg_indices = torch.arange(negs_start, negs_start + num_neg, dtype=dtype_idx, device=device)
#             # TODO (alexta): this is temporary
#             assert neg_indices.shape[0] == 1
#             all_negative_idx.append(neg_indices)
#
#             # The total number of indices should be same as the number of labels
#             # assert 1+ positive_idx.shape[0]  + neg_indices.shape[0] == subbatch['labels'].shape[0]
#             # The last index should be the same as the total size of labels in the "subbatch"
#             assert neg_indices[-1] - anchor_index == subbatch['labels'].shape[0] -1
#
#             # This is a quick and dirty way, that assumes we don't have too many bad patches per good patch (which is true)
#             # Old way: match everything only to the anchor
#             # good_bad_pairs_triplet = torch.cartesian_prod(positive_idx, neg_indices)
#             # anchor_indices_triplet = torch.full((good_bad_pairs_triplet.shape[0],), anchor_index, dtype=dtype_idx, device=device)
#             # idx_a.append(anchor_indices_triplet)
#             # idx_p.append(good_bad_pairs_triplet[:,0])
#             # idx_n.append(good_bad_pairs_triplet[:,1])
#
#             # Commented out on 11/14 due to accuracy regression.
#             # # These are all the combinations of "good" (including anchor and positive) and "bad" patches
#             # # Note: we create separate sets of pairs for mutual positive and positive to negatives
#             # idx_ap.append(mutual_positive_pairs_idx[:, 0])
#             # idx_p.append(mutual_positive_pairs_idx[:, 1])
#             # good_bad_pairs_triplet = torch.cartesian_prod(anchor_and_positive_idx, neg_indices)
#             # idx_an.append(good_bad_pairs_triplet[:, 0])
#             # idx_n.append(good_bad_pairs_triplet[:, 1])
#
#             # 11/14 - revert to old functionality. I.e. pass triplets explicitly for negative pairs
#             # Note: this explicitly assumes we have a single negative
#             tensor_neg = torch.full((mutual_positive_pairs_idx[:, 0].shape[0],), neg_indices[0].item(),
#                                     dtype=dtype_idx, device=device)
#             # "anchors to positives" pairs. Although note that we have a full cartesian product for these
#             idx_ap.append(mutual_positive_pairs_idx[:, 0])
#             idx_p.append(mutual_positive_pairs_idx[:, 1])
#             # "anchors to negatives" pairs. Use the same positive as for AP (i.e. explicitly create triplet)
#             # but assign negatives to each of these
#             idx_an.append(mutual_positive_pairs_idx[:, 0])
#             idx_n.append(tensor_neg)
#
#         tuple_start_index += B
#
#     all_good_idx_tensor = torch.cat(all_good_idx, dim=0)
#     all_negative_idx_tensor = torch.cat(all_negative_idx, dim=0)
#
#     # Now build all (a, n) pairs across the whole batch. Note: here anchors are all positive (anomaly-free) patches.
#     # Note: this will include existing tuples of "good vs anomalous". Not sure if this is a problem.
#     all_positive_negative_pairs_tensor = torch.cartesian_prod(all_good_idx_tensor, all_negative_idx_tensor)
#     # Make sure that there's absolutely no intersection between indices of "good" and "negative" patches
#     assert torch.sum(all_positive_negative_pairs_tensor[:, 0] == all_positive_negative_pairs_tensor[:, 1]) == 0
#     # Now build all (a, p) pairs across the batch.
#     all_positive_pairs_tensor = torch.cat(all_positive_pairs, dim=0)
#     # Create a final tensor.
#     all_valid_indices_tuple = (all_positive_pairs_tensor[:,0], all_positive_pairs_tensor[:,1],
#                                all_positive_negative_pairs_tensor[:,0], all_positive_negative_pairs_tensor[:,1])
#
#     # There's no overlap between indices of "good" and "bad" patches
#
#     all_patches_tensor = torch.cat(all_patches, dim=0)
#     all_labels_tensor = torch.cat(all_labels, dim=0)
#     assert len(idx_ap) == len(idx_p)
#     assert len(idx_an) == len(idx_n)
#
#     if len(idx_ap) == 0:
#         triplet_indices_tuple = None
#     else:
#         triplet_indices_tuple = (torch.cat(idx_ap, dim=0), torch.cat(idx_p, dim=0),
#                                  torch.cat(idx_an, dim=0),  torch.cat(idx_n, dim=0))
#     return {'patches': all_patches_tensor, 'labels': all_labels_tensor,
#             'triplet_indices_tuple': triplet_indices_tuple,
#             'all_valid_indices_tuple': all_valid_indices_tuple
#         }
