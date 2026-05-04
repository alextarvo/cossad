"""Patch memory bank functionality."""
import logging

from utils.pc_utils import INVAL_PC_IDX
import numpy as np
import torch
from scipy.spatial.distance import pdist, cdist, cosine
from external.patchcore.common import FaissNN, NearestNeighbourScorer, RescaleSegmentor

from abc import ABC, abstractmethod
import pingouin as pg
from utils.debug import require

logger = logging.getLogger(__name__)

# For FPFH we may have zero std. deviation within the memory bank.
# This is the minimum allowed value for pair_dist standard deviation. If the value is less than that
# we will regularize the array of mutual pair distances with the root of its value.
MIN_PAIR_DIST_STD = 1e-8

from typing import Callable, Mapping
# This is the set of possible aggregations
_TOPK_AGGREGATORS: Mapping[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean,
    "max": np.max,
    "sum": np.sum,
    "median": np.median,
}

class GlobalAnomalyScoreComputer(object):
    """Computes anomaly score based on the top K highest points. Aggregation op could be one of the mean, max, sum etc."""
    _top_k: int
    _aggregator: Callable[[np.ndarray], float]

    def __init__(self, top_k:int, aggregation_method:str):

        """A static factory method pattern. Separates class construction from the functionality."""
        if top_k <= 0:
            raise ValueError("Number of points selected must be greater than 0")
        self._top_k = top_k
        try:
            self._aggregator = _TOPK_AGGREGATORS[aggregation_method]
        except KeyError:
            raise ValueError(f"Unknown aggregation method: {aggregation_method!r}") from None

    def get_aggregated_score(self, effect_sizes:np.ndarray) -> float:
        require(effect_sizes.ndim == 1, 'Effect sizes must be 1-d array')
        if self._top_k >= effect_sizes.shape[0]:
            logger.warning(f'Top k selection size {self._top_k} is equal or greater than number of points {effect_sizes.shape[0]}')
            top_k_points = effect_sizes
        else:
            # np.partition partitions the array  not by top K highest / rest, but by bottom K lowest / rest points
            top_k_points = np.partition(effect_sizes, effect_sizes.shape[0]-self._top_k)[-self._top_k:]
        return float(self._aggregator(top_k_points))

class Distance(object):
    """Measures pairwise distance between the set of patch representations."""
    def __init__(self, distance_metric:str):
        self.distance_metric = distance_metric

    def get_pairwise_distances(self, X:np.ndarray) -> np.ndarray:
        dists = pdist(X, metric=self.distance_metric)
        return dists.ravel()

    def get_distance(self, X:np.ndarray, v:np.ndarray) -> np.ndarray:
        dists = cdist(X, v[None, :].numpy(), metric=self.distance_metric)
        return dists.ravel()

class PatchMemoryBankInterface(ABC):
    """This is a base interface for all the patch memory banks"""

    @abstractmethod
    def add_patches(self, template_feature_vectors: np.ndarray):
        """Populates the memory banks with feature vectors of template patches

        Args:
            template_feature_vectors: (N,d) tensor of patch feature vectors. N is the num. patches; d - dimension of a feature.
        """
        pass

    @abstractmethod
    def compute_pairwise_distances(self):
        """Computes required distances in the memory bank. Expected to be called prior to detect_anomaly"""
        pass

    @abstractmethod
    def detect_anomaly(self, test_feature_vectors: np.ndarray, patch_center_indices: np.ndarray) ->tuple[float, np.ndarray]:
        """Detect an anomaly.

         Args:
             test_feature_vectors: an (N, d) tensor of feature vector for the patch centers, from the test object
                The position of the feature vector in the list corresponds to the patch position in
                the list of patches, defined for the template shape.
             patch_center_indices: an (N) ndarray with indices of the points that constitute valid patch centers.
                I.e. some patches may be not present in the patches_test array. This is a case for Real3DAD, which
                contains not full 3D shapes, but a 2.5D (i.e. a top-down or bottom-up views).

        Returns:
            - a global object-wide anomaly score
            - an array of point-wise anomaly scores
        """
        pass


class IndexedPatchMemoryBankBase(PatchMemoryBankInterface):
    """A base class for all the registration-based patch (i.e. representation of the patch) comparison.

    Provides a unified API for all the spatially aware MB solutions.
    """

    def __init__(self, distance_metric:str = 'cosine', top_k:int=1, aggregation_method:str='max'):
        # Library of patch representations for a given location; keys are the location IDs.
        # Values are a NxD array where N - number of template patches in the bank and D - representation dimension.
        self.patch_lib = {}
        self.distance = Distance(distance_metric=distance_metric)  # Can be also 'euclidean'
        self.anomaly_scorer = GlobalAnomalyScoreComputer(top_k=top_k, aggregation_method=aggregation_method)

    def add_patches(self, template_feature_vectors):
        for loc_id in range(template_feature_vectors.shape[0]):
            if loc_id not in self.patch_lib.keys():
                self.patch_lib[loc_id] = template_feature_vectors[loc_id]
            else:
                self.patch_lib[loc_id] = np.vstack((self.patch_lib[loc_id], template_feature_vectors[loc_id]))

    def _get_distance_diff(self, loc_id, patch_test):
        raise NotImplementedError

    def detect_anomaly(self, test_feature_vectors, patch_center_indices):
        """Detect an anomaly.

         Args:
             test_feature_vectors: an (N, d) tensor of feature vector for the patch centers, from the test object
                The position of the feature vector in the list corresponds to the patch position in
                the list of patches, defined for the template shape.
             patch_center_indices: an (N) ndarray with indices of the points that constitute valid patch centers.
                I.e. some patches may be not present in the patches_test array. This is a case for Real3DAD, which
                contains not full 3D shapes, but a 2.5D (i.e. a top-down or bottom-up views).
        """
        logging.debug(f'Received patch features: {test_feature_vectors}')

        effect_sizes = np.array([self._get_distance_diff(loc_id, test_feature_vectors[loc_id])
                                 for loc_id in range(test_feature_vectors.shape[0])
                                 if patch_center_indices[loc_id] != INVAL_PC_IDX])
        logging.debug(f'Effect sizes for patch distances: {effect_sizes}')
        global_effect_size = self.anomaly_scorer.get_aggregated_score(effect_sizes)

        return global_effect_size, effect_sizes

class PatchcoreMemoryBank(PatchMemoryBankInterface):
    def __init__(self):
        print('Setting up PatchCore memory bank')
        self._nn_method = FaissNN(on_gpu=False, num_workers=4)  # set on_gpu=True if desired
        self._scorer = NearestNeighbourScorer(n_nearest_neighbours=1, nn_method=self._nn_method)


    def add_patches(self, template_feature_vectors):
        # detection_features expected to be np.ndarray of shape [N_patches, D]
        self._scorer.fit(detection_features=[template_feature_vectors])

    def compute_pairwise_distances(self):
        pass

    def detect_anomaly(self, test_feature_vectors, patch_center_indices):
        valid_test_feature_vectors = test_feature_vectors[patch_center_indices != INVAL_PC_IDX]
        # query_features: np.ndarray of shape [M_patches, D]
        # anomaly_scores: [M_patches], mean L2 distance to k nearest neighbors
        anomaly_scores, query_distances, query_nns = self._scorer.predict(query_features=[valid_test_feature_vectors])
        return np.max(anomaly_scores), anomaly_scores

class IndexedPatchMemoryBank(IndexedPatchMemoryBankBase):
    """This is a patch memory bank were the index denotes a location of the patches.
     There are multiple patches (from multiple templates) for a given location. The patch representation is NxD array
     """
    def __init__(self, distance_metric:str = 'cosine', top_k:int=1, aggregation_method:str='max'):
        super().__init__(distance_metric, top_k, aggregation_method)
        self.pair_dist = {}

    def compute_pairwise_distances(self):
        """Compute the pairwise distances between the patches in the memory bank.

        This pre-computes the pairwise distances for further anomaly deteciton, and also
        using these for debugging / visuaization.
        Used mostly for debugging.

        Returns:
            pairwise distances between template patches for each location.
            A measure of "uncertainty" about the patch location.
        """
        for loc_id in self.patch_lib.keys():
            # A new way - cosine istance
            self.pair_dist[loc_id] = self.distance.get_pairwise_distances(self.patch_lib[loc_id])
        logging.debug(f'Memory bank size: {len(self.patch_lib)}')
        patches_in_bank = [self.patch_lib[loc_id].shape[0] for loc_id in self.patch_lib.keys()]
        logging.debug(f'Num patches per location: f{patches_in_bank}')

        logging.debug(f'Memory bank contents: {self.patch_lib}')
        pairwise_distances = [self.pair_dist[loc_id] for loc_id in self.pair_dist.keys()]
        logging.debug(f'Patch bank pairwise distances: f{pairwise_distances}')
        logging.debug(f'Mean pairwise distances: f{np.mean(pairwise_distances, axis=1)}')

        # patch_lib_tensor = np.stack(list(self.patch_lib.values()))
        # np.save(f'patch_lib.npy', patch_lib_tensor)
        # pair_dist_tensor = np.stack(list(self.pair_dist.values()))
        # np.save(f'pair_dist.npy', pair_dist_tensor)
        # np.save(f'pairwise_distances.npy', pairwise_distances)
        return self.pair_dist


    def _get_distance_diff(self, loc_id, patch_test):
        """Computes the pairwise distances between representations of a patches in the bank and the query patch."""
        # Do the pairwise distance between the "train" points
        if loc_id not in self.pair_dist.keys():
            # A new way - cosine istance
            self.pair_dist[loc_id] = self.distance.get_pairwise_distances(self.patch_lib[loc_id])

        pair_dist = self.pair_dist[loc_id]
        # TODO (alexta): for FPFH it is very common to have exactly same FPFH descriptors that are [0, ..., 0] for some
        # points. To do any ablation studies for FPFH we need to account for this. But, in theory, this could
        # also introduce unnecessary noise and reduce accuracy - even for the deep learning features.
        # So disabling this for now.
        # if np.std(pair_dist) < MIN_PAIR_DIST_STD:
        #     # Regularization for FPFH: if pair distance std is too low (degenerate case),
        #     # add noise to prevent division-by-zero in Cohen's effect size computation.
        #     pair_dist = pair_dist + np.random.normal(loc=0.0, scale=np.sqrt(MIN_PAIR_DIST_STD), size=pair_dist.shape)

        # Distance from the test point to each train point
        dist_to_test_patch = self.distance.get_distance(self.patch_lib[loc_id], patch_test)


        # # Do the simplest: thing possible:
        # # ratio of an average dist. of vector to all reference points vs. average dist. between reference points
        # return np.mean(dist_to_test_patch) / np.mean(self.pair_dist[loc_id])

        # Treat these as 1d distributions. Compute Cohen's effect size. Usually effect size > 0.5 is a sign of a problem
        d = abs(pg.compute_effsize(pair_dist, dist_to_test_patch.flatten(), eftype='cohen'))
        return d


class IndexedPatchMemoryBankSingleTemplate(IndexedPatchMemoryBankBase):
    """A patch memory bank were the index denotes a location, but we compare query patch against a single template.

    Note: this is purely for experimentation and ablation studies; not an intended COSSAD functionality.
    """
    def __init__(self, distance_metric:str = 'cosine', top_k:int=1, aggregation_method:str='max'):
        super().__init__(distance_metric, top_k, aggregation_method)

    def compute_pairwise_distances(self):
        pass

    def _get_distance_diff(self, loc_id, patch_test):
        """Do a simple cosine distance between the test patch and the 1st patch in the bank."""
        assert self.patch_lib[loc_id] is not None
        # Get a random reference template
        idx_reference = np.random.randint(0, self.patch_lib[loc_id].shape[0])
        assert self.patch_lib[loc_id][idx_reference].shape[0] == patch_test.shape[0]
        return cosine(self.patch_lib[loc_id][idx_reference], patch_test)
