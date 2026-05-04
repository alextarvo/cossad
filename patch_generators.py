import logging

import numpy as np
from utils.pc_utils import farthest_pt_sampling, INVAL_PC_IDX
import copy
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class PatchGenerator(object):
    """This is an interface class that generates patches during anomaly search (inference).

    We will compute features for each of these patches, and then detect anomalies by comparing features.
    """

    def __init__(self):
        pass

    def get_point_samples(self, np_pc):
        raise NotImplementedError


class SimpleSamplerPatchGenerator(PatchGenerator):
    """A simplest patch generator that just samples points deterministically."""

    def __init__(self, sampling_rate):
        super(SimpleSamplerPatchGenerator, self).__init__()
        self.sampling_rate = sampling_rate

    def get_point_samples(self, np_pc):
        num_points = np_pc.shape[0]
        sample_indices = [i * self.sampling_rate for i in range(int(num_points / self.sampling_rate))]
        return np.array(sample_indices), np_pc[sample_indices], np.array([], dtype=int)


class FPSRandomizedPatchGenerator(PatchGenerator):
    """
    FPS sampling for the PC with the random initial point. Normally used to generate the reference
    sampling points for the template PC.
    """

    def __init__(self, num_points):
        super(FPSRandomizedPatchGenerator, self).__init__()
        self.num_points = num_points

    def get_point_samples(self, np_pc):
        start_point_idx = np.random.choice(np_pc.shape[0])
        patch_centers_indices = farthest_pt_sampling(np_pc, start_point_idx, self.num_points)
        return np.array(patch_centers_indices), np_pc[patch_centers_indices], np.array([], dtype=int)


class FPSFixedPatchGenerator(PatchGenerator):
    """
    Takes the reference template and generates a fixed points where the features will be sampled.
    Every call to get_point_samples with a new cloud will be sampled at exactly the same points.
    """

    def __init__(self, num_points, np_template_pc, radius, filter_by_point_count=False):
        super(FPSFixedPatchGenerator, self).__init__()
        self.num_points = num_points
        start_point_idx = np.random.choice(np_template_pc.shape[0])
        patch_centers_indices = farthest_pt_sampling(np_template_pc, start_point_idx, self.num_points)
        self.patch_centers = copy.deepcopy(np_template_pc[patch_centers_indices])
        self.radius = radius
        self.filter_by_point_count = filter_by_point_count
        self.point_count_coefficient = 0.0

    def set_filter_by_point_count(self, do_filter, point_count_coefficient=0.0):
        self.filter_by_point_count = do_filter
        self.point_count_coefficient = point_count_coefficient

    def get_point_samples(self, np_pc):
        """
        Vectorized approach using cKDTree. Returns array of indices of closest test_cloud points
        or None where the neighborhood condition is not satisfied.
        """
        tree = cKDTree(np_pc)

        # Get points within radius for all patch centers
        points_lists = tree.query_ball_point(self.patch_centers, r=self.radius)

        # Check if each patch has enough points in it.
        # has_enough_points = np.array([len(neighbors) >= self.min_neighbors for neighbors in neighbors_list])
        num_points = np.array([len(neighbors) for neighbors in points_lists])

        if self.filter_by_point_count:
            # Try to deduce the lower threshold on a patch number of points. Take only patches whose
            # number of points is at least the mean over the whole PC...
            # This is necessary if our "test" PC is a 2.5D view - i.e. top or bottom. We don't want
            # to compare the missing parts to the template, and we also don't want to compare PC boundary
            # as it is likely will be noisy.
            # TODO(alex): this is hacky, and this is specific for real3dad. Think of a better way!
            has_nz_points = np.array([len(points) > 0 for points in points_lists])
            # Here we are setting actual threshold.
            # The more we subtract from the mean, the more points we leave in PC. The more we add, the less points we retain.
            # min_points = np.mean(num_points[has_nz_points]) is a middleground, works well for eveyone except seahorse and starfish.
            min_points = np.mean(num_points[has_nz_points]) + np.std(num_points[has_nz_points]) * self.point_count_coefficient
            has_enough_points = np.array([len(points) >= min_points for points in points_lists])
        else:
            has_enough_points = np.ones(num_points.shape)

        # Compute closest points in test_cloud to all patch_centers
        _, indices_all = tree.query(self.patch_centers)
        center_points_all = np_pc[indices_all]

        # Track which patches were skipped due to insufficient point count
        has_not_enough_points = ~has_enough_points.astype(bool)
        skipped_patch_centers_indices = np.where(has_not_enough_points, indices_all, INVAL_PC_IDX)

        # Apply mask - set index to INVAL_PC_IDX or nan if not enough neighbors were selected
        patch_centers_indices = np.where(has_enough_points, indices_all, INVAL_PC_IDX)
        patch_center_points = np.where(has_enough_points[:, np.newaxis], center_points_all, np.nan)

        return patch_centers_indices, patch_center_points, skipped_patch_centers_indices

#
# class FPSFixedPatchGeneratorTemplateCentered(FPSFixedPatchGenerator):
#     """
#     Takes the reference template and returns a list of points - patch centers.
#     NOTE: these are not coordinates of points in the PC. Just patch centers
#     """
#
#     def __init__(self, num_points, np_template_pc, radius, filter_by_point_count=False):
#         super(FPSFixedPatchGeneratorTemplateCentered, self).__init__(
#             num_points, np_template_pc, radius, filter_by_point_count)
#
#     def get_point_samples(self, np_pc):
#         """
#         Vectorized approach using cKDTree. Returns array of indices of closest test_cloud points
#         or None where the neighborhood condition is not satisfied.
#         """
#         tree = cKDTree(np_pc)
#
#         # Get points within radius for all patch centers
#         points_lists = tree.query_ball_point(self.patch_centers, r=self.radius)
#
#         # Check if each patch has enough points in it.
#         # has_enough_points = np.array([len(neighbors) >= self.min_neighbors for neighbors in neighbors_list])
#         num_points = np.array([len(neighbors) for neighbors in points_lists])
#
#         if self.filter_by_point_count:
#             # Try to deduce the lower threshold on a patch number of points. Take only patches whose
#             # number of points is at least the mean over the whole PC...
#             # This is necessary if our "test" PC is a 2.5D view - i.e. top or bottom. We don't want
#             # to compare the missing parts to the template, and we also don't want to compare PC boundary
#             # as it is likely will be noisy.
#             # TODO(alex): this is hacky, and this is specific for real3dad. Think of a better way!
#             has_nz_points = np.array([len(points) > 0 for points in points_lists])
#             min_points = np.mean(num_points[has_nz_points]) - np.std(num_points[has_nz_points])  # /2, /3 etc
#             has_enough_points = np.array([len(points) >= min_points for points in points_lists])
#         else:
#             has_enough_points = np.ones(num_points.shape)
#
#         center_points_all = copy.deepcopy(self.patch_centers)
#
#         # Apply mask - set index to INVAL_PC_IDX or nan if not enough neighbors were selected
#         patch_center_points = np.where(has_enough_points[:, np.newaxis], center_points_all, np.nan)
#         return patch_center_points
