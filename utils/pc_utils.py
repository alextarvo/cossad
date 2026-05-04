from typing import Tuple
import logging
import utils.pc_utils as pc_utils
import transforms
import itertools
import constants

import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
from typing import List
import torch

INVAL_PC_IDX = -1

logger = logging.getLogger(__name__)

def np2o3d(np_pc):
    """Convert numpy array to o3d pointcloud"""
    assert(np_pc.shape[1] == 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pc)
    return pcd

def o3d2np(o3d_pc):
    """Convert o3d pointcloud to numpy array"""
    return np.array(o3d_pc.points)

def mean_average_distance(np_pc_dist_from, np_pc_dist_to):
    tree1 = cKDTree(np_pc_dist_to)
    dists, _ = tree1.query(np_pc_dist_from, k=1)
    return dists.mean()

def pc_center(np_pc):
    """Returns the center of the point cloud"""
    return np.mean(np_pc, axis=0)

def normalize_pc_tensor(tensor_pc, standardize=False):
    """Normalize point cloud to have 0 mean. We don't normalize it to the
    unit sphere since we did it already implicitly while generating patches"""
    centroid = torch.mean(tensor_pc, dim=0)
    pc = tensor_pc - centroid
    if standardize:
        std = torch.std(pc, dim=0, unbiased=False)
        pc = pc / (std + 1e-8)
    return pc

def normalize_pc(np_pc, standardize=False):
    centroid = np.mean(np_pc, axis=0)
    pc = np_pc - centroid
    if standardize:
        std = np.std(pc, axis=0)
        pc = pc / (std + 1e-8)
    return pc

def resample_pc_tensor(pc, num_points):
    """Randomly sample points from point cloud to yield exactly num_points"""
    if pc.shape[0] >= num_points:
        indices = np.random.choice(pc.shape[0], num_points, replace=False)
    else:
        indices = np.random.choice(pc.shape[0], num_points, replace=True)
    return pc[indices]

def resample_pc(np_pc, num_points):
    if np_pc.shape[0] >= num_points:
        indices = np.random.choice(np_pc.shape[0], num_points, replace=False)
    else:
        indices = np.random.choice(np_pc.shape[0], num_points, replace=True)
    return np_pc[indices]

def center_triplet_tensor(centering_type, anchor, positive, negative):
    """Center the triplet of tensors. Deprecated as of 11/1/2025"""
    if centering_type == constants.TripletCentereing.TEMPLATE:
        center = torch.mean(anchor, dim=0)
        return anchor-center, positive-center, negative-center
    elif centering_type == constants.TripletCentereing.COMMON:
        center = torch.cat((anchor, positive, negative)).mean(dim=0)
        return anchor - center, positive - center, negative - center
    elif centering_type == constants.TripletCentereing.SEPARATE:
        center_anchor = torch.mean(anchor, dim=0)
        center_positive = torch.mean(positive, dim=0)
        center_negative = torch.mean(negative, dim=0)
        return anchor-center_anchor, positive-center_positive, negative-center_negative
    raise ValueError(f'Unknown type of centering: {centering_type}')

def find_correspondences_and_patches(pc_ref, points_query, patch_radius):
    """
    Find points in the reference point cloud pc_ref that are closest to each point in  point_query.
    Then find patches that have these points as centers

    Args:
        pc_ref: (N, 3) reference point cloud (anchor or good)
        points_query: (M, 3) query points (from anomaly)

    Returns:
        ref_centers: (M, 3) points in pc_ref closest to each points_query
        ref_patches: [(?,3)] a list of M patches in pc_ref whose centers are ref_centers
    """
    # Build a kD tree for the reference point cloud
    tree_ref = cKDTree(pc_ref)
    # Find the points in the ref. pc closest to the points_query - their indices and
    # points themselves
    dists, ref_centers_indices = tree_ref.query(points_query, k=1)
    ref_centers = pc_ref[ref_centers_indices]
    # Now, for each point get the patch with a given radius.
    # Remember, KDTree returns not the points themselves but their indices
    ref_patches_indices = tree_ref.query_ball_point(ref_centers, r=patch_radius)
    ref_patches = [pc_ref[patch_indices] for patch_indices in ref_patches_indices]
    return ref_centers, ref_patches


def farthest_pt_sampling(pc, starting_point_idx, num_points):
    """
    A crude (and slow) implementation of the FPS algorithm.
    Args:
        pc: (N, 3) input point cloud
        starting_point_idx: an index of the starting poit in the pc
        num_points: the number of points to be sampled

    Returns:
        ref_centers: [num_points] array contining the sampled points in the reference point cloud
    """
    # An implementation of the farthest point sampling algorithm that will yield
    # more or less uniform sampling of PC points
    # Fill out the return array of farthest indices
    farthest_indices = np.zeros(num_points, dtype=int)
    farthest_indices[0] = starting_point_idx
    # Maintain the array of minimum distances from the currently selected point set
    # to the rest of the point cloud.
    distances = np.full(pc.shape[0], np.inf)
    # Init the index of the most recently added "most distant" point.
    farthest_idx = starting_point_idx
    for i in range(1, num_points):
        # Vectorized distance computation from the last point to the rest of the PC
        dist = np.linalg.norm(pc - pc[farthest_idx], axis=1)
        # For each point of the PC, update the minimum distance to the set of the
        # currently selected points. Note: the min. distance from a PC point
        # to a currently selected set can't _increase_, it may only decrease
        distances = np.minimum(distances, dist)
        # Select the next most distant point
        farthest_idx = np.argmax(distances)
        farthest_indices[i] = farthest_idx
    return farthest_indices


def adaptive_farthest_pt_sampling(np_pc, starting_point_idx, patch_radius,
                                  min_num_points_threshold, max_num_points_threshold,
                                  min_coverage_threshold, max_coverage_threshold):
    """
    A crude (and slow) implementation of the FPS algorithm that maintains a desired coverage of the PC by
    patches of a given radius. Coverage is computed for each point, as the number of patches the point belongs to.
    Args:
        np_pc: (N, 3) input point cloud
        starting_point_idx: an index of the starting poit in the pc
        patch_radius: the radius of a patch
        min_num_points_threshold, max_num_points_threshold: the number of central points to be sampled from the PC
        min_coverage_threshold, max_coverage_threshold: minimum and maximum coverage of the PC by the patches.

    Returns:
        farthest_indices: [num_points] array containing the sampled central points in the PC
    """
    # Initialize the coverage of the PC
    tree = cKDTree(np_pc)
    coverage = np.zeros(np_pc.shape[0])
    covered_indices = tree.query_ball_point(np_pc[starting_point_idx], patch_radius)
    coverage[covered_indices] += 1

    # An implementation of the farthest point sampling algorithm that will yield
    # more or less uniform sampling of PC points
    # Fill out the return array of farthest indices
    farthest_indices = np.zeros(max_num_points_threshold, dtype=int)
    farthest_indices[0] = starting_point_idx
    # Maintain the array of minimum distances from the currently selected point set
    # to the rest of the point cloud.
    distances = np.full(np_pc.shape[0], np.inf)
    # Init the index of the most recently added "most distant" point.
    farthest_idx = starting_point_idx
    for i in range(1, max_num_points_threshold):
        # Vectorized distance computation from the last point to the rest of the PC
        dist = np.linalg.norm(np_pc - np_pc[farthest_idx], axis=1)
        # For each point of the PC, update the minimum distance to the set of the
        # currently selected points. Note: the min. distance from a PC point
        # to a currently selected set can't _increase_, it may only decrease
        distances = np.minimum(distances, dist)
        # Select the next most distant point
        farthest_idx = np.argmax(distances)
        farthest_indices[i] = farthest_idx

        # Now, update the coverage of the PC by the patches
        covered_indices = tree.query_ball_point(np_pc[farthest_idx], patch_radius)
        coverage[covered_indices] += 1

        if np.min(coverage) >= min_coverage_threshold and np.max(coverage) <= max_coverage_threshold \
            and min_num_points_threshold <= i <= max_num_points_threshold:
            #  Algorithm finished successfully
            break
        if np.min(coverage) < min_coverage_threshold and i > max_num_points_threshold:
            logger.warning(f'The minimum coverage of the PC  {np.min(coverage)} is still less than {min_coverage_threshold}'
                           f' but the number of patches exceeds the threshold {max_num_points_threshold}')
            break
        if np.max(coverage) > max_coverage_threshold and i < min_num_points_threshold:
            logger.warning(f'The minimum coverage of the PC  {np.max(coverage)} is higher less than {max_coverage_threshold}'
                           f' but the number of patches is less than the threshold {min_num_points_threshold}')
            break
        if i > max_num_points_threshold or np.max(coverage) > max_coverage_threshold:
            logger.warning(f'FPS terminated prematurily')
            logger.warning(f'The min coverage of the PC  {np.min(coverage)} (threshold: {min_coverage_threshold}); '
                         f'max coverage: {np.max(coverage)} (threshold: {max_coverage_threshold})')
            logger.warning(f'The number of centers {i}; low/max thresholds: {min_num_points_threshold}, {max_num_points_threshold}')
            break
    return farthest_indices[:i]

def randomly_rotate_pcs(np_pcs: List[np.ndarray]) -> Tuple[np.ndarray, ...]:
    """Rotate PCs in the list of np_pcs by the same random amount. """
    mads_before = [pc_utils.mean_average_distance(np_pcs[i], np_pcs[i+1])
                   for i in range(len(np_pcs)-1)
                  ]
    rot_matrix = transforms.get_random_rotation_matrix((-90, 90), (-90, 90), (-90, 90))
    np_pcs_rotated = [pc @ rot_matrix for pc in np_pcs]
    mads_after = [pc_utils.mean_average_distance(np_pcs_rotated[i], np_pcs_rotated[i+1])
                   for i in range(len(np_pcs_rotated)-1)
                 ]
    assert len(mads_before) == len(mads_after)
    assert len(np_pcs) == len(np_pcs_rotated)
    for mad_before, mad_after in zip(mads_before, mads_after):
        # Here we verify that the mean average distance was not affected by the rotation (defensive programming)
        assert np.allclose(mad_before, mad_after, rtol=0.05), \
            (f'As a result of random rotation, MAD between good and bad pointclouds changed from'
             f' {mad_before} to {mad_after}')
    return tuple(np_pcs_rotated)

def stack_pcs(patches: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Collate separate tensors [Ni, 3] into a single tensor [N1,+...,Nn, 3].
    Returns the collated tensors and the indices of relative positions of each Ni in it
    """
    if len(patches) == 0:
        return np.array([]), np.array([])
    if len(patches) == 1:
        return np.array([]), patches[0]
    # Compute the relative offsets of individual PC patches in the array
    ret_offsets = list(itertools.accumulate(t.shape[0] for t in patches[:-1]))
    assert len(ret_offsets) == len(patches)-1
    patches_stacked = np.concatenate(patches)
    return np.array(ret_offsets), patches_stacked

def unstack_pcs(offsets: np.ndarray, patches_stacked:np.ndarray) -> List[np.ndarray]:
    """Collate separate tensors [Ni, 3] into a single tensor [N1,+...,Nn, 3].
    Returns the collated tensors and the indices of relative positions of each Ni in it
    """
    if patches_stacked.shape[0] == 0:
        return []
    if offsets.shape[0] == 0:
        return [patches_stacked]
    return np.split(patches_stacked, offsets)
