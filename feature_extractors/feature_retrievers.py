"""Classes that retrieve features for a patch in a point cloud."""
import logging

import numpy as np
import open3d as o3d
from wandb.integration.prodigy.prodigy import standardize

from feature_extractors.encoders_pointnet import PointNetEncoder, PointNet2Encoder
from feature_extractors.encoders_base import instantiate_model
import torch
import torch.nn.functional as F
from dataloaders import patch_triplet_dataloaders as patch3_dloaders
from scipy.spatial import cKDTree
from utils.debug import assert_nans_tensor
from utils.logging_util import log_nans_tensor_error
import utils.pc_utils as pc_utils

logger = logging.getLogger(__name__)

class FeatureRetriever(object):
    def get_features(self, np_pc):
        return None, None, None, None, None


class FPFHFeatureRetriever(FeatureRetriever):
    def __init__(self, patch_generator, voxel_size=0.5):
        super(FPFHFeatureRetriever, self).__init__()
        self.voxel_size = voxel_size
        self.patch_generator = patch_generator


    def get_features(self, np_pc):
        # Transfer NP point cloud into Open3D and compute normals.
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np_pc))
        radius_normal = self.voxel_size * 2
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        # Now given normals compute FPFH features
        radius_feature = self.voxel_size * 5
        fpfh_set = o3d.pipelines.registration.compute_fpfh_feature(
            o3d_pc, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        patch_indices, fpfh_sampled, patch_indices_skipped = self.patch_generator.get_point_samples(fpfh_set.data.T)
        return patch_indices, torch.from_numpy(fpfh_sampled), None, None, patch_indices_skipped


class FPFHPerPatchFeatureRetriever(FeatureRetriever):
    def __init__(self, patch_generator, patch_radius):
        super(FPFHPerPatchFeatureRetriever, self).__init__()
        self.patch_generator = patch_generator
        self.patch_radius = patch_radius
        self.normal_radius_scale = 0.5

    def resample_pc(self, pc: np.ndarray, num_points: int) -> np.ndarray:
        """
        Randomly sample points from point cloud to yield exactly `num_points`.

        Args:
            pc (np.ndarray): [N, 3] point cloud.
            num_points (int): Number of points to sample.

        Returns:
            np.ndarray: [num_points, 3] resampled point cloud.
        """
        if pc.shape[0] >= num_points:
            indices = np.random.choice(pc.shape[0], num_points, replace=False)
        else:
            indices = np.random.choice(pc.shape[0], num_points, replace=True)
        return pc[indices]

    def fpfh_for_patch(self, patch):
        if patch.shape[0] < 5:
            return np.zeros(33, dtype=np.float32)
        norm_r = self.patch_radius * self.normal_radius_scale
        # Build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(patch)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=norm_r, max_nn=32)
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.patch_radius, max_nn=512)
        )
        fpfh_all = np.asarray(fpfh.data, dtype=np.float64).T  # (N, 33)
        center_idx = int(np.argmin(np.sum(patch ** 2, axis=1)))
        fpfh_patch = fpfh_all[center_idx]
        return fpfh_patch

    def get_features(self, np_pc):
        # split the PC into patches. Here we are getting indices in the np_pc point cloud of central points for these patches.
        # Note: it may be possible that some patches are "none", i.e.. not available. These are -1s
        patch_centers_indices_unfiltered, _, patch_centers_indices_skipped = self.patch_generator.get_point_samples(np_pc)
        patch_centers_indices_skipped = patch_centers_indices_skipped[patch_centers_indices_skipped != -1]
        # Now, some center indices may have value of -1 - i.e. no valid center is there. Skip them.
        # but remember their locations (For future)
        patch_centers_valid_entries = np.where(patch_centers_indices_unfiltered != -1)[0]
        # These are indices only of those patch centers that are present in this (possibly incomplete) PC
        patch_centers_indices_filtered = patch_centers_indices_unfiltered[patch_centers_valid_entries]
        tree = cKDTree(np_pc)
        patch_points_indices = tree.query_ball_point(np_pc[patch_centers_indices_filtered], self.patch_radius)
        logger.debug(f'Out of {len(patch_centers_indices_unfiltered)} patches, filtered {len(patch_centers_indices_filtered)} valid ones')

        features = []
        for patch_point_index in patch_points_indices:
            patch = np_pc[patch_point_index]
            sampled_patch = self.resample_pc(patch, 512)
            center = np.mean(sampled_patch, axis=0)
            normalized_patch = sampled_patch - center
            fpfh = self.fpfh_for_patch(normalized_patch)
            features.append(fpfh)
        features_filtered = torch.tensor(features, dtype=torch.float32)

        features_all = torch.full((patch_centers_indices_unfiltered.shape[0], features_filtered.shape[1]), float('nan'))
        assert(features_filtered.shape[0] == patch_centers_indices_filtered.shape[0])
        features_all[patch_centers_valid_entries,:] = features_filtered
        log_nans_tensor_error('NaNs in features_all', features_all, 0.0)

        return patch_centers_indices_unfiltered, features_all, patch_centers_indices_filtered, features_filtered, patch_centers_indices_skipped


class ContrastiveFeatureRetriever(FeatureRetriever):
    def __init__(self, patch_generator, model_name, model_path, patch_radius, device_idx=0, embedding_dim=64, points_per_patch=512,
                 standardize_patch=True):
        super(ContrastiveFeatureRetriever, self).__init__()
        self.patch_generator = patch_generator
        self.device = f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu'
        self.model = instantiate_model(model_name, embedding_dim, normalize_embeddings=True).to(self.device)
        print('Active device: ', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        # Note: this must be the same radius as we used in patch generatoor to generate feature patches
        self.patch_radius = patch_radius
        # Note: these must be the same settings we used in our implementation of the deep learning feature extractor
        self.points_per_patch = points_per_patch
        self.embedding_dim = embedding_dim
        # We can't run a full batch inference on PointNet++. So we pass patches in a minibatches of ths size.
        self.minibatch_size = 64
        self.standardize_patch = standardize_patch

    def get_features(self, np_pc):
        # split the PC into patches. Here we are getting indices in the np_pc point cloud of central points for these patches.
        # Note: it may be possible that some patches are "none", i.e.. not available. These are -1s
        patch_centers_indices_unfiltered, _, patch_centers_indices_skipped = self.patch_generator.get_point_samples(np_pc)
        patch_centers_indices_skipped = patch_centers_indices_skipped[patch_centers_indices_skipped != -1]
        # Now, some center indices may have value of -1 - i.e. no valid center is there. Skip them.
        # but remember their locations (For future)
        patch_centers_valid_entries = np.where(patch_centers_indices_unfiltered != -1)[0]
        # These are indices only of those patch centers that are present in this (possibly incomplete) PC
        patch_centers_indices_filtered = patch_centers_indices_unfiltered[patch_centers_valid_entries]

        tree = cKDTree(np_pc)
        patch_points_indices = tree.query_ball_point(np_pc[patch_centers_indices_filtered], self.patch_radius)
        logger.debug(f'Out of {len(patch_centers_indices_unfiltered)} patches, filtered {len(patch_centers_indices_filtered)} valid ones')

        patch_set = []
        features_set = []
        for patch_point_index in patch_points_indices:
            patch = np_pc[patch_point_index]
            # resampled_patch = patch3_dloaders.DatasetPatchTriplet.resample_pc(patch, self.points_per_patch)
            # normalized_patch = patch3_dloaders.DatasetPatchTriplet.normalize_pc(
            #     torch.from_numpy(resampled_patch).float()).to(self.device)

            patch_tensor = torch.from_numpy(patch).float().to(self.device)
            normalized_patch = pc_utils.normalize_pc_tensor(patch_tensor, self.standardize_patch)
            resampled_patch_tensor = pc_utils.resample_pc_tensor(normalized_patch, self.points_per_patch)
            patch_set.append(resampled_patch_tensor)

            #TODO(alexta): this is a new code. Just got a weird perf drop on it - not sure if
            # regression or just an accident.
            # normalized_patch = pc_utils.normalize_pc(patch, self.standardize_patch)
            # resampled_patch = pc_utils.resample_pc(normalized_patch, self.points_per_patch)
            # patch_set.append(resampled_patch)

        n_splits = len(patch_set) // self.minibatch_size
        patch_minibatches = torch.tensor_split(torch.stack(patch_set), n_splits)
        # patch_set_tensor =  torch.from_numpy(np.stack(patch_set)).float().to(self.device)
        # patch_minibatches = torch.tensor_split(patch_set_tensor, n_splits)
        with torch.no_grad():
            for minibatch in patch_minibatches:
                out_features, _, _ = self.model(minibatch)
                out_features = out_features.detach().cpu()
                assert_nans_tensor('Contrastive deep feature extractor output:', out_features, 0.0)
                features_set.append(out_features)

        # This would work if we could fit the whole batch into GPU RAM
        # batch = torch.stack(patch_set, dim=0)
        # minibatches = batch.split()
        # features =self.model(batch).detach().cpu()

        features = torch.cat(features_set, dim=0)
        # May help to empty GPU cache and save some memory
        torch.cuda.empty_cache()
        features_filtered = features.detach().cpu()

        features_all = torch.full((patch_centers_indices_unfiltered.shape[0], features_filtered.shape[1]), float('nan'))
        assert(features_filtered.shape[0] == patch_centers_indices_filtered.shape[0])
        features_all[patch_centers_valid_entries,:] = features_filtered
        log_nans_tensor_error('NaNs in features_all', features_all, 0.0)

        return patch_centers_indices_unfiltered, features_all, patch_centers_indices_filtered, features_filtered, patch_centers_indices_skipped
#
# class ContrastiveFeatureRetrieverTemplateCentered(ContrastiveFeatureRetriever):
#     def __init__(self, patch_generator, model_name, model_path, centering, embedding_dim=64, points_per_patch=512, radius_feature=2):
#         super(ContrastiveFeatureRetrieverTemplateCentered, self).__init__(
#             patch_generator, model_name, model_path, centering, embedding_dim=64, points_per_patch=512, radius_feature=2)
#
#     def get_features(self, np_pc):
#         patch_centers = self.patch_generator.get_point_samples(np_pc)
#         patch_centers_valid_entries = np.where(patch_centers != np.nan)[0]