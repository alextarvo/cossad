"""ICP registration module."""

import open3d as o3d
import numpy as np

from . import util_registration as util_reg
from . import base_registration as base_reg

import logging
logger = logging.getLogger(__name__)


class RegistrationICP(base_reg.RegistrationBase):
    """Uses ICP algorithm to register point clouds (PCs)"""

    def __init__(self, np_pc_template=None, np_pc_target=None, desired_num_voxels=None, initial_voxel_size=0.5, icp_distance_threshold=0.4):
        super(RegistrationICP, self).__init__(np_pc_template, np_pc_target)
        self._voxel_size = initial_voxel_size
        self._icp_distance_threshold = icp_distance_threshold
        self._result_ransac = None
        self._result_icp = None
        # This is the desired number of voxels we want to have in our PC for the registration
        self.find_optimal_voxel_count_ = False
        if desired_num_voxels is not None:
            self._desired_num_voxels = desired_num_voxels
            self.find_optimal_voxel_count_ = True

    def _optimal_voxelize_and_downsample(self, o3d_pc):
        """Downsample the PC.
        If requested, for template, search for the optimal voxel to voxelize this point cloud to desired_num_voxels"""
        if not self.find_optimal_voxel_count_:
            pcd_downsampled = o3d_pc.voxel_down_sample(self._voxel_size)
            return pcd_downsampled

        current_voxel_size = self._voxel_size
        voxel_step = 0.5
        pcd_downsampled = o3d_pc.voxel_down_sample(self._voxel_size)
        binary_search_iter_ct = 0
        # Do binary search to find the optimum num. voxels
        while (len(pcd_downsampled.points) > self._desired_num_voxels * 1.2) or \
                (len(pcd_downsampled.points) < self._desired_num_voxels / 1.2):
            if len(pcd_downsampled.points) > self._desired_num_voxels * 1.2:
                current_voxel_size = current_voxel_size + current_voxel_size * voxel_step
            elif len(pcd_downsampled.points) < self._desired_num_voxels / 1.2:
                current_voxel_size = current_voxel_size - current_voxel_size * voxel_step
            else:
                assert True, 'Error while searching for optimal voxel size!'
            pcd_downsampled = o3d_pc.voxel_down_sample(current_voxel_size)
            binary_search_iter_ct += 1
            assert binary_search_iter_ct < 100, \
                'Exceeded the number of iterations while searching for optimal voxel size'
        self._voxel_size = current_voxel_size
        logger.debug(f'Found the optimal voxel size {self._voxel_size}')
        return pcd_downsampled

    def _preprocess_point_cloud(self, o3d_pc):
        """downsample the point cloud, compute its normals and FPFH descriptors."""

        pcd_downsampled = self._optimal_voxelize_and_downsample(o3d_pc)
        logger.debug(f'Downsampled the PC with voxel {self._voxel_size} from {len(o3d_pc.points)} '
                     f'to {len(pcd_downsampled.points)} points')

        radius_normal = self._voxel_size * 2
        pcd_downsampled.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self._voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_downsampled, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_downsampled, pcd_fpfh

    def _prepare_dataset(self):
        """Convert the source and target point clouds from Numpy 3D arrays into the Open3D unorganized point cloud.
        During the pre-processing, downsample the point cloud, compute its normals and FPFH descriptors.

        Returns:
            for each object, return its unorganized PC in original resolution; a downsampled one; and its FPFH descriptors
        """
        o3d_pc_template = o3d.geometry.PointCloud()
        o3d_pc_template.points = o3d.utility.Vector3dVector(self._np_pc_template)
        o3d_pc_target = o3d.geometry.PointCloud()
        o3d_pc_target.points = o3d.utility.Vector3dVector(self._np_pc_target)

        # alexta: This seems to be purely for testing purposees  - rotate source point cloud
        # summary_pointcloud(target)
        # source = random_rotate(source)
        # draw_registration_result(source, target, np.identity(4))

        # Note: here we need to downsample the PC to a smaller number of points to ensure registration ends in a
        # reasonable time. So we downsample template only. Our invariant is the "template" is _always_ a superset
        # of a "target" PC; they could be roughly the same (as in AShapeNet, MulSen) or target is a half-shape
        # (Real3D-AD), but not vice versa.
        # So finx optimal voxel size from template, and then use the same voxel size for the target
        # o3d_pc_template_down, template_fpfh = self._preprocess_point_cloud(o3d_pc_template, True)
        o3d_pc_template_down, template_fpfh = self._preprocess_point_cloud(o3d_pc_template)
        # template_voxel_size = self.voxel_size
        o3d_pc_target_down, target_fpfh = self._preprocess_point_cloud(o3d_pc_target)
        #
        # Old code where we ensure that target and template have more or less voxel size
        # if template_voxel_size < self.voxel_size * 0.5 or template_voxel_size > self.voxel_size * 1.5:
        #     # Since we are registering the same object, we really expect the num. voxels will be same for
        #     raise ValueError(f'Search for optimal voxel size is inconsistent:'
        #                      f' template voxel size is {template_voxel_size}; target voxel size is {self.voxel_size}')
        return o3d_pc_template, o3d_pc_target, o3d_pc_template_down, o3d_pc_target_down, template_fpfh, target_fpfh

    def _execute_global_registration(self, o3d_pc_target_down, o3d_pc_template_down, target_fpfh, template_fpfh):
        """ Do an _initial_ global registration of point clouds using global RANSAC-based registration"""
        # distance_threshold = voxel_size * 1.5
        distance_threshold = self._voxel_size * 1.2
        # print(":: RANSAC registration on downsampled point clouds.")
        # print("   Since the downsampling voxel size is %.3f," % voxel_size)
        # print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            o3d_pc_target_down, o3d_pc_template_down, target_fpfh, template_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def compute_registration_transform(self) -> bool:
        o3d_pc_template, o3d_pc_target, o3d_pc_template_down, o3d_pc_target_down, template_fpfh, target_fpfh = (
            self._prepare_dataset())
        result_ransac = self._execute_global_registration(
            o3d_pc_target_down, o3d_pc_template_down, target_fpfh, template_fpfh)
        o3d_pc_target.transform(result_ransac.transformation)
        result_icp = o3d.pipelines.registration.registration_icp(
            o3d_pc_target, o3d_pc_template, self._icp_distance_threshold)
        o3d_pc_target.transform(result_icp.transformation)
        #    draw_registration_result(source, target, np.identity(4))
        self._np_pc_target_registered = np.asarray(o3d_pc_target.points)
        self._accuracy = util_reg.RegistrationAccuracy(o3d_pc_target, o3d_pc_template)
        logger.debug(f'ICP registration complete. Accuracy: {self._accuracy}')
        self._result_ransac = result_ransac
        self._result_icp = result_icp
        return True

    def get_transform(self):
        return self._result_icp.transformation @ self._result_ransac.transformation


class ICPRegistrationFactory(base_reg.RegistrationFactory):
    def __init__(self, **defaults):
        self._defaults = defaults

    def __call__(self, **overrides) -> RegistrationICP:
        params = {**self._defaults, **overrides}
        return RegistrationICP(**params)

