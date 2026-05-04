from dataclasses import dataclass
import numpy as np

@dataclass
class RegistrationAccuracy:
    mean_dist_src_to_target: float
    max_dist_src_to_target: float
    mean_dist_target_to_src: float
    max_dist_target_to_src: float
    chamfer_distance: float

    def __init__(self, pcd_source, pcd_target):
        dists_src_to_target = pcd_source.compute_point_cloud_distance(pcd_target)
        dists_target_to_src = pcd_target.compute_point_cloud_distance(pcd_source)
        self.mean_dist_src_to_target = np.mean(dists_src_to_target)
        self.max_dist_src_to_target = np.max(dists_src_to_target)
        self.mean_dist_target_to_src = np.mean(dists_target_to_src)
        self.max_dist_target_to_src = np.max(dists_target_to_src)
        self.chamfer_distance = self.mean_dist_src_to_target + self.mean_dist_target_to_src
