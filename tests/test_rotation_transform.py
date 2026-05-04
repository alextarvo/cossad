# test_random_rotation_transform_o3d.py
import math
import torch
import pytest
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

import transforms as tr

def visualize_point_clouds(pcs):
    """
    Visualize up to 3 numpy arrays of shape (N,3) in Open3D with different colors.
    """
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # red, green, blue
    geometries = []
    for i, pc in enumerate(pcs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        col = np.tile(colors[i], (pc.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(col)
        geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries)


def chamfer_o3d(A_xyz: np.ndarray, B_xyz: np.ndarray) -> float:
    """Chamfer distance using Open3D's nearest-neighbor distances (both directions)."""
    pcA = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(A_xyz))
    pcB = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(B_xyz))
    dAB = np.asarray(pcA.compute_point_cloud_distance(pcB)).mean()
    dBA = np.asarray(pcB.compute_point_cloud_distance(pcA)).mean()
    return float(dAB + dBA)


def hemisphere_points(radius: float, center=(0.0, 0.0, 0.0), n_points: int = 2048) -> torch.Tensor:
    """
    Generate hemisphere points on the SURFACE using Open3D:
    1) make a sphere mesh, 2) sample uniformly, 3) keep z>=0, 4) shift & scale.
    """
    cx, cy, cz = center
    xy = np.random.uniform(-radius, radius, size=(n_points, 2))
    z_sample = radius ** 2 - np.sum(xy ** 2, axis=1)
    z_filter = z_sample[z_sample >= 0]
    z = np.sqrt(z_filter)
    pts = np.hstack((xy[z_sample >= 0, :], z.reshape(-1, 1)))
    pts += center
    return torch.from_numpy(pts.astype(np.float32))  # [N,3]


def test_rotation_transform():
    points1 = hemisphere_points(radius=2, center=(0.0, 0.0, 0.0), n_points=4096)[:1000]
    points2 = hemisphere_points(radius=2, center=(0.0, 0.0, 0.5), n_points=4096)[:1000]
    points3 = hemisphere_points(radius=2, center=(0.0, 0.0, 1.0), n_points=4096)[:1000]

    ch12 = chamfer_o3d(points1, points2)
    ch13 = chamfer_o3d(points1, points3)
    ch23 = chamfer_o3d(points2, points3)

    points_batch = torch.cat((points1.unsqueeze(0), points2.unsqueeze(0), points3.unsqueeze(0)))

    for i in range(1000):
        rrt = tr.RandomRotationTransform(angle_range_deg=(-90, 90))
        points_batch_transformed = rrt(points_batch)

        ch12_trans = chamfer_o3d(points_batch_transformed[0], points_batch_transformed[1])
        ch13_trans = chamfer_o3d(points_batch_transformed[0], points_batch_transformed[2])
        ch23_trans = chamfer_o3d(points_batch_transformed[1], points_batch_transformed[2])

        assert(np.isclose(ch12, ch12_trans, rtol=1e-3))
        assert(np.isclose(ch13, ch13_trans, rtol=1e-3))
        assert(np.isclose(ch23, ch23_trans, rtol=1e-3))

    # visualize_point_clouds((points1, points2, points3))
    # visualize_point_clouds((points_batch_transformed[0], points_batch_transformed[1], points_batch_transformed[2]))
