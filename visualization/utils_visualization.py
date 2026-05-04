import math

import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt
import utils.pc_utils as pc_utils

def create_pointcloud_with_anomaly_mask_colored(np_points, anomaly_mask=None, do_normalize_mask=False):
    """Colorize the point cloud for visualization.

    Args:
        np_points: O3D PC with N points
        anomaly_mask: an integer ndarray size N; i-th point in np_points is anomalous if anomaly_mask[i] == 1
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)

    if anomaly_mask is None:
        # Light blue color
        colors = np.tile([0.6, 0.8, 1.0], (np_points.shape[0], 1))
    else:
        if not do_normalize_mask:
            # This is the actual mask. Do simple color map: light blue for normal, light red for anomalies
            colors = np.tile([0.6, 0.8, 1.0], (np_points.shape[0], 1))
            colors[anomaly_mask == 1] = [1.0, 0.6, 0.6]
        else:
            # This is a prediction anomaly score. Normalize and convert to the color map
            mask_norm = (anomaly_mask - anomaly_mask.min()) / (anomaly_mask.max() - anomaly_mask.min() + 1e-12)
            cmap = plt.get_cmap("viridis")
            colors = cmap(mask_norm)[:, :3]  # drop alpha channel
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def show_registered_pointcloud(o3d_pc_template, o3d_pc_target, transformation=np.identity(4), window_name=''):
    """Visualize the result of a registration for a single point cloud.

    Args:
        o3d_pc_template: a template PC in the Open3D format
        o3d_pc_target: a target (one to be transformed) PC in the Open3D format
        transformation: a transformation to apply to the o3d_pc_target (optional)
        window_name: the caption of the window (optional)
    """
    target_temp = copy.deepcopy(o3d_pc_target)
    template_temp = copy.deepcopy(o3d_pc_template)
    target_temp.paint_uniform_color([1, 0.706, 0])
    template_temp.paint_uniform_color([0, 0.651, 0.929])
    target_temp.transform(transformation)
    o3d.visualization.draw_geometries([target_temp, template_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556],
                                      window_name=window_name)


def show_prediction(np_test_pc, np_anomaly_mask, np_prediction_scores, window_title=''):
    spacing = 1.2  # for layout

    test_pcd = create_pointcloud_with_anomaly_mask_colored(np_test_pc, anomaly_mask=np_anomaly_mask)
    predict_pcd = create_pointcloud_with_anomaly_mask_colored(
        np_test_pc, anomaly_mask=np_prediction_scores, do_normalize_mask=True)
    # Normalize everything for layout
    all_geoms = [test_pcd, predict_pcd]
    dims = np.array([p.get_axis_aligned_bounding_box().get_extent() for p in all_geoms])
    max_dims = dims.max(axis=0)
    dx = max_dims[0] * spacing
    # dy = max_dims[1] * spacing
    predict_pcd.translate((dx, 0, 0))
    o3d.visualization.draw_geometries(
        [test_pcd, predict_pcd], width=2000, height=1400, window_name=window_title)


def show_registered_pointclouds_grid(template, pointclouds, spacing=50.0):
    """
    Display multiple [registered] point clouds in a 2D grid layout using Open3D.

    Args:
        template: the PC template for a registered point cloud
        pointclouds: an array of pointclouds.
        spacing: float, how far apart point clouds are placed in the grid.
    """
    total = len(pointclouds)

    # Compute number of rows and columns (closest to square)
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)

    pcds = []

    for idx, pc in enumerate(pointclouds):
        # if pc.shape[1] != 3:
        #     raise ValueError(f"Point cloud {idx} is not [N, 3], got {pc.shape}")

        row = idx // cols
        col = idx % cols
        offset = np.array([col * spacing, -row * spacing, 0])

        template_copy = copy.deepcopy(template)
        pc.translate(offset)
        template_copy.translate(offset)
        pc.paint_uniform_color([1, 0.706, 0])
        template_copy.paint_uniform_color([0, 0.651, 0.929])
        pcds.append(pc)
        pcds.append(template_copy)
    o3d.visualization.draw_geometries(pcds)


def show_unregistered_and_registered_pc_triplet(
        template_pc, good_pc_unreg, bad_pc_unreg,
        good_pc_reg, bad_pc_reg, spacing=50.0, window_title=''):
    """
    Visualize the results of the registration for point clouds.

    Args:
        template_pc: a template (reference frame) Point Cloud
        good_pc_unreg, bad_pc_unreg: unregistered good and bad (anomalous) point clouds
        good_pc_reg, bad_pc_reg: registered good and bad (anomalous) point clouds
        spacing: interval between PCs in visualization
        window_title: a text for window title
    """

    def translate_paint_append(pcs, pc, translation, color=None):
        # For a point cloud, translate it with a translation,
        # paint it with a color, and add to the list pcs of the point clouds
        pc_copy = copy.deepcopy(pc)
        pc_copy.translate(translation)
        if color is not None:
            pc_copy.paint_uniform_color(color)
        pcs.append(pc_copy)

    pcds = []
    # First row: original (un-registered) PCs
    translate_paint_append(pcds, template_pc, np.array([0, 0, 0]), [0, 0.651, 0.929])
    translate_paint_append(pcds, good_pc_unreg, np.array([spacing, 0, 0]), [0, 1, 0])
    translate_paint_append(pcds, bad_pc_unreg, np.array([2*spacing, 0, 0]), [1, 0, 0])

    # Second row: registered PCs
    translate_paint_append(pcds, template_pc, np.array([0, -spacing, 0]), [0, 0.651, 0.929])
    translate_paint_append(pcds, good_pc_reg, np.array([spacing, -spacing, 0]), [0, 1, 0])
    translate_paint_append(pcds, bad_pc_reg, np.array([2*spacing, -spacing, 0]), [1, 0, 0])

    # Third row: un-registered+registered PCs in the same frame
    translate_paint_append(pcds, template_pc, np.array([spacing, -2*spacing, 0]), [0, 0.651, 0.929])
    translate_paint_append(pcds, good_pc_reg, np.array([spacing, -2*spacing, 0]), [0, 1, 0])
    translate_paint_append(pcds, template_pc, np.array([2*spacing, -2*spacing, 0]), [0, 0.651, 0.929])
    translate_paint_append(pcds, bad_pc_reg, np.array([2*spacing, -2*spacing, 0]), [1, 0, 0])
    o3d.visualization.draw_geometries(pcds, window_name=window_title)


def translate_paint_append(pcs, pc, translation, color=None):
    # For a point cloud, translate it with a translation,
    # paint it with a color, and add to the list pcs of the point clouds
    pc_copy = copy.deepcopy(pc)
    pc_copy.translate(translation)
    if color is not None:
        pc_copy.paint_uniform_color(color)
    pcs.append(pc_copy)

def show_patch_triplet(
        template_pc, good_pc, bad_pc,
        anchor_patch, good_patch, bad_patch,
        spacing=50.0, window_title=''):
    """
    Visualize the patch triplet for contrastive learning, compared to the corresponding point clouds.

    Args:
        template_pc, good_pc, bad_pc: a triplet of point clouds
        anchor_patch, good_patch, bad_patch: a triplet of patches extracted from these PCs
    """
    pcds = []
    # First row: oPCs
    if template_pc is not None:
        translate_paint_append(pcds, template_pc, np.array([0, 0, 0]), [0, 0, 0.5])
    if good_pc is not None:
        translate_paint_append(pcds, good_pc, np.array([spacing, 0, 0]), [0, 0.5, 0])
    if bad_pc is not None:
        translate_paint_append(pcds, bad_pc, np.array([2*spacing, 0, 0]), [0.5, 0, 0])
    translate_paint_append(pcds, anchor_patch, np.array([0, 0, 0]), [0, 0, 1])
    translate_paint_append(pcds, good_patch, np.array([spacing, 0, 0]), [0, 1, 0])
    translate_paint_append(pcds, bad_patch, np.array([2*spacing, 0, 0]), [1, 0, 0])

    # Second row: patches, drawn separately
    translate_paint_append(pcds, anchor_patch, np.array([0, -spacing, 0]), [0, 0, 1])
    translate_paint_append(pcds, good_patch, np.array([spacing, -spacing, 0]), [0, 1, 0])
    translate_paint_append(pcds, bad_patch, np.array([2*spacing, -spacing, 0]), [1, 0, 0])

    # Third row: patches, with anchor/good and anchor/bad overlapping
    translate_paint_append(pcds, anchor_patch, np.array([spacing, -2*spacing, 0]), [0, 0, 1])
    translate_paint_append(pcds, good_patch, np.array([spacing, -2*spacing, 0]), [0, 1, 0])
    translate_paint_append(pcds, anchor_patch, np.array([2*spacing, -2*spacing, 0]), [0, 0, 1])
    translate_paint_append(pcds, bad_patch, np.array([2*spacing, -2*spacing, 0]), [1, 0, 0])
    o3d.visualization.draw_geometries(pcds, window_name=window_title)


def show_registered_np_pointcloud(np_pc_target, np_pc_template, transform=None, window_name=''):
    # visualize registration, if desired
    target_pc = pc_utils.np2o3d(np_pc_target)
    template_pc = pc_utils.np2o3d(np_pc_template)
    if transform is not None:
        target_pc_registered = copy.deepcopy(target_pc)
        target_pc_registered.transform(transform)
    else:
        target_pc_registered = target_pc
    show_registered_pointcloud(template_pc, target_pc_registered, window_name=window_name)

def show_patch_pairs_grid(pairs, grid_cols=3, spacing=10.0, color1=[0, 1, 0], color2=[0, 0, 1], window_title=''):
    """
    Visualize up to len(pairs) patch pairs in a grid.
    Each pair is overlapped (same translation), colored differently.
    """
    pcds = []
    for k, (p1, p2) in enumerate(pairs):
        row, col = divmod(k, grid_cols)
        base = np.array([col * spacing, -row * spacing, 0.0])

        # Overlapped pair
        translate_paint_append(pcds, p1, base, color1)   # blue
        translate_paint_append(pcds, p2, base, color2)   # green

    o3d.visualization.draw_geometries(pcds, window_name=window_title)
