import os
import open3d as o3d
import sys
import numpy as np
import random

import torch


def in_debugger():
    return sys.gettrace() is not None


def np_point_cloud_stats(pcd_np, name):
    """
    Prints diagnostic info about a point cloud:
    - Number of points
    - Axis-aligned bounding box size (max dimensions)
    - Diagonal length of bounding box
    """
    if pcd_np.shape[0] == 0:
        print(f"[{name}] Empty point cloud.")
        return

    min_bounds = pcd_np.min(axis=0)
    max_bounds = pcd_np.max(axis=0)
    dims = max_bounds - min_bounds
    diagonal = np.linalg.norm(dims)

    print(f"Point cloud [{name}] Info:")
    print(f"  Num. points     : {pcd_np.shape[0]}")
    print(f"  X range         : {min_bounds[0]:.4f} to {max_bounds[0]:.4f}  (delta = {dims[0]:.4f})")
    print(f"  Y range         : {min_bounds[1]:.4f} to {max_bounds[1]:.4f}  (delta = {dims[1]:.4f})")
    print(f"  Z range         : {min_bounds[2]:.4f} to {max_bounds[2]:.4f}  (delta = {dims[2]:.4f})")
    print(f"  Bounding box diagonal: {diagonal:.4f}")



def save_registered_pointclouds(base_path, subfolder, split, idx, basic_template, registered_np):
    """ Save a pair of registertered point clouds to the disk
    base_path: path to the output folder
    subfolder: the name of the object
    split: train vs. test

    """
    # Construct full directory path
    output_dir = os.path.join(base_path, subfolder)
    os.makedirs(output_dir, exist_ok=True)  # Create directories if they don't exist

    # Define file names using idx
    template_path = os.path.join(output_dir, f"template.pcd")
    registered_path = os.path.join(output_dir, f"registered_{split}_{idx}.pcd")

    # Save point clouds
    # if not os.path.exists(template_path):
    if not os.path.exists(template_path):
        basic_template_pc = o3d.geometry.PointCloud()
        basic_template_pc.points = o3d.utility.Vector3dVector(basic_template)
        o3d.io.write_point_cloud(template_path, basic_template_pc)

    registered_pc = o3d.geometry.PointCloud()
    registered_pc.points = o3d.utility.Vector3dVector(registered_np)
    o3d.io.write_point_cloud(registered_path, registered_pc)


def assert_nans_nparray(str_prefix, np_arr, max_percent_nan):
    total_nans = np.isnan(np_arr).sum()
    nan_perc = (total_nans / np_arr.size())*100
    assert nan_perc <= max_percent_nan, \
    f'{str_prefix} {total_nans} out of {np_arr.size} ({nan_perc:.2f}%) entries are NaN'

def assert_nans_tensor(str_prefix, tensor, max_percent_nan):
    total_nans = np.isnan(tensor).sum().item()
    nan_perc = (total_nans / tensor.numel())*100
    assert nan_perc <= max_percent_nan, \
    f'{str_prefix} {total_nans} out of {tensor.numel()} ({nan_perc:.2f}%) entries are NaN'

def set_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Only if available:
    if hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(seed)

def is_emb_tensor_normalized(emb: torch.Tensor) -> bool:
    emb_norm = emb.norm(dim=1)
    return torch.allclose(emb_norm, torch.ones_like(emb_norm), atol=1e-2)

def require(condition: bool, message: str='') -> None:
    """An equivalent of the runtime assertion that cannot be disabled"""
    if not condition:
        raise AssertionError(message)