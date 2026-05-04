import logging
from pathlib import Path

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import copy

from sklearn.model_selection import train_test_split

import constants

# Alexta: these are auto-generated k-fold validation splits with their string tags in a form {dataset}_{train|test}_{i}.
# we expect this will pe periodically re-generated
# import class_splits_2025_08_18 as class_splits
import class_splits_2025_11_13 as class_splits

logger = logging.getLogger(__name__)


def drop_rows_over_threshold(df, column_name, threshold, logger):
    """Drops rows from the dataset whose values in the column exceed a given threshold"""
    initial_len = len(df)
    df = df[df[column_name] <= threshold]  # keep only rows <= threshold
    num_dropped_rows = initial_len - len(df)
    if num_dropped_rows > 0:
        logger.warning(f'Dropped {num_dropped_rows} rows whose {column_name} values exceed {threshold}')
    return df

def string_to_matrix(matrix_str):
    arr = np.array([float(x) for x in matrix_str.split(',')])
    return arr.reshape(4, 4)


def matrix_to_string(matrix):
    """Convert 4x4 matrix to comma-separated string"""
    return ','.join(map(str, matrix.flatten()))


def normalize_pc(point_cloud):
    """Centers the point cloud at 0,0.
    Works for actual PCs or for a PC where 4th column is an anomaly label
    """
    center = np.average(point_cloud[:, :3], axis=0)
    point_cloud[:, :3] -= center
    return point_cloud


def np_sorted_rows(X):
    """Sort numpy array rows"""
    return X[np.lexsort(X.T[::-1])]


def np_sorted_idx(np_cloud):
    """Sort the numpy point cloud and return the sorting index.
    Note: please use this everywhere you sort point clouds for consistency
    """
    return np.lexsort((np_cloud[:, 2], np_cloud[:, 1], np_cloud[:, 0]))


def scale_pc(point_cloud, scale):
    """Scale the point cloud, in the form of the numpy array, by a given factor"""
    return point_cloud * scale


def mask_defects_in_pc(np_pcd_src, np_pcd_anomaly,
                       good_mask_value=constants.GOOD_MASK, anomaly_mask_value=constants.ANOMALY_MASK,
                       max_radius=None):
    """ Create a defect mask for points in the np_pcd_src.
    If a point in np_pcd_src is within the radius r from any anomalous point in np_pcd_anomaly, then its mask is set
     to anomaly_mask_value. Otherwise its mask is set to good_mask_value.
    This is used if the np_pcd_anomaly is the "anomaly mask" point cloud, and the np_pcd_dst is created from the mesh.
    max_radius: a radius around teh anomalous point. If max_radius is not specified, we compute it as a maximum distance
    between the adjacent points in the np_pcd_anomaly PC.
    """
    # Fill mask with defect-free value
    mask = np.full(shape=np_pcd_src.shape[0], fill_value=good_mask_value, dtype=np.int32)
    if max_radius is None:
        # Deteremine the maximum distance between adjacent points in np_pcd_anomaly.
        # BUild a KD tree for np_pcd_anomaly and query it against the self.
        tree_dist = cKDTree(np_pcd_anomaly)
        # Get two closest points. The first one will be self, the second - the closest one.
        matches_distances, _ = tree_dist.query(np_pcd_anomaly, k=2)
        distances = [item[1] for item in matches_distances if item[1] != np.Inf]
        max_dist = np.max(distances)
        max_radius = max_dist * 1.2

    # Get the median distance between the points in the "defective" PC.
    # median_r_defects = np.median(np.linalg.norm(np_pcd_anomaly[0,:] - np_pcd_anomaly[0,:].T))
    # max_radius = median_r_defects * 1.5

    tree = cKDTree(np_pcd_src)
    # Query all the points in the PC that are within a radius (plus some delta) of the anomalous point
    # TODO(alexta): this is a hack. It is possible that we will add extra points as anomalous (i.e. on boundaries),
    # and also leave some "gaps" in the anomalous region if the np_pcd_anomaly is very non-uniform
    src_matches_indices = tree.query_ball_point(np_pcd_anomaly, r=max_radius, eps=0)
    # We expect to have at least one match in the PC for each anomaly point
    num_matched_points_in_src = [len(indices) for indices in src_matches_indices]
    if min(num_matched_points_in_src) == 0:
        points_zero_neighbors = [idx for idx in range(len(num_matched_points_in_src)) if
                                 num_matched_points_in_src[idx] == 0]
        logging.warning(f'Creating an anomaly mask for the PC: {len(points_zero_neighbors)} out '
                        f'of {np_pcd_anomaly.shape[0]} anomalous points have no correspondence to the main PC')
    # Now set the mask values for all these selected points as anomalous points
    all_matched_indices = [item for sublist in src_matches_indices for item in sublist]
    all_matched_indices = list(set(all_matched_indices))
    all_matched_indices = np.array(all_matched_indices)
    mask[all_matched_indices] = anomaly_mask_value
    return mask


def read_point_cloud(pcd_file):
    """Reads a point cloud from a pcd file and sort it.
    Returns point cloud as Open3D PC and as Numpy array

    Use this function whenever possible, as it maintains the sorted invariant for PC"""
    pcd_cloud = o3d.io.read_point_cloud(pcd_file)
    np_cloud = np.array(pcd_cloud.points)
    return pcd_cloud, np_cloud

    # Alexta: here we attempted to maintain a simple invariant:  PC is always sorted
    # IT DOES NOT WORK! It seems to introduce strong inconsistenceis into results from Real3D

    # Sort Numpy PC
    # sort_idx = np_sorted_idx(np_cloud)
    # np_cloud_sorted = np_cloud[sort_idx]
    #
    # # If point cloud has colors/normals, sort them the same way
    # if pcd_cloud.has_colors():
    #     colors = np.asarray(pcd_cloud.colors)[sort_idx]
    # if pcd_cloud.has_normals():
    #     normals = np.asarray(pcd_cloud.normals)[sort_idx]
    #
    # # Create new, sorted point cloud
    # pcd_cloud_sorted = o3d.geometry.PointCloud()
    # pcd_cloud_sorted.points = o3d.utility.Vector3dVector(np_cloud_sorted)
    # if pcd_cloud.has_colors():
    #     pcd_cloud_sorted.colors = o3d.utility.Vector3dVector(colors)
    # if pcd_cloud.has_normals():
    #     pcd_cloud_sorted.normals = o3d.utility.Vector3dVector(normals)
    # return pcd_cloud_sorted, np_cloud_sorted


def read_mesh(mesh_file, num_points):
    """Reads a mesh from the file and converts it into the point cloud with the given number of points."""
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    pcd_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
    np_cloud = np.array(pcd_cloud.points)
    return pcd_cloud, np_cloud


def dloader_dict_collate(batch_in):
    """Collate function for all the dataloaders used by COSSAD"""
    if batch_in is None or batch_in[0] is None:
        return {'np_pointcloud': None,
                'point_mask': None,
                'object_label': None,
                'input_file': None,
                'anomaly_mask_file': None,
                'anomaly_type': None,
                'scale': None,
                'points_removed': None
                }
    return {
        key: [d[key] for d in batch_in if d is not None] for key in batch_in[0]
    }

def get_object_classes_names(object_classes_arg):
    """Get the set of object classes from the command line comma-separated list"""
    ret_object_classes = []
    object_classes_arg_list = object_classes_arg.split(',')
    if object_classes_arg in class_splits.OBJECT_CLASS_SPLITS.keys():
        # This is the name of pre-generated split
        return sorted(class_splits.OBJECT_CLASS_SPLITS[object_classes_arg])
    for object_class_arg in object_classes_arg_list:
        if object_class_arg == constants.ALL_CLASSES_NAME:
            ret_object_classes.extend(constants.real3d_object_classes)
            ret_object_classes.extend(constants.mulsen_object_classes)
            ret_object_classes.extend(constants.shapenet_object_classes)
            return ret_object_classes
        if object_class_arg == constants.REAL3D_ALL_CLASSES_NAME:
            ret_object_classes.extend(constants.real3d_object_classes)
        elif object_class_arg == constants.MULSEN_ALL_CLASSES_NAME:
            ret_object_classes.extend(constants.mulsen_object_classes)
        elif object_class_arg == constants.SHAPENET_ALL_CLASSES_NAME:
            ret_object_classes.extend(constants.shapenet_object_classes)
        else:
            ret_object_classes.append(object_class_arg)
    return sorted(list(set(ret_object_classes)))


def prepare_splits(dataset_dir, train_fraction, test_fraction, val_fraction, allowed_objects=None):
    """
    Split all the patches into training, validation and test sets for training the contrastive encoder.

    Args:
        train_fraction: a fraction of data to go to teh
    """
    assert (np.isclose(train_fraction + test_fraction + val_fraction, 1, rtol=1e-5, atol=1e-3))
    logger.info(f'Doing datasplits on a dir {dataset_dir}')
    data_files_gen = Path(dataset_dir).glob("*.npz")
    data_files = []
    if allowed_objects is not None:
        data_files = [str(file) for file in data_files_gen if file.name.split("_")[1] in allowed_objects]

    if len(data_files) == 0:
        raise ValueError(f'Failed to load dataset from {dataset_dir}. Exiting.')

    # Split files (not individual point clouds)
    if test_fraction != 0:
        train_files, temp_files = train_test_split(data_files, test_size=1 - train_fraction)
        val_files, test_files = train_test_split(temp_files, test_size=test_fraction / (test_fraction + val_fraction))
    else:
        train_files, val_files = train_test_split(data_files, test_size=1 - train_fraction)
        test_files = []
    return train_files, test_files, val_files

