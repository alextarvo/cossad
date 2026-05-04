""" Legacy Readers for the Real3D dataset.

They can drop-in replace the "regular" DatasetCossad dataloaders and were used for debugging/investigation
of COSSAD performance issues.
"""

import pathlib

from torch.utils.data import Dataset
import glob
import os
import open3d as o3d
import numpy as np
import logging
import pandas as pd
import constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


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


def real3d_dict_collate(batch_in):
    if batch_in is None or batch_in[0] is None:
        return {'np_pointcloud': None,
                'point_mask': None,
                'object_label': None,
                'input_file': None}
    return {
        key: [d[key] for d in batch_in if d is not None] for key in batch_in[0]
    }


class Dataset3DADLegacyTrain(Dataset):
    """Loads a 3D point clouds from Real3D-AD training set.
    This set is fairly small, and typically contains 4 objects per class. All labels are "good", and
    """

    def __init__(self, dataset_dir, class_name, normalize=True):
        """
        :param dataset_dir: the path to the Real3D-AD dataset
        :param class_name: the name of the class (plane, car, ....  - see constants.py)
        """
        self.train_files_list = glob.glob(str(os.path.join(dataset_dir, class_name, 'train')) + '/*template*.pcd')
        self.normalize = normalize

    def __getitem__(self, idx):
        """Returns a tuple consisting of:
        - unordered point cloud as Numpy array
        - mask for each point in the pointcloud (0: good, 1: anomalous)
        - label for the whole object: 0: good; 1: anomalous
        - name of the file PC was loaded from
        """
        # pcd = o3d.io.read_point_cloud(self.train_files_list[idx])
        # np_pointcloud = np.array(pcd.points)
        pcd, np_pointcloud = read_point_cloud(self.train_files_list[idx])
        if self.normalize:
            np_pointcloud = normalize_pc(np_pointcloud)
        # This is a per-pixel mask of "correct / anomalous" for PC patches
        mask = np.zeros((np_pointcloud.shape[0]))
        # Global label for this object pointcloud: good (0), anomalous (1)
        return {'np_pointcloud': np_pointcloud,
                'point_mask': mask,
                'object_label': constants.GOOD_MASK,
                'input_file': self.train_files_list[idx]
                }

    def __len__(self):
        return len(self.train_files_list)


def load_text_verify_against_pcd(np_pointcloud, pcd_file_name, txt_file_name):
    """Load the pointcloud with labels from the .txt file and verify the contains against the .pcd file

    Returns pointcloud in the Numpy array format. First 3 dimensions are coordinates, last dim is a label
    """
    pcd_text = np.genfromtxt(txt_file_name, delimiter=" ")
    logging.info(f'Loaded file {txt_file_name}; dimensions: {pcd_text.shape}')
    if pcd_text.shape[1] != 4:
        logging.warning(f'Contents of {pcd_file_name} have wrong shape {pcd_text.shape}')
        return None
    pointcloud_text = pcd_text[:, :3]
    if pointcloud_text.shape != np_pointcloud.shape:
        logging.warning(f'File {pcd_file_name}; the shape {pointcloud_text.shape} of the text '
                        f'point cloud do not match the shape {np_pointcloud.shape} of the PCD point cloud')
        return None
    # if not np.allclose(np_sorted_rows(np_pointcloud), np_sorted_rows(pointcloud_text), rtol=1e-2):
    if not np.allclose(np_pointcloud[np_sorted_idx(np_pointcloud)], pointcloud_text[np_sorted_idx(pointcloud_text[:,:3])], rtol=1e-2):
        logging.warning(f'Contents of {pcd_file_name} do not match contents of {txt_file_name}')
        return None

    # # Sort the text-based point cloud and make sure its very close to _already sorted_ NP point cloud
    # txt_sorted_idx = np_sorted_idx(pointcloud_text)
    # pointcloud_text = pointcloud_text[txt_sorted_idx]
    # pcd_text = pcd_text[txt_sorted_idx]
    # if not np.allclose(np_pointcloud, pointcloud_text, rtol=1e-2):
    #     logging.warning(f'Contents of {pcd_file_name} do not match contents of {txt_file_name}')
    #     return None
    return pcd_text


class Dataset3DADLegacyTest(Dataset):
    def __init__(self, dataset_dir, class_name, normalize=True, full_pc=True):
        """
        :param dataset_dir: the path to the Real3D-AD dataset
        :param class_name: the name of the class (plane, car, ....  - see constants.py)

        """
        self.class_name = class_name
        self.dataset_dir = dataset_dir
        all_files_names = glob.glob(str(os.path.join(dataset_dir, class_name, 'test')) + '/*.pcd')
        self.test_files_names = [os.path.splitext(os.path.basename(f))[0] for f in all_files_names]
        self.test_files_names.sort()
        if full_pc:
            self.test_files_names = [os.path.splitext(os.path.basename(f))[0]
                                    for f in all_files_names if not 'cut' in f]
        # else:
        #     self.test_files_names = [os.path.splitext(os.path.basename(f))[0]
        #                             for f in all_files_names if 'cut' in f]
        self.normalize = normalize

    def __getitem__(self, idx):
        file_name = self.test_files_names[idx]
        pcd_file_name = str(os.path.join(self.dataset_dir, self.class_name, 'test', file_name)) + '.pcd'
        txt_file_name = str(os.path.join(self.dataset_dir, self.class_name, 'gt', file_name)) + '.txt'
        # pcd = o3d.io.read_point_cloud(pcd_file_name)
        # np_pointcloud = np.array(pcd.points)
        pcd, np_pointcloud = read_point_cloud(pcd_file_name)
        logging.info(f'Loaded file {pcd_file_name}; dimensions: {np_pointcloud.shape}')
        if 'good' in file_name:
            mask = np.zeros((np_pointcloud.shape[0]))
            label = constants.GOOD_MASK
        else:
            pcd_text = load_text_verify_against_pcd(np_pointcloud, pcd_file_name, txt_file_name)
            if pcd_text is None:
                logging.warning(f'Failed to load text label for item {idx}')
                return None
            logging.info(f'Loaded file {txt_file_name}; dimensions: {pcd_text.shape}')
            mask = pcd_text[:, 3]
            label = constants.ANOMALY_MASK

        if self.normalize:
            np_pointcloud = normalize_pc(np_pointcloud)
        assert (np_pointcloud is not None)
        return {'np_pointcloud': np_pointcloud,
                'point_mask': mask,
                'object_label': label,
                'input_file': pcd_file_name}

    def __len__(self):
        return len(self.test_files_names)


def collate_fn_pc_list(batch):
    """
    Collate function for batching point clouds.
    Each item in the batch is an open3d.geometry.PointCloud.
    """
    return batch  # just returns the list of PointClouds

