""" Constants for the COSSAD project for industrial anomaly detection in point clouds"""
from collections import defaultdict
import numpy as np

from enum import Enum


class TripletCentereing(Enum):
    TEMPLATE = 'template'
    COMMON = 'common'
    SEPARATE = 'separate'


MIN_DYNAMIC_PATCH_ID = 10000
MAX_DYNAMIC_PATCH_ID = 99999

# label values for the good and anomalous points, as well as for the whole PCs
GOOD_MASK = 0
ANOMALY_MASK = 1

# The textual description of non-anomalous object
NO_ANOMALY = 'good'

DATASET_SHAPENET = 'ashapenet'
DATASET_MULSEN = 'mulsen'
DATASET_REAL3DAD = 'real3dad'

ALL_CLASSES_NAME = 'all_classes'
REAL3D_ALL_CLASSES_NAME = 'real3d_all_classes'

# Classes of the objects in Real3D dataset (12 total).
real3d_object_classes = sorted([
    'airplane', 'candybar', 'car', 'chicken', 'diamond', 'duck',
    'fish', 'gemstone', 'seahorse', 'shell', 'starfish', 'toffees'
])

MULSEN_ALL_CLASSES_NAME = 'mulsen_all_classes'

# Classes of the objects in MulSen dataset (11 total).
mulsen_object_classes = sorted([
    'button_cell', 'capsule', 'cube', 'flat_pad', 'light', 'nut',
    'piggy', 'plastic_cylinder', 'screen', 'solar_panel', 'toothbrush'
])

mulsen_anomaly_free_types = ['good', 'color']

SHAPENET_ALL_CLASSES_NAME = 'shapenet_all_classes'

# Classes of the objects in Anomaly-ShapeNet dataset (52 total).
shapenet_object_classes = sorted([
    'ashtray0', 'bag0', 'bottle0', 'bottle1', 'bottle3', 'bowl0', 'bowl1',
    'bowl2', 'bowl3', 'bowl4', 'bowl5', 'bucket0', 'bucket1', 'cabinet0',
    'cap0', 'cap1', 'cap2', 'cap3', 'cap4', 'cap5', 'chair0', 'cup0', 'cup1',
    'cup2', 'desk0', 'eraser0', 'headset0', 'headset1', 'helmet0', 'helmet1',
    'helmet2', 'helmet3', 'jar0', 'knife0', 'knife1', 'microphone0',
    'microphone1', 'screen0', 'shelf0', 'tap0', 'tap1', 'vase0', 'vase1',
    'vase10', 'vase2', 'vase3', 'vase4', 'vase5', 'vase6', 'vase7', 'vase8', 'vase9'
])

SHAPENET_REAL3D_ALL_CLASSES_NAME = 'shapenet_real3d_object_classes'
shapenet_real3d_object_classes = sorted(shapenet_object_classes+real3d_object_classes)

# Locations per class: allows up to 100k locations per object
# Each location uses 2 IDs (good=even, bad=odd)
LOCATIONS_PER_CLASS = 100_000

# Dataset registry with base IDs
# Base IDs spaced to allow LOCATIONS_PER_CLASS * 2 IDs per object
DATASET_REGISTRY = {
    DATASET_REAL3DAD: {'classes': real3d_object_classes, 'base_id': 0},
    DATASET_MULSEN: {'classes': mulsen_object_classes, 'base_id': 10_000_000},
    DATASET_SHAPENET: {'classes': shapenet_object_classes, 'base_id': 20_000_000},
}

def get_class_id(object_class: str):
    """Get numeric ID for an object class.

    Args:
        object_class: Name of the object class

    Returns:
        Base numeric ID for the class (starting point for patch IDs)

    Raises:
        ValueError: If class not found in any dataset
    """
    for dataset_name, dataset_info in DATASET_REGISTRY.items():
        classes = dataset_info['classes']
        if object_class in classes:
            base_id = dataset_info['base_id']
            class_index = classes.index(object_class)
            return base_id + class_index * LOCATIONS_PER_CLASS * 2
    raise ValueError(f"Class '{object_class}' not found in any dataset")

def dataset_for_class(class_name: str) -> str:
    """Given a class name, returns the name of the dataset. Assumes that the class names are unique"""
    for dataset_name, dataset_info in DATASET_REGISTRY.items():
        if class_name in dataset_info['classes']:
            return dataset_name
    raise ValueError(f'Unknown dataset for class {class_name}')

# If object -> template mean distance is over this threshold,  registration is unsuccessful
registration_mean_distances_thresholds = defaultdict(
    lambda: np.finfo(np.float32).max,
    {
        # Real3dAD classes
        'shell': 0.1,
        # 'starfish': 0.2,
        'starfish': 0.15,  # was 0.12
        # 'airplane': 0.5,
        'airplane': 0.35,
        # 'car': 1.0,
        'car': 0.15,
        'candybar': 0.12,
        'chicken': 0.4,
        # 'chicken': 0.30,
        'diamond': 0.2,
        'duck': 0.2,
        # 'fish': 0.1,
        'fish': 0.2,
        # 'gemstone': 0.15,
        'gemstone': 0.25,
        'seahorse': 0.15,  # was 0.1
        'toffees': 0.15,  # was 0.1

        # MulSen classes
        'light': 0.16,
        'nut': 0.12,
        'piggy': 0.24,
        # Poor registration quality overall
        'solar_panel': 0.25,
        'capsule': 0.1,

        # Anomaly-ShapeNet classes
        'ashtray0': 0.09,
        'bag0': 0.09,
        # 'bucket0': 0.25,
        'bucket0': 0.15,  # Lots of poor regs for "broken"
        'bottle1': 0.15,
        'bottle3': 0.15,
        'bowl2': 0.15,
        'cap0': 0.1,
        'cap1': 0.08,
        'cap2': 0.08,
        'cap3': 0.08,
        'cap4': 0.08,
        'cap5': 0.08,
        'chair0': 0.08,
        'cup0': 0.08,
        'cup1': 0.1,
        'cup2': 0.08,

        'desk0': 0.08,
        'eraser0': 0.08,
        'headset0': 0.08,
        'headset1': 0.08,
        'helmet0': 0.08,
        'helmet1': 0.08,
        'helmet2': 0.08,
        'helmet3': 0.08,

        'jar0': 0.08,
        'knife0': 0.075,
        'knife1': 0.075,

        'microphone0': 0.15,
        'microphone1': 0.1,
        'screen0': 0.1,  # Could be even lower, e.g. 0.08
        'shelf0': 0.1,  # Could be even lower, e.g. 0.08

        'tap0': 0.1,
        'tap1': 0.1,

        'vase0': 0.075,
        'vase1': 0.075,
        'vase2': 0.075,
        'vase3': 0.075,
        'vase4': 0.075,
        'vase5': 0.075,
        'vase6': 0.075,
        'vase7': 0.075,
        'vase8': 0.075,
        'vase10': 0.17,
        'vase9': 0.1,  # maybe could be around 0.075... but not bad even now
    }
)


class PatchSource(Enum):
    TEMPLATE = 'template'
    GOOD = 'good'
    ANOMALOUS = 'anomalous'
