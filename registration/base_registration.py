"""
A base class for all the registration functionality in COSSAD.

Allows a drop-in replacement of different registration algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
from . import util_registration as util_reg
import logging
import utils.pc_utils as pc_utils

logger = logging.getLogger(__name__)

class RegistrationInterface(ABC):
    @abstractmethod
    def compute_registration_transform(self) -> bool:
        """Computes (i.e. learns) the registration to transform the target PC into template frame.
        Returns True if the registration successful (i.e. mean difference less than a threshold); False otherwise.
        """
        pass

    @abstractmethod
    def get_registered_target_pc(self) -> np.ndarray:
        """Returns a regstered point cloud as a Numnpy array."""
        pass

    @abstractmethod
    def get_accuracy(self) -> util_reg.RegistrationAccuracy:
        pass

    @abstractmethod
    def get_transform(self) -> np.ndarray:
        """Obtains the learned transformation that transforms target PC into the template frame"""
        pass


class RegistrationBase(RegistrationInterface):
    """Base registration class"""

    def __init__(self, np_pc_template=None, np_pc_target=None):
        """Initializes the registration.

        Args:
        np_pc_template: a template PC, representing a reference frame
        np_pc_target:  a target PC to be registered against the template
        """
        self._np_pc_template = None
        self._np_pc_target = None
        self._np_pc_target_registered = None
        self._accuracy = None
        self.min_deviation_from_zero = 1e-4
        self.set_template(np_pc_template)
        self.set_target(np_pc_target)

    def set_template(self, np_pc_template):
        assert np_pc_template is not None
        """Set a template PC that represents a reference frame"""
        assert np.allclose(np.mean(np_pc_template, axis=0), [0.0, 0.0, 0.0], atol=self.min_deviation_from_zero), \
            'Template PC is not normalized!'
        self._np_pc_template = np_pc_template

    def set_target(self, np_pc_target):
        assert np_pc_target is not None
        """Set a target PC that must be registered against a reference frame"""
        assert np.allclose(np.mean(np_pc_target, axis=0), [0.0, 0.0, 0.0], atol=self.min_deviation_from_zero), \
            'Target PC is not normalized!'
        self._np_pc_target = np_pc_target

    def get_template(self) -> np.ndarray:
        """Returns a template PC that represents a reference frame"""
        return self._np_pc_template

    def get_target(self) -> np.ndarray:
        """Returns a target PC that must be registered against a reference frame"""
        return self._np_pc_target

    def compute_registration_transform(self) -> bool:
        """Computes (i.e. learns) the registration to transform the target PC into template frame."""
        raise NotImplementedError

    def get_registered_target_pc(self):
        return self._np_pc_target_registered

    def get_accuracy(self):
        return self._accuracy

    def get_transform(self):
        """Obtains the learned transformation that transforms target PC into the template frame"""
        raise NotImplementedError

    # def validate_registration(self, threshold: float|None = None) -> bool:
    @staticmethod
    def validate_registration(np_pc_template: np.ndarray, np_pc_target: np.ndarray, transform: Any, threshold: float) -> Tuple[bool, float, Any]:
        """Self-check if a complex registration was successful. """
        o3d_pcd_target = pc_utils.np2o3d(np_pc_target)
        o3d_pcd_target.transform(transform)
        mad_after = pc_utils.mean_average_distance(pc_utils.o3d2np(o3d_pcd_target), np_pc_template)
        if mad_after >= threshold:
            logger.error(f'Registration internal error: mean_average_distance = {mad_after}, '
                         f'higher than a threshold {threshold}')
            return False, mad_after, o3d_pcd_target
        return True, mad_after, o3d_pcd_target


class RegistrationFactory(ABC):
    @abstractmethod
    def __call__(self) -> RegistrationBase:
        pass

