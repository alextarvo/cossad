"""
A wrapper class over the base registration class that allows multiple registration attempts.

"""

import open3d as o3d
import numpy as np

from . import util_registration as util_reg
from . import base_registration as base_reg

import copy
import logging

logger = logging.getLogger(__name__)


class MultipleAttemptRegistrationBase(base_reg.RegistrationInterface):
    """Base registration class"""

    def __init__(self, alg_factory: base_reg.RegistrationFactory, target_accuracy: float, max_attempts: float=5):
        """Initializes the registration.

        Args:
        np_pc_template: a template PC, representing a reference frame
        np_pc_target:  a target PC to be registered against the template
        """
        self.alg_factory = alg_factory
        self.target_accuracy = target_accuracy
        self.best_registered_pc = None
        self.best_reg_accuracy = None
        self.best_transform = None
        self.max_attempts = max_attempts

    def compute_registration_transform(self):
        """Computes (i.e. learns) the registration to transform the target PC into template frame."""
        raise NotImplementedError

    def get_registered_target_pc(self):
        return self.best_registered_pc

    def get_accuracy(self):
        return self.best_reg_accuracy

    def get_transform(self):
        """Obtains the learned transformation that transforms target PC into the template frame"""
        return self.best_transform


class MultipleAttemptRegistration(MultipleAttemptRegistrationBase):
    def __init__(self, alg_factory: base_reg.RegistrationFactory,
                 target_accuracy: float, max_attempts: float=5):
        super(MultipleAttemptRegistration, self).__init__(alg_factory, target_accuracy, max_attempts)

    def pre_reg(self):
        pass

    def post_reg(self):
        pass

    def pre_attempt(self, reg):
        pass

    def post_attempt(self, reg):
        pass

    def save_best_values(self, reg):
        self.best_reg_accuracy = reg.get_accuracy()
        self.best_registered_pc = copy.deepcopy(reg.get_registered_target_pc())
        self.best_transform = reg.get_transform()

    def compute_registration_transform(self) -> bool:
        for i in range(self.max_attempts):
            reg = self.alg_factory()
            reg.compute_registration_transform()
            if self.best_reg_accuracy is None:
                self.save_best_values(reg)
            elif reg.get_accuracy().mean_dist_src_to_target < self.best_reg_accuracy.mean_dist_src_to_target:
                self.save_best_values(reg)
        if self.best_reg_accuracy.mean_dist_src_to_target > self.target_accuracy:
            logging.warning(
                f'Overly large distance between source and target point clouds {self.best_reg_accuracy.mean_dist_src_to_target}!')
            logging.warning(f'Likely wrong registration, skipping')
            return False
        return True
        # logging.debug(f'Registered target PC {np_target.shape} against template {np_template.shape}.')
        # logging.debug(f'Best accuracy: {self.best_reg_accuracy}; resulting PC {best_reg_pc.shape}')


class MultipleAttemptRegistrationEugenRotation(MultipleAttemptRegistration):
    def __init__(self, alg_factory: base_reg.RegistrationFactory, target_accuracy: float, max_attempts: float=8):
        super(MultipleAttemptRegistrationEugenRotation, self).__init__(alg_factory, target_accuracy, max_attempts)
        Rx_pi = np.diag([1, -1, -1])
        Ry_pi = np.diag([-1, 1, -1])
        Rz_pi = np.diag([-1, -1, 1])
        Rxy_pi = Rx_pi @ Ry_pi
        Rxz_pi = Rx_pi @ Rz_pi
        Ryz_pi = Ry_pi @ Rz_pi
        Rxyz_pi = Rx_pi @ Ry_pi @ Rz_pi
        self.rotations = [np.eye(3), Rx_pi, Ry_pi, Rz_pi, Rxy_pi, Rxz_pi, Ryz_pi, Rxyz_pi]
        # Limit the number of attempts to the number requested
        self.rotations  = self.rotations[:self.max_attempts]
        self.best_rotation = None

    def pca_axes(self, np_pc_centered: np.ndarray) -> np.ndarray:
        # points_centered: (N, 3), already zero-mean
        C = (np_pc_centered.T @ np_pc_centered) / max(1, len(np_pc_centered))
        evals, evecs = np.linalg.eigh(C)  # ascending eigenvalues
        V = evecs[:, np.argsort(evals)[::-1]]  # columns: v1, v2, v3 (lambda1 >= lambda2 >= lambda3)
        if np.linalg.det(V) < 0:  # enforce right-handed basis
            V[:, -1] *= -1.0
        return V

    def compute_registration_transform(self) -> bool:
        for rotation in self.rotations:
            reg = self.alg_factory()
            assert np.allclose(reg.get_target().mean(axis=0), 0, atol=1e-5)
            V = self.pca_axes(reg.get_target())
            full_rotation = V @ rotation @ V.T
            reg.set_target(reg.get_target() @ full_rotation)
            reg.compute_registration_transform()
            if self.best_reg_accuracy is None:
                self.save_best_values(reg)
                self.best_rotation = full_rotation
            elif reg.get_accuracy().mean_dist_src_to_target < self.best_reg_accuracy.mean_dist_src_to_target:
                self.save_best_values(reg)
                self.best_rotation = full_rotation
        if self.best_reg_accuracy.mean_dist_src_to_target > self.target_accuracy:
            logging.warning(
                f'Overly large distance between source and target point clouds {self.best_reg_accuracy.mean_dist_src_to_target}!')
            return False
        return True

    def get_transform(self):
        """Obtains the learned transformation that transforms target PC into the template frame"""
        homo_transform = np.eye(4)
        homo_transform[:3, :3] = self.best_rotation
        # final_transform = homo_transform @ self.best_transform
        final_transform = self.best_transform @ homo_transform
        return final_transform
