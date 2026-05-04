import numpy as np


class RandomPerturbation:
    """Random rigid perturbation with controlled rotation and translation magnitudes.

    Rotation magnitude is the geodesic distance on SO(3) (the angle of axis-angle representation).
    Translation magnitude is the Euclidean length of the translation vector, expressed as a percent
    of the perturbed object's radius (max distance from its centroid; computed inside perturb()).
    Direction of both is sampled uniformly on the unit sphere, so the perturbation
    has the requested *magnitude* but a random *direction*.

    Rotation is applied around the origin. To rotate around an object's centroid,
    pre-center the point cloud before calling perturb() and re-add the centroid afterwards.
    """

    def __init__(self,
                 rotation_deg: float,
                 translation_percent: float,
                 rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()

        # Random rotation axis
        rotation_axis = self._sample_unit_vector(rng)
        rotation_angle_rad = np.deg2rad(rotation_deg)
        R = self._axis_angle_to_matrix(rotation_axis, rotation_angle_rad)

        # Translation direction is fixed at construction. Magnitude depends on the perturbed
        # object's radius and is materialized into T[:3, 3] per-call by perturb().
        self.translation_percent = translation_percent
        self.translation_direction = self._sample_unit_vector(rng)

        # Transform matrix for homogenous input. Rotation block is final;
        # translation column starts as zero and is overwritten by perturb().
        self.transform = np.eye(4)
        self.transform[:3, :3] = R

        self.rotation_axis = rotation_axis
        self.rotation_angle_rad = rotation_angle_rad

    @staticmethod
    def _sample_unit_vector(rng: np.random.Generator) -> np.ndarray:
        """
        Sample a unit vector having a random direction in the unit sphere.
        Müller, M. E. (1959). "A note on a method for generating points uniformly on n-dimensional spheres.
        """
        v = rng.standard_normal(3)
        return v / np.linalg.norm(v)

    @staticmethod
    def _axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        """Create the rotation matrix according to the Rodrigues rotation formula.
         Args:
            axis: a rotation axis
            angle_rad: a rotation angle, in radians
         """
        ax, ay, az = axis
        # This is an expansion of the cross product kxv into a matrix multiplication form K@v,
        # where v is the point in the point cloud to be rotated
        K = np.array([[0.0, -az, ay],
                      [az, 0.0, -ax],
                      [-ay, ax, 0.0]])
        # Rodrigues rotation formula in the matrix form
        return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)

    @staticmethod
    def _object_radius(pc: np.ndarray) -> float:
        """Max distance from the point cloud's centroid — proxy for object scale."""
        centroid = pc.mean(axis=0)
        return float(np.linalg.norm(pc - centroid, axis=1).max())

    def perturb(self, pc: np.ndarray) -> np.ndarray:
        """Randomly perturbs the point cloud PC around its central point,
         according to the random rotation and translation."""
        assert pc.ndim == 2 and pc.shape[1] == 3, f"Expected [B, 3], got {pc.shape}"

        centroid = pc.mean(axis=0)
        pc = pc - centroid

        # Compute the per-call translation vector from the input PC's radius and write into T.
        object_radius = self._object_radius(pc)
        translation_distance = self.translation_percent / 100.0 * object_radius
        self.transform[:3, 3] = self.translation_direction * translation_distance

        # Convert PC into homogenous coordinates
        ones = np.ones((pc.shape[0], 1), dtype=pc.dtype)
        pc_h = np.concatenate([pc, ones], axis=1)
        # Apply the random perturbation
        perturbed_h = pc_h @ self.transform.T.astype(pc.dtype)
        # Normalize by the homogeneous coordinate. For affine transforms (bottom row [0, 0, 0, 1])
        # this is a no-op (w == 1), but the division stays correct if the matrix ever becomes projective.
        perturbed_pc = perturbed_h[:, :3] / perturbed_h[:, 3:4]
        perturbed_pc = perturbed_pc + centroid
        return perturbed_pc
