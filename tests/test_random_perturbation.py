import numpy as np
import pytest

from registration.random_perturbation import RandomPerturbation


@pytest.fixture
def rng():
    # Fixed seed so tests are deterministic. The seed value itself is arbitrary.
    return np.random.default_rng(42)


@pytest.fixture
def cube():
    # Four vertices of the unit cube: origin plus the three axis tips.
    # Inter-point distances [1, 1, 1] are easy to compare against after rigid transforms.
    return np.array([[0.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def make_pc_with_radius(r: float) -> np.ndarray:
    # Two antipodal points along the x-axis: centroid is the origin, max distance from centroid is r.
    # This gives a point cloud whose RandomPerturbation._object_radius is exactly r.
    return np.array([[r, 0.0, 0.0], [-r, 0.0, 0.0]], dtype=np.float32)


class TestRandomPerturbation:

    def test_rotation_magnitude_recovered(self, rng):
        # The geodesic distance on SO(3) from identity to R is theta = arccos((tr(R) - 1) / 2).
        # Rodrigues' formula with axis-angle (u_hat, theta) yields tr(R) = 1 + 2*cos(theta) exactly,
        # so the recovered angle must equal the requested one regardless of axis direction.
        pert = RandomPerturbation(rotation_deg=10.0, translation_percent=5.0, rng=rng)
        R = pert.transform[:3, :3]
        angle_recovered = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        assert np.isclose(angle_recovered, 10.0)

    def test_translation_magnitude_recovered(self, rng):
        # Translation magnitude is materialized only after perturb() runs and computes the radius.
        # Using a PC of radius 2.0 and 5% translation -> expected length 0.10.
        pert = RandomPerturbation(rotation_deg=10.0, translation_percent=5.0, rng=rng)
        pert.perturb(make_pc_with_radius(2.0))
        t = pert.transform[:3, 3]
        assert np.isclose(np.linalg.norm(t), 0.10)

    def test_translation_direction_is_unit_vector(self, rng):
        # The stored direction should be a unit vector regardless of percent or call to perturb().
        pert = RandomPerturbation(rotation_deg=10.0, translation_percent=5.0, rng=rng)
        assert np.isclose(np.linalg.norm(pert.translation_direction), 1.0)

    def test_translation_percent_stored(self, rng):
        # Constructor argument should be exposed verbatim as the class attribute used by perturb().
        pert = RandomPerturbation(rotation_deg=10.0, translation_percent=5.0, rng=rng)
        assert pert.translation_percent == 5.0

    def test_rotation_is_orthogonal(self, rng):
        # Rotation matrices satisfy R^T @ R = I (orthogonality preserves dot products / lengths).
        # This catches construction bugs in Rodrigues' formula or the skew-symmetric K matrix.
        pert = RandomPerturbation(rotation_deg=10.0, translation_percent=5.0, rng=rng)
        R = pert.transform[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3))

    def test_rotation_has_unit_determinant(self, rng):
        # Proper rotations (SO(3)) have det = +1. det = -1 would be a reflection (improper rotation),
        # which is also orthogonal but flips chirality. This separates rotations from reflections.
        pert = RandomPerturbation(rotation_deg=10.0, translation_percent=5.0, rng=rng)
        R = pert.transform[:3, :3]
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_homogeneous_bottom_row(self, rng):
        # The 4x4 homogeneous-coordinate transform must have bottom row [0, 0, 0, 1] for an affine
        # transform; non-zero entries here would represent a projective transform (perspective).
        pert = RandomPerturbation(rotation_deg=10.0, translation_percent=5.0, rng=rng)
        assert np.allclose(pert.transform[3, :], [0, 0, 0, 1])

    def test_perturb_output_shape(self, rng, cube):
        # perturb() must return the same [N, 3] shape as the input — the homogeneous coordinate
        # is added internally and then dropped, but should not leak into the output.
        pert = RandomPerturbation(rotation_deg=10.0, translation_percent=5.0, rng=rng)
        out = pert.perturb(cube)
        assert out.shape == cube.shape

    def test_perturb_preserves_dtype(self, rng, cube):
        # Point cloud dtype (float32 vs float64) must be preserved through the transform —
        # implicit upcasting to float64 would silently double memory in downstream pipelines.
        pert = RandomPerturbation(rotation_deg=10.0, translation_percent=5.0, rng=rng)
        assert pert.perturb(cube).dtype == cube.dtype
        cube64 = cube.astype(np.float64)
        assert pert.perturb(cube64).dtype == cube64.dtype

    def test_perturb_preserves_inter_point_distances(self, rng, cube):
        # Rigid transforms (rotation + translation) are isometries: pairwise distances are invariant.
        # Translation cancels out when subtracting two transformed points; rotation is orthogonal,
        # so it preserves Euclidean norm. This is a global sanity check on the perturb() pipeline.
        pert = RandomPerturbation(rotation_deg=10.0, translation_percent=5.0, rng=rng)
        out = pert.perturb(cube)
        original = np.linalg.norm(cube[1:] - cube[0], axis=1)
        perturbed = np.linalg.norm(out[1:] - out[0], axis=1)
        assert np.allclose(original, perturbed, atol=1e-5)

    @pytest.mark.parametrize("angle_deg", [0.5, 5.0, 30.0, 90.0, 179.0])
    def test_rotation_magnitude_various_angles(self, rng, angle_deg):
        # Sweep angle across the full valid range (0, 180) deg. Specifically:
        #  - 0.5 deg: very small angle, where small-angle numerical issues could appear.
        #  - 90 deg: tr(R) = 1 + 2*cos(90) = 1, sits at midpoint of arccos domain.
        #  - 179 deg: near the antipode, where arccos((tr(R)-1)/2) approaches its endpoint
        #    and is most numerically sensitive (slope of arccos diverges at +/-1).
        pert = RandomPerturbation(rotation_deg=angle_deg, translation_percent=0.0, rng=rng)
        R = pert.transform[:3, :3]
        angle_recovered = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        assert np.isclose(angle_recovered, angle_deg, atol=1e-4)

    @pytest.mark.parametrize("percent,radius,expected", [
        # Verify the |t| = (percent/100) * radius formula across edge cases.
        # Each case perturbs a PC of the given radius and inspects T[:3, 3].
        (0.0, 2.0, 0.0),     # zero percent -> no translation regardless of radius
        (5.0, 2.0, 0.10),    # the canonical case used in other tests
        (10.0, 5.0, 0.5),    # check linearity in radius
        (100.0, 1.0, 1.0),   # 100% of unit radius -> length 1
    ])
    def test_translation_magnitude_various(self, rng, percent, radius, expected):
        pert = RandomPerturbation(rotation_deg=0.0, translation_percent=percent, rng=rng)
        pert.perturb(make_pc_with_radius(radius))
        assert np.isclose(np.linalg.norm(pert.transform[:3, 3]), expected)

    def test_zero_rotation_yields_identity_rotation(self, rng):
        # Rodrigues with theta = 0: sin(0) = 0 and (1 - cos(0)) = 0, so R = I + 0*K + 0*K^2 = I.
        # Independent of the random axis. Verifies the formula degenerates correctly.
        pert = RandomPerturbation(rotation_deg=0.0, translation_percent=5.0, rng=rng)
        assert np.allclose(pert.transform[:3, :3], np.eye(3))

    def test_zero_translation_yields_zero_vector(self, rng):
        # |t| = 0% * radius = 0; the stored direction is irrelevant once scaled by zero.
        # perturb() still runs but writes a zero translation column.
        pert = RandomPerturbation(rotation_deg=10.0, translation_percent=0.0, rng=rng)
        pert.perturb(make_pc_with_radius(2.0))
        assert np.allclose(pert.transform[:3, 3], 0.0)

    def test_identity_perturbation_leaves_pc_unchanged(self, rng, cube):
        # Both magnitudes zero -> R = I and t = 0 -> perturb() is a no-op.
        # End-to-end check that the homogeneous-coordinate plumbing is symmetric.
        pert = RandomPerturbation(rotation_deg=0.0, translation_percent=0.0, rng=rng)
        assert np.allclose(pert.perturb(cube), cube)

    def test_rng_seed_reproducibility(self, cube):
        # Same seed -> identical random axis and translation direction -> identical transform
        # and identical perturbed PC. Critical for reproducible experiments.
        a = RandomPerturbation(10.0, 5.0, rng=np.random.default_rng(123))
        b = RandomPerturbation(10.0, 5.0, rng=np.random.default_rng(123))
        assert np.allclose(a.translation_direction, b.translation_direction)
        assert np.allclose(a.transform, b.transform)
        assert np.allclose(a.perturb(cube), b.perturb(cube))

    def test_different_seeds_yield_different_transforms(self):
        # Sanity check on the converse: two distinct seeds must not (accidentally) collide.
        # If this ever fails, randomness is being short-circuited somewhere.
        a = RandomPerturbation(10.0, 5.0, rng=np.random.default_rng(1))
        b = RandomPerturbation(10.0, 5.0, rng=np.random.default_rng(2))
        assert not np.allclose(a.transform, b.transform)

    def test_direction_is_isotropic(self):
        # Statistical check that translation directions are uniform on S^2 (no axis bias).
        # The expected mean of n uniform unit vectors on the sphere is the zero vector;
        # by CLT, |sample_mean| scales as O(1/sqrt(n)). For n=4000, std ~= 1/sqrt(4000) ~= 0.016,
        # so a threshold of 0.05 (~3 sigma) is generous enough to avoid flakes but tight enough
        # to catch a biased sampler (e.g. uniform-cube-then-normalize).
        # Inspect translation_direction directly — no perturb() call needed.
        n = 4000
        rng = np.random.default_rng(0)
        directions = np.stack([
            RandomPerturbation(0.0, 100.0, rng=rng).translation_direction
            for _ in range(n)
        ])
        assert np.linalg.norm(directions.mean(axis=0)) < 0.05

    def test_object_radius_computation(self):
        # _object_radius = max distance from the centroid. For two antipodal points at +/- r along
        # any axis, centroid is the origin and the radius is r exactly.
        pc = make_pc_with_radius(3.0)
        assert np.isclose(RandomPerturbation._object_radius(pc), 3.0)
        # Translated PCs should give the same radius, since centroid moves with them.
        assert np.isclose(RandomPerturbation._object_radius(pc + 100.0), 3.0)

    def test_perturb_uses_input_radius(self, rng):
        # The same RandomPerturbation applied to two PCs of different radii should produce
        # translations whose magnitudes scale with the corresponding input radii.
        pert = RandomPerturbation(rotation_deg=0.0, translation_percent=10.0, rng=rng)
        pert.perturb(make_pc_with_radius(1.0))
        t1 = pert.transform[:3, 3].copy()
        pert.perturb(make_pc_with_radius(4.0))
        t2 = pert.transform[:3, 3].copy()
        # Same direction, magnitudes in 1:4 ratio.
        assert np.isclose(np.linalg.norm(t1), 0.1)
        assert np.isclose(np.linalg.norm(t2), 0.4)
        assert np.allclose(t1 / np.linalg.norm(t1), t2 / np.linalg.norm(t2))

    def test_rotation_is_about_centroid_not_origin(self, rng):
        # For an off-center PC, the centroid of the perturbed PC must move by exactly the
        # translation vector t — i.e. the rotation contributes no centroid drift.
        # Without centering inside perturb(), centroid would shift by (I - R)*c + t instead.
        pert = RandomPerturbation(rotation_deg=30.0, translation_percent=10.0, rng=rng)
        # Off-center PC: centroid at (10, 20, 30), radius 1.0 (two antipodal points along x).
        offset = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        pc = make_pc_with_radius(1.0) + offset
        out = pert.perturb(pc)
        centroid_shift = out.mean(axis=0) - pc.mean(axis=0)
        expected_t = pert.transform[:3, 3]
        assert np.allclose(centroid_shift, expected_t, atol=1e-5)

    def test_perturb_rejects_bad_shape(self, rng):
        # perturb() expects [N, 3]. Other shapes (wrong second dim, or 1D vector) should fail
        # loudly via the assertion, not silently broadcast or crash deeper in the pipeline.
        pert = RandomPerturbation(10.0, 5.0, rng=rng)
        with pytest.raises(AssertionError):
            pert.perturb(np.zeros((4, 2)))   # wrong second dimension
        with pytest.raises(AssertionError):
            pert.perturb(np.zeros(3))         # 1D, missing batch dimension


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
