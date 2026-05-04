import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation
import torch


def get_random_rotation_matrix(angle_range_deg_x, angle_range_deg_y, angle_range_deg_z):
    """ Generates a rotation matrix that represents 3 subsequent rotations with random Euler angles.

    Args:
        angle_range_deg_x: a tuple with (min, max) rotation along X axis
        angle_range_deg_y: a tuple with (min, max) rotation along X axis
        angle_range_deg_z: a tuple with (min, max) rotation along X axis
    """
    angles_deg = [
        random.uniform(*angle_range_deg_x),
        random.uniform(*angle_range_deg_y),
        random.uniform(*angle_range_deg_z)
    ]

    # Convert to a Rotation object using Euler angles
    rot = Rotation.from_euler(seq='xyz', angles=angles_deg, degrees=True)

    return rot.as_matrix()

class RandomTranslationTransform:
    def __init__(self, translation_range=0.02, same_translation=False):
        self.translation_range = translation_range
        self.same_translation = same_translation

    def __call__(self, point_clouds):
        """
        point_clouds: Tensor of shape [B, N, 3] — batch of B point clouds
        returns: Transformed tensor of shape [B, N, 3]
        """
        device = point_clouds.device

        if self.same_translation:
            translations = self.random_translation(device)
            # One translation will broadcast to all point clouds
            return point_clouds + translations
        else:
            Bdim = point_clouds.shape[0]
            translations = self.random_translation(device, batch_size=Bdim)
            return point_clouds + translations

    def random_translation(self, device, batch_size=1):
        return (2 * torch.rand(batch_size, 1, 3, device=device) - 1) * self.translation_range


class RandomRotationTransform:
    """Applies a random 3D rotation to each point cloud in a [B, N, 3] tensor."""

    def __init__(self, angle_range_deg=(-3, 3), same_rotation=True):
        self.angle_range_deg = angle_range_deg
        self.same_rotation = same_rotation
        self.center_first = True

    def __call__(self, point_clouds):
        """
        Args:
            point_clouds (torch.Tensor): Shape [B, N, 3].
            First dim is the triplet, second - no. points in the cloud. Third - coordinates.
        Returns:
            torch.Tensor: Rotated point clouds, shape [B, N, 3]
        """
        device = point_clouds.device

        if self.same_rotation:
            rot_matrix = self.get_rotation_matrix(self.angle_range_deg).to(device)
            return self.rotate_batch(point_clouds, rot_matrix)
        else:
            # TODO(alexta): make it more efficient with no loops
            pcs = []
            Bdim = point_clouds.shape[0]
            for i in range(Bdim):
                rot_matrix = self.get_rotation_matrix(self.angle_range_deg).to(device)
                pcs.append(self.rotate_batch(point_clouds[i:i + 1], rot_matrix)[0])
            return torch.stack(pcs, dim=0)

    def rotate_batch(self, pcs, rot_matrix):
        """
        pcs: [B, N, 3]
        rot_matrix: [3, 3]
        Returns: [B, N, 3]
        """
        pcs_rotated = pcs @ rot_matrix.T  # [B, N, 3]
        return pcs_rotated

    def get_rotation_matrix(self, angle_range_deg):
        """ Generates a rotation matrix that represents 3 subsequent rotations with random Euler angles."""
        rot = get_random_rotation_matrix(angle_range_deg, angle_range_deg, angle_range_deg)
        return torch.from_numpy(rot).float()


class RandomNoisePointsTransform:
    """
    Randomly adds noise points to the input data.

    Args:
        noise_scale (float): Scale factor for noise spread, w.r.t the bounding box of the point cloud.
        num_noise_points (int): Number of noise points to add. If not specified, defaults to 5% of the point cloud size.
    """

    def __init__(self, noise_scale=0.1, num_noise_points=None, negatives_only=False):
        self.noise_scale = noise_scale
        self.num_noise_points = num_noise_points
        self.negatives_only = negatives_only

    def __call__(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_clouds (Tensor): Input tensor of shape [B, N, 3].

        Returns:
            Tensor: Output tensor of shape [B, N + M, 3] (M = num_noise_points).
        """
        B, N, _ = point_clouds.shape
        M = self.num_noise_points or int(N * 0.05)
        output = []

        for b in range(B):
            pc = point_clouds[b]  # [N, 3]

            if self.negatives_only and b != B - 1:
                output.append(pc)
                continue

            min_bounds = pc.min(dim=0).values
            max_bounds = pc.max(dim=0).values
            bounds_range = max_bounds - min_bounds

            # Expand the bounds
            min_bounds -= self.noise_scale * bounds_range
            max_bounds += self.noise_scale * bounds_range

            # Uniform noise within expanded box
            noise = torch.rand((M, 3), device=pc.device) * (max_bounds - min_bounds) + min_bounds

            pc_aug = torch.cat([pc, noise], dim=0)  # [N+M, 3]
            output.append(pc_aug)

        # Pad shorter point clouds with zeros to match length
        max_len = max(pc.shape[0] for pc in output)
        padded = [torch.cat([pc, torch.zeros((max_len - pc.shape[0], 3), device=pc.device)], dim=0) for pc in output]
        return torch.stack(padded, dim=0)  # [B, max_len, 3]


class GaussianNoiseTransform:
    def __init__(self, sigma=0.005, clip=0.02):
        """
        Add Gaussian noise to each point in the point cloud.
        
        Args:
            sigma (float): Standard deviation of noise.
            clip (float): Maximum absolute value of noise.
        """
        self.sigma = sigma
        self.clip = clip

    def __call__(self, point_clouds):
        noise = self.sigma * torch.randn_like(point_clouds)
        noise = torch.clamp(noise, min=-self.clip, max=self.clip)

        return point_clouds + noise


class RandomCutTransform:
    def __init__(self, min_cuts, max_cuts, min_cut_r, max_cut_r, max_percent_pts_cut):
        self.min_cuts = min_cuts
        self.max_cuts = max_cuts
        self.min_cut_r = min_cut_r
        self.max_cut_r = max_cut_r
        self.max_percent_pts_cut = max_percent_pts_cut
        self.weights_power = 3

    def random_cut_patch(self, patch):
        """
                patch: (N, 3) float tensor
                returns: (M, 3) tensor with M <= N
                """
        assert patch.ndim == 2 and patch.size(1) == 3, "patch must be (N,3)"
        N = patch.size(0)
        if N == 0:
            return patch

        # Get distances from the center to the edge of the centered patch
        dist2 = (patch ** 2).sum(dim=1)
        # Max radius of a patch
        max_r = torch.sqrt(dist2).max()

        num_cuts = int(
            torch.randint(low=self.min_cuts, high=self.max_cuts + 1, size=(1,), device=patch.device).item())
        # Sample the most distant points from the patch center, and make them centers of "cutout" regions
        weights = torch.pow(dist2, self.weights_power)
        weights = weights / weights.sum()
        idx_centers_cut = torch.multinomial(weights, num_samples=num_cuts, replacement=False)
        centers_cut = patch[idx_centers_cut]
        dist_to_centers_cut = patch[:, None, :] - centers_cut[None, :, :]
        min_dist2_to_centers_cut = (torch.einsum('ijk,ijk->ij', dist_to_centers_cut, dist_to_centers_cut)
                                    .min(dim=1).values)

        r_cut = (torch.empty(1, device=patch.device, dtype=patch.dtype).
                 uniform_(max_r * self.min_cut_r, max_r * self.max_cut_r).item())

        cut_patch = patch[min_dist2_to_centers_cut > r_cut ** 2]
        percent_cut = (N - cut_patch.size(0)) / float(N)
        if percent_cut < self.max_percent_pts_cut:
            return cut_patch
        return patch

    def __call__(self, point_clouds):
        # assert(len(patch_triplet_arr) == 3)
        transformed = []
        Bdim = len(point_clouds)

        # TODO(alexta): here we assume that the last patch is always a negative. Find a more elegant way
        for i in range(Bdim-1):
            center = torch.mean(point_clouds[i], dim=0)
            pc_cut = self.random_cut_patch(point_clouds[i]-center)
            transformed.append(pc_cut+center)
        # Append the negative patch
        transformed.append(point_clouds[Bdim-1])

        # Old code, for potential debugging / bug fixing
        # center_good = torch.mean(patch_triplet_arr[1], dim=0)
        # good_cut = self.random_cut_patch(patch_triplet_arr[1]-center_good)
        # transformed.append(good_cut+center_good)

        return transformed



class ElasticDeformationTransform:
    """
    Apply 3D elastic deformation to a batch of point clouds.

    Args:
        grid_size (int): Coarseness of the deformation grid.
        alpha (float): Scaling factor for displacement magnitude.
        smoothing (bool): Apply Gaussian filter to the displacement field.
        sigma (float): Smoothing strength for Gaussian filter.
    """

    def __init__(self, grid_size=4, alpha=0.1, smoothing=True, sigma=1.0, negatives_only=False):
        self.grid_size = grid_size
        self.alpha = alpha
        self.smoothing = smoothing
        self.sigma = sigma
        self.negatives_only = negatives_only

    def __call__(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_clouds (torch.Tensor): Shape [B, N, 3]
        Returns:
            torch.Tensor: Deformed point clouds, shape [B, N, 3]
        """
        assert point_clouds.ndim == 3 and point_clouds.shape[2] == 3, "Expected shape [B, N, 3]"

        batch_deformed = []
        for pc in point_clouds:
            if self.negatives_only and pc is not point_clouds[-1]:
                batch_deformed.append(pc)
                continue

            batch_deformed.append(self._deform(pc))
        return torch.stack(batch_deformed)

    def _deform(self, point_cloud: torch.Tensor) -> torch.Tensor:
        pc_np = point_cloud.detach().cpu().numpy()

        min_coords = pc_np.min(axis=0)
        max_coords = pc_np.max(axis=0)

        shape = [self.grid_size] * 3
        grid_x = np.linspace(min_coords[0], max_coords[0], shape[0])
        grid_y = np.linspace(min_coords[1], max_coords[1], shape[1])
        grid_z = np.linspace(min_coords[2], max_coords[2], shape[2])

        displacement = np.random.randn(*shape, 3)
        if self.smoothing:
            for i in range(3):
                displacement[..., i] = gaussian_filter(displacement[..., i], self.sigma, mode='reflect')

        displacement *= self.alpha

        # Get grid coordinates for interpolation
        grid_coords = np.vstack([
            np.interp(pc_np[:, 0], grid_x, np.arange(shape[0])),
            np.interp(pc_np[:, 1], grid_y, np.arange(shape[1])),
            np.interp(pc_np[:, 2], grid_z, np.arange(shape[2]))
        ])

        deformed = np.empty_like(pc_np)
        for i in range(3):
            deformed[:, i] = pc_np[:, i] + map_coordinates(
                displacement[..., i], grid_coords,
                order=3, mode='reflect'
            )

        return torch.from_numpy(deformed).to(point_cloud.device).type_as(point_cloud)
