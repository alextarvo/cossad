import numpy as np
import pytest

import utils.pc_utils as pc_utils
import transforms


def get_square_test_pc(width, height, num_points):
    center = [width / 2, height / 2, 0]

    # Define the corners
    corners = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width, height, 0],
        [0, height, 0],
    ])

    corner_indices = [0, 1, 2, 3]
    center_index = 4

    # Insert center point explicitly
    center_point = np.array([center])

    # Generate 995 random points inside the rectangle
    random_points = np.random.uniform(low=[0, 0, 0], high=[width, height, 0], size=(995, 3))

    # Combine all points
    point_cloud = np.vstack([corners, center_point, random_points])
    assert point_cloud.shape[0] == 1000

    return point_cloud, center_index, corner_indices, center, corners


def test_fps_dense_rectangle_finds_corners():
    """Test basic FPS functionality.
    Creates a square PC. Sets center as an initial point. Ensures that we sample its center and corners"""
    np.random.seed(42)  # for reproducibility

    # Define rectangle bounds
    width, height = 10, 10

    point_cloud, center_index, corner_indices, center, corners = get_square_test_pc(width, height , 1000)
    # Run FPS
    sampled_indices = pc_utils.farthest_pt_sampling(point_cloud, center_index, 5)
    sampled_points = point_cloud[sampled_indices]

    # Convert sampled points (except center) to tuple set
    sampled_corners = {
        tuple(pt) for i, pt in zip(sampled_indices, sampled_points) if i != center_index
    }

    expected_corners = {tuple(corner) for corner in corners}

    # Check that the center point is in the result
    assert center_index in sampled_indices, "Center point must be included."

    # Check that all 4 corners are selected
    assert expected_corners.issubset(
        sampled_corners), f"Expected corners not selected: {expected_corners - sampled_corners}"


def test_fps_rotated_dense_rectangle_finds_corners():
    """Test basic FPS functionality.
    Creates a square PC. Sets center as an initial point. Ensures that we sample its center and corners"""
    np.random.seed(42)  # for reproducibility

    # Define rectangle bounds
    width, height = 10, 10

    point_cloud, center_index, corner_indices, center, corners = get_square_test_pc(width, height , 1000)
    # Run FPS
    sampled_indices = pc_utils.farthest_pt_sampling(point_cloud, center_index, 5)
    R = transforms.get_random_rotation_matrix((-90, 90), (-90, 90), (-90, 90))
    point_cloud = point_cloud@R

    # Check that the center point is in the result
    assert center_index in sampled_indices, "Center point must be included."
    assert set(corner_indices).issubset(sampled_indices), "Corner points must be included."
    assert len(sampled_indices) == 5


def test_adaptive_fps_dense_rectangle_finds_corners():
    """Test basic FPS functionality.
    Creates a square PC. Sets center as an initial point. Ensures that we sample its center and corners"""
    np.random.seed(42)  # for reproducibility

    # Define rectangle bounds
    width, height = 10, 10
    radius = 5

    point_cloud, center_index, corner_indices, center, corners = get_square_test_pc(width, height , 1000)
    # Run FPS
    sampled_indices = pc_utils.adaptive_farthest_pt_sampling(
        point_cloud, center_index, patch_radius=5,
        min_num_points_threshold=5, max_num_points_threshold=10,
        min_coverage_threshold=1, max_coverage_threshold=2,
    )
    sampled_points = point_cloud[sampled_indices]

    # Convert sampled points (except center) to tuple set
    sampled_corners = {
        tuple(pt) for i, pt in zip(sampled_indices, sampled_points) if i != center_index
    }

    expected_corners = {tuple(corner) for corner in corners}

    # Check that the center point is in the result
    assert center_index in sampled_indices, "Center point must be included."

    # Check that all 4 corners are selected
    assert expected_corners.issubset(
        sampled_corners), f"Expected corners not selected: {expected_corners - sampled_corners}"
    assert len(sampled_indices) == 5



def test_adaptive_fps_dense_rectangle_high_min_num_points():
    """Test basic FPS functionality.
    Creates a square PC. Sets center as an initial point. Ensures that we sample its center and corners"""
    np.random.seed(42)  # for reproducibility

    # Define rectangle bounds
    width, height = 10, 10
    radius = 5

    point_cloud, center_index, corner_indices, center, corners = get_square_test_pc(width, height , 1000)
    # Run FPS
    sampled_indices = pc_utils.adaptive_farthest_pt_sampling(
        point_cloud, center_index, patch_radius=5,
        min_num_points_threshold=7, max_num_points_threshold=10,
        min_coverage_threshold=1, max_coverage_threshold=3,
    )
    sampled_points = point_cloud[sampled_indices]

    # Convert sampled points (except center) to tuple set
    sampled_corners = {
        tuple(pt) for i, pt in zip(sampled_indices, sampled_points) if i != center_index
    }

    expected_corners = {tuple(corner) for corner in corners}

    # Check that the center point is in the result
    assert center_index in sampled_indices, "Center point must be included."

    # Check that all 4 corners are selected
    assert expected_corners.issubset(
        sampled_corners), f"Expected corners not selected: {expected_corners - sampled_corners}"
    assert len(sampled_indices) == 6
    print(f' Number of points selected: {len(sampled_indices)}')


def test_adaptive_fps_dense_rectangle_high_min_coverage():
    """Test basic FPS functionality.
    Creates a square PC. Sets center as an initial point. Ensures that we sample its center and corners"""
    np.random.seed(42)  # for reproducibility

    # Define rectangle bounds
    width, height = 10, 10
    radius = 5

    point_cloud, center_index, corner_indices, center, corners = get_square_test_pc(width, height , 1000)
    # Run FPS
    sampled_indices = pc_utils.adaptive_farthest_pt_sampling(
        point_cloud, center_index, patch_radius=5,
        min_num_points_threshold=3, max_num_points_threshold=15,
        min_coverage_threshold=2, max_coverage_threshold=30,
    )
    sampled_points = point_cloud[sampled_indices]

    # Convert sampled points (except center) to tuple set
    sampled_corners = {
        tuple(pt) for i, pt in zip(sampled_indices, sampled_points) if i != center_index
    }

    expected_corners = {tuple(corner) for corner in corners}

    # Check that the center point is in the result
    assert center_index in sampled_indices, "Center point must be included."

    # Check that all 4 corners are selected
    assert expected_corners.issubset(
        sampled_corners), f"Expected corners not selected: {expected_corners - sampled_corners}"
    # We expect thatt in addition to the original points (center + corners),
    # There will be 5  more points: two very close to the boundaries, and one - somewhere between a center and a corner
    assert len(sampled_indices) == 10
    print(f' Number of points selected: {len(sampled_indices)}')
