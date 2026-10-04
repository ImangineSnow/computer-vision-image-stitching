import numpy as np

from ransac_homography import (
    calculate_homography,
    estimate_homography_ransac,
    reprojection_errors,
)


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = (homography @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


def test_direct_linear_transform_recovers_known_transform():
    source = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [1, 0.5]], dtype=float)
    expected = np.array([[1.2, 0.1, 3], [0.05, 0.9, -2], [0.001, 0.002, 1]])
    destination = _project(source, expected)

    actual = calculate_homography(source, destination)

    assert np.allclose(actual / actual[2, 2], expected / expected[2, 2], atol=1e-8)


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(7)
    source = rng.uniform(-20, 20, size=(60, 2))
    expected = np.array([[1.1, -0.04, 5], [0.03, 0.95, -3], [0.0005, -0.0008, 1]])
    destination = _project(source, expected)
    destination += rng.normal(0, 0.08, size=destination.shape)
    destination[:12] = rng.uniform(-50, 50, size=(12, 2))

    actual, inliers = estimate_homography_ransac(
        source, destination, iterations=1500, threshold=0.5, seed=3
    )

    assert inliers.sum() >= 45
    assert reprojection_errors(source[12:], destination[12:], actual).mean() < 0.2

