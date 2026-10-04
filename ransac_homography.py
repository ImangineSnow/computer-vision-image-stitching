"""Homography estimation with Direct Linear Transform and RANSAC."""

from __future__ import annotations

import numpy as np


def create_design_matrix(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Build the DLT design matrix for corresponding 2D points."""
    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and destination must both have shape (N, 2)")
    if len(source) < 4:
        raise ValueError("at least four point pairs are required")

    matrix = []
    for (x, y), (u, v) in zip(source, destination):
        matrix.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        matrix.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    return np.asarray(matrix, dtype=np.float64)


def calculate_homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Estimate a homography from point pairs using DLT."""
    matrix = create_design_matrix(source, destination)
    _, _, vt = np.linalg.svd(matrix)
    homography = vt[-1].reshape(3, 3)
    if np.isclose(homography[2, 2], 0):
        raise ValueError("degenerate point configuration")
    return homography / homography[2, 2]


def reprojection_errors(
    source: np.ndarray, destination: np.ndarray, homography: np.ndarray
) -> np.ndarray:
    """Return Euclidean reprojection errors for every correspondence."""
    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    homogeneous = np.column_stack([source, np.ones(len(source))])
    projected = (homography @ homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return np.linalg.norm(projected - destination, axis=1)


def estimate_homography_ransac(
    source: np.ndarray,
    destination: np.ndarray,
    iterations: int = 2000,
    threshold: float = 4.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a robust homography and return it with its inlier mask."""
    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if source.shape != destination.shape or len(source) < 4:
        raise ValueError("matching source and destination arrays need at least four points")

    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = 0
    best_error = np.inf

    for _ in range(iterations):
        indices = rng.choice(len(source), size=4, replace=False)
        try:
            candidate = calculate_homography(source[indices], destination[indices])
            errors = reprojection_errors(source, destination, candidate)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue

        mask = errors < threshold
        count = int(mask.sum())
        mean_error = float(errors[mask].mean()) if count else np.inf
        if count > best_count or (count == best_count and mean_error < best_error):
            best_mask = mask
            best_count = count
            best_error = mean_error

    if best_mask is None or best_count < 4:
        raise ValueError("RANSAC could not find a valid homography")

    refined = calculate_homography(source[best_mask], destination[best_mask])
    return refined, best_mask

