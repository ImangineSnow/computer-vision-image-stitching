"""Create a panorama from two overlapping images using SIFT and homography."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ransac_homography import estimate_homography_ransac


def load_image(path: str | Path, width: int | None = None) -> np.ndarray:
    """Load a color image and optionally resize it while preserving aspect ratio."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if width is not None and width > 0 and image.shape[1] != width:
        scale = width / image.shape[1]
        image = cv2.resize(image, (width, round(image.shape[0] * scale)))
    return image


def extract_features(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detect SIFT keypoints and descriptors."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 4:
        raise ValueError("not enough visual features were detected")
    return keypoints, descriptors


def match_features(
    source_descriptors: np.ndarray,
    destination_descriptors: np.ndarray,
    ratio: float = 0.72,
) -> list[cv2.DMatch]:
    """Match descriptors with FLANN and Lowe's ratio test."""
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=80)
    )
    pairs = matcher.knnMatch(source_descriptors, destination_descriptors, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    if len(matches) < 4:
        raise ValueError("not enough reliable matches to estimate a homography")
    return matches


def matched_points(
    source_keypoints: list[cv2.KeyPoint],
    destination_keypoints: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert descriptor matches to paired point arrays."""
    source = np.float64([source_keypoints[m.queryIdx].pt for m in matches])
    destination = np.float64([destination_keypoints[m.trainIdx].pt for m in matches])
    return source, destination


def compose_panorama(
    left: np.ndarray, right: np.ndarray, right_to_left: np.ndarray
) -> np.ndarray:
    """Warp the right image into the left image coordinate system and blend overlap."""
    left_h, left_w = left.shape[:2]
    right_h, right_w = right.shape[:2]

    right_corners = np.float32(
        [[0, 0], [right_w, 0], [right_w, right_h], [0, right_h]]
    ).reshape(-1, 1, 2)
    left_corners = np.float32(
        [[0, 0], [left_w, 0], [left_w, left_h], [0, left_h]]
    ).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(right_corners, right_to_left)
    all_corners = np.vstack([warped_corners, left_corners])

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    translation = np.array(
        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64
    )
    size = (x_max - x_min, y_max - y_min)

    warped_right = cv2.warpPerspective(right, translation @ right_to_left, size)
    right_mask = cv2.warpPerspective(
        np.ones((right_h, right_w), dtype=np.uint8), translation @ right_to_left, size
    ).astype(bool)

    panorama = warped_right.copy()
    y0, x0 = -y_min, -x_min
    left_region = panorama[y0 : y0 + left_h, x0 : x0 + left_w]
    right_region_mask = right_mask[y0 : y0 + left_h, x0 : x0 + left_w]

    overlap = right_region_mask
    left_only = ~right_region_mask
    left_region[left_only] = left[left_only]
    left_region[overlap] = (
        0.5 * left_region[overlap].astype(np.float32)
        + 0.5 * left[overlap].astype(np.float32)
    ).astype(np.uint8)
    return panorama


def stitch_images(
    left: np.ndarray,
    right: np.ndarray,
    ratio: float = 0.72,
    threshold: float = 4.0,
    iterations: int = 2000,
) -> tuple[np.ndarray, int, int]:
    """Stitch two images and return panorama, match count, and inlier count."""
    left_keypoints, left_descriptors = extract_features(left)
    right_keypoints, right_descriptors = extract_features(right)
    matches = match_features(right_descriptors, left_descriptors, ratio)
    source, destination = matched_points(right_keypoints, left_keypoints, matches)
    homography, inliers = estimate_homography_ransac(
        source, destination, iterations=iterations, threshold=threshold
    )
    panorama = compose_panorama(left, right, homography)
    return panorama, len(matches), int(inliers.sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", default="data/img_left.jpg")
    parser.add_argument("--right", default="data/img_right.jpg")
    parser.add_argument("--output", default="output/stitched_image.jpg")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--ratio", type=float, default=0.72)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--iterations", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = load_image(args.left, args.width)
    right = load_image(args.right, args.width)
    panorama, matches, inliers = stitch_images(
        left, right, args.ratio, args.threshold, args.iterations
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), panorama):
        raise OSError(f"could not write output image: {output}")
    print(f"Saved {output} using {inliers}/{matches} inlier matches.")


if __name__ == "__main__":
    main()

