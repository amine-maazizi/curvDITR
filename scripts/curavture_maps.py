from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import open3d as o3d


def load_points(pointcloud_path: str | Path) -> np.ndarray:
    path = Path(pointcloud_path)
    if path.suffix.lower() == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"Expected float32 XYZI layout in {path}, got {raw.size} values")
        return raw.reshape(-1, 4)[:, :3]

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"No points loaded from {path}")
    return points.astype(np.float32)


def compute_surface_variation_from_points(points: np.ndarray, k: int = 30) -> np.ndarray:
    """
    Compute local surface variation (roughness) from PCA on k-nearest neighbors.

    surface_variation ~= smallest eigenvalue / sum(eigenvalues).
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    n = points.shape[0]
    if n == 0:
        raise ValueError("point cloud is empty")

    knn = max(3, min(int(k), n))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(pcd)

    roughness = np.empty(n, dtype=np.float32)
    for i in range(n):
        _, idx, _ = tree.search_knn_vector_3d(points[i], knn)
        neighbors = points[np.asarray(idx, dtype=np.int64), :]
        centered = neighbors - neighbors.mean(axis=0, keepdims=True)
        denom = max(centered.shape[0] - 1, 1)
        cov = (centered.T @ centered) / float(denom)
        eigenvalues = np.linalg.eigvalsh(cov)
        total = float(eigenvalues.sum())
        roughness[i] = float(eigenvalues[0] / (total + 1e-12))

    return roughness


def compute_roughness_from_points(points: np.ndarray, k: int = 30) -> np.ndarray:
    return compute_surface_variation_from_points(points, k=k)


def compute_roughness(pcd_path: str, k: int = 30) -> np.ndarray:
    points = load_points(pcd_path)
    return compute_surface_variation_from_points(points, k=k)


# Backward-compatible aliases.
def compute_curvature_from_points(points: np.ndarray, k: int = 30) -> np.ndarray:
    return compute_surface_variation_from_points(points, k=k)


def compute_curvature(pcd_path: str, k: int = 30) -> np.ndarray:
    return compute_roughness(pcd_path, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute local roughness map for a point cloud")
    parser.add_argument("pointcloud", type=str, help="Path to point cloud (.bin or Open3D-readable format)")
    parser.add_argument("--k", type=int, default=30, help="k-nearest neighbors for local PCA")
    parser.add_argument("--save", type=str, default="", help="Optional output .npy path")
    args = parser.parse_args()

    roughness = compute_roughness(args.pointcloud, k=args.k)
    print(f"points={roughness.shape[0]} min={float(roughness.min()):.8f} max={float(roughness.max()):.8f}")

    if args.save:
        np.save(args.save, roughness)
        print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
