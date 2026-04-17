#!/usr/bin/env python3
"""
Procedural terrain + vegetation dataset generator (binary semantic labels).

Example usage:
  python tools/generate_procedural_terrain_dataset.py --num-samples 10 --seed 42
  python tools/generate_procedural_terrain_dataset.py --preview --seed 7
  python tools/generate_procedural_terrain_dataset.py --terrain-size-x 80 --terrain-size-y 60 --ground-resolution 0.4

Outputs (default root: data/proceedural_generation):
  - pointclouds/<sample_name>_pcl.bin   (float32 XYZI, shape Nx4)
  - labels/<sample_name>.npy            (int32 labels, shape N, 0=terrain, 1=vegetation)
  - labels/<sample_name>.json           (metadata + generation parameters)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

LABEL_TERRAIN = 0
LABEL_VEGETATION = 1


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _make_low_freq_field(rng: np.random.Generator, grid_shape: tuple[int, int]) -> np.ndarray:
    """Create smooth pseudo-noise over [0, 1]x[0, 1] using sinusoid mixtures."""
    gy, gx = grid_shape
    yy = np.linspace(0.0, 1.0, gy, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, gx, dtype=np.float32)[None, :]

    field = np.zeros((gy, gx), dtype=np.float32)
    n_terms = int(rng.integers(4, 8))
    for _ in range(n_terms):
        fx = rng.uniform(0.5, 3.5)
        fy = rng.uniform(0.5, 3.5)
        phase_x = rng.uniform(0.0, 2.0 * np.pi)
        phase_y = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.2, 1.0)

        term = (
            np.sin(2.0 * np.pi * fx * xx + phase_x)
            * np.cos(2.0 * np.pi * fy * yy + phase_y)
        ).astype(np.float32)
        field += float(amp) * term

    # Add a few smooth random Gaussian bumps.
    for _ in range(int(rng.integers(2, 6))):
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.9)
        sx = rng.uniform(0.08, 0.25)
        sy = rng.uniform(0.08, 0.25)
        amp = rng.uniform(-0.8, 0.8)
        dx = (xx - cx) / sx
        dy = (yy - cy) / sy
        field += float(amp) * np.exp(-0.5 * (dx * dx + dy * dy)).astype(np.float32)

    # Normalize to roughly [-1, 1].
    f_min = float(field.min(initial=0.0))
    f_max = float(field.max(initial=1.0))
    if f_max > f_min:
        field = 2.0 * (field - f_min) / (f_max - f_min) - 1.0
    return field.astype(np.float32)


def _bilinear_sample(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Bilinear sampling of field defined over rect domain."""
    h, w = field.shape

    u = (x - x_min) / max(x_max - x_min, 1e-6)
    v = (y - y_min) / max(y_max - y_min, 1e-6)
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    px = u * (w - 1)
    py = v * (h - 1)

    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = (px - x0).astype(np.float32)
    wy = (py - y0).astype(np.float32)

    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]

    top = f00 * (1.0 - wx) + f10 * wx
    bot = f01 * (1.0 - wx) + f11 * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def _make_height_function(
    rng: np.random.Generator,
    terrain_size_x: float,
    terrain_size_y: float,
) -> tuple[dict[str, Any], callable]:
    x_half = 0.5 * terrain_size_x
    y_half = 0.5 * terrain_size_y

    grid_h = 96
    grid_w = 96
    low_field = _make_low_freq_field(rng, (grid_h, grid_w))

    params: dict[str, Any] = {
        "base_amp": float(rng.uniform(0.8, 2.2)),
        "fx1": float(rng.uniform(0.04, 0.11)),
        "fy1": float(rng.uniform(0.04, 0.11)),
        "phase1": float(rng.uniform(0.0, 2.0 * np.pi)),
        "phase2": float(rng.uniform(0.0, 2.0 * np.pi)),
        "med_amp": float(rng.uniform(0.25, 0.9)),
        "fx2": float(rng.uniform(0.10, 0.35)),
        "fy2": float(rng.uniform(0.10, 0.35)),
        "phase3": float(rng.uniform(0.0, 2.0 * np.pi)),
        "phase4": float(rng.uniform(0.0, 2.0 * np.pi)),
        "radial_amp": float(rng.uniform(-1.1, 1.2)),
        "radial_scale": float(rng.uniform(0.35, 0.9)),
        "tilt_x": float(rng.uniform(-0.02, 0.02)),
        "tilt_y": float(rng.uniform(-0.02, 0.02)),
        "noise_amp": float(rng.uniform(0.15, 0.55)),
    }

    def height_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float32)
        yy = np.asarray(y, dtype=np.float32)

        low = params["base_amp"] * (
            np.sin(2.0 * np.pi * params["fx1"] * xx + params["phase1"])
            + 0.7 * np.cos(2.0 * np.pi * params["fy1"] * yy + params["phase2"])
        )
        med = params["med_amp"] * (
            np.sin(2.0 * np.pi * params["fx2"] * xx + params["phase3"])
            * np.cos(2.0 * np.pi * params["fy2"] * yy + params["phase4"])
        )

        xr = xx / max(x_half, 1e-6)
        yr = yy / max(y_half, 1e-6)
        rr2 = xr * xr + yr * yr
        radial = params["radial_amp"] * np.exp(-rr2 / max(params["radial_scale"], 1e-6))

        smooth_noise = params["noise_amp"] * _bilinear_sample(
            low_field,
            xx,
            yy,
            -x_half,
            x_half,
            -y_half,
            y_half,
        )

        tilt = params["tilt_x"] * xx + params["tilt_y"] * yy

        z = low + med + radial + smooth_noise + tilt
        return z.astype(np.float32)

    return params, height_fn


def _compute_intensity(
    rng: np.random.Generator,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    height_fn,
    terrain_size_x: float,
    terrain_size_y: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)

    dz = 0.15
    hx1 = height_fn(np.clip(x + dz, -0.5 * terrain_size_x, 0.5 * terrain_size_x), y)
    hx0 = height_fn(np.clip(x - dz, -0.5 * terrain_size_x, 0.5 * terrain_size_x), y)
    hy1 = height_fn(x, np.clip(y + dz, -0.5 * terrain_size_y, 0.5 * terrain_size_y))
    hy0 = height_fn(x, np.clip(y - dz, -0.5 * terrain_size_y, 0.5 * terrain_size_y))

    slope = np.sqrt(((hx1 - hx0) / (2.0 * dz)) ** 2 + ((hy1 - hy0) / (2.0 * dz)) ** 2)

    z_min = float(z.min(initial=0.0))
    z_max = float(z.max(initial=1.0))
    z_norm = (z - z_min) / max(z_max - z_min, 1e-6)

    slope_norm = slope / max(float(np.percentile(slope, 95)), 1e-6)
    slope_norm = np.clip(slope_norm, 0.0, 1.0)

    base = 0.42 + 0.35 * z_norm + 0.20 * slope_norm
    noise = rng.normal(0.0, 0.035, size=z.shape[0]).astype(np.float32)
    intensity = np.clip(base + noise, 0.0, 1.0).astype(np.float32)
    return intensity


def _generate_terrain_points(
    rng: np.random.Generator,
    height_fn,
    terrain_size_x: float,
    terrain_size_y: float,
    ground_resolution: float,
    jitter_xy: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_vals = np.arange(-0.5 * terrain_size_x, 0.5 * terrain_size_x + 1e-6, ground_resolution, dtype=np.float32)
    y_vals = np.arange(-0.5 * terrain_size_y, 0.5 * terrain_size_y + 1e-6, ground_resolution, dtype=np.float32)
    xx, yy = np.meshgrid(x_vals, y_vals, indexing="xy")

    x = xx.reshape(-1)
    y = yy.reshape(-1)

    if jitter_xy > 0.0:
        x += rng.normal(0.0, jitter_xy, size=x.shape[0]).astype(np.float32)
        y += rng.normal(0.0, jitter_xy, size=y.shape[0]).astype(np.float32)
        x = np.clip(x, -0.5 * terrain_size_x, 0.5 * terrain_size_x)
        y = np.clip(y, -0.5 * terrain_size_y, 0.5 * terrain_size_y)

    z = height_fn(x, y)
    intensity = _compute_intensity(rng, x, y, z, height_fn, terrain_size_x, terrain_size_y)

    pts = np.column_stack([x, y, z, intensity]).astype(np.float32)
    labels = np.full(pts.shape[0], LABEL_TERRAIN, dtype=np.int32)
    return pts, labels


def _generate_single_vegetation_clump(
    rng: np.random.Generator,
    cx: float,
    cy: float,
    cz: float,
    n_points: int,
) -> np.ndarray:
    """Generate one organic clump as stem + canopy points anchored at terrain height."""
    radius = float(rng.uniform(0.35, 1.4))
    height = float(rng.uniform(0.8, 2.8))
    taper = float(rng.uniform(0.45, 0.9))
    lean_x = float(rng.uniform(-0.10, 0.10))
    lean_y = float(rng.uniform(-0.10, 0.10))

    stem_n = max(6, int(0.10 * n_points))
    canopy_n = max(0, n_points - stem_n)

    # Stem-like points near the center with mild lean.
    stem_t = rng.uniform(0.0, 1.0, size=stem_n).astype(np.float32)
    stem_ang = rng.uniform(0.0, 2.0 * np.pi, size=stem_n).astype(np.float32)
    stem_r = rng.uniform(0.0, 0.08 * radius, size=stem_n).astype(np.float32)

    stem_x = cx + lean_x * stem_t + stem_r * np.cos(stem_ang)
    stem_y = cy + lean_y * stem_t + stem_r * np.sin(stem_ang)
    stem_z = cz + 0.02 + stem_t * (0.65 * height)

    # Canopy points: denser near center and upper half, tapered footprint with height.
    u = rng.uniform(0.0, 1.0, size=canopy_n).astype(np.float32)
    v = rng.uniform(0.0, 1.0, size=canopy_n).astype(np.float32)
    ang = rng.uniform(0.0, 2.0 * np.pi, size=canopy_n).astype(np.float32)

    canopy_h = (height * (0.2 + 0.8 * (u ** 0.7))).astype(np.float32)
    radial_max = (radius * (1.0 - taper * (canopy_h / max(height, 1e-6)))).astype(np.float32)
    radial_max = np.clip(radial_max, 0.12 * radius, radius)

    # sqrt(v) keeps more points near center compared to uniform radius.
    rr = radial_max * np.sqrt(v)
    asym = 1.0 + 0.25 * np.sin(ang * 2.0 + rng.uniform(0.0, 2.0 * np.pi))

    can_x = cx + lean_x * (canopy_h / max(height, 1e-6)) + rr * asym * np.cos(ang)
    can_y = cy + lean_y * (canopy_h / max(height, 1e-6)) + rr * np.sin(ang)
    can_z = cz + 0.05 + canopy_h + rng.normal(0.0, 0.05, size=canopy_n).astype(np.float32)

    x = np.concatenate([stem_x.astype(np.float32), can_x.astype(np.float32)], axis=0)
    y = np.concatenate([stem_y.astype(np.float32), can_y.astype(np.float32)], axis=0)
    z = np.concatenate([stem_z.astype(np.float32), can_z.astype(np.float32)], axis=0)

    # Keep vegetation on or above local ground anchor.
    z = np.maximum(z, np.float32(cz + 0.01))

    # Slightly broader intensity range for vegetation.
    i_base = 0.55 + 0.35 * rng.uniform(0.0, 1.0, size=x.shape[0]).astype(np.float32)
    i_noise = rng.normal(0.0, 0.04, size=x.shape[0]).astype(np.float32)
    intensity = np.clip(i_base + i_noise, 0.0, 1.0).astype(np.float32)

    return np.column_stack([x, y, z, intensity]).astype(np.float32)


def _generate_vegetation_points(
    rng: np.random.Generator,
    height_fn,
    terrain_size_x: float,
    terrain_size_y: float,
    min_vegetation_clumps: int,
    max_vegetation_clumps: int,
    min_points_per_clump: int,
    max_points_per_clump: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_clumps = int(rng.integers(min_vegetation_clumps, max_vegetation_clumps + 1))
    clump_points: list[np.ndarray] = []

    clump_meta: list[dict[str, float | int]] = []
    for _ in range(n_clumps):
        cx = float(rng.uniform(-0.48 * terrain_size_x, 0.48 * terrain_size_x))
        cy = float(rng.uniform(-0.48 * terrain_size_y, 0.48 * terrain_size_y))
        cz = float(height_fn(np.array([cx], dtype=np.float32), np.array([cy], dtype=np.float32))[0])

        n_points = int(rng.integers(min_points_per_clump, max_points_per_clump + 1))
        clump = _generate_single_vegetation_clump(rng, cx, cy, cz, n_points)
        clump_points.append(clump)
        clump_meta.append({"cx": cx, "cy": cy, "cz": cz, "points": n_points})

    if clump_points:
        pts = np.concatenate(clump_points, axis=0).astype(np.float32)
    else:
        pts = np.zeros((0, 4), dtype=np.float32)

    labels = np.full(pts.shape[0], LABEL_VEGETATION, dtype=np.int32)
    meta = {
        "num_clumps": n_clumps,
        "clumps": clump_meta,
    }
    return pts, labels, meta


def _validate(points_xyzi: np.ndarray, labels: np.ndarray) -> None:
    if points_xyzi.ndim != 2 or points_xyzi.shape[1] != 4:
        raise ValueError(f"points must have shape (N, 4), got {points_xyzi.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got {labels.shape}")
    if points_xyzi.shape[0] != labels.shape[0]:
        raise ValueError("point/label length mismatch")
    if not np.isfinite(points_xyzi).all():
        raise ValueError("points contain NaN or Inf")
    uniq = np.unique(labels)
    if not np.all(np.isin(uniq, np.array([LABEL_TERRAIN, LABEL_VEGETATION], dtype=labels.dtype))):
        raise ValueError(f"labels must be only 0/1, got {uniq.tolist()}")


def _save_sample(
    sample_name: str,
    out_root: Path,
    points_xyzi: np.ndarray,
    labels: np.ndarray,
    generation_params: dict[str, Any],
) -> tuple[Path, Path, Path]:
    point_dir = out_root / "pointclouds"
    label_dir = out_root / "labels"
    _mkdir(point_dir)
    _mkdir(label_dir)

    bin_path = point_dir / f"{sample_name}_pcl.bin"
    npy_path = label_dir / f"{sample_name}.npy"
    json_path = label_dir / f"{sample_name}.json"

    points_xyzi.astype(np.float32, copy=False).tofile(bin_path)
    np.save(npy_path, labels.astype(np.int32, copy=False))

    terrain_count = int((labels == LABEL_TERRAIN).sum())
    vegetation_count = int((labels == LABEL_VEGETATION).sum())

    metadata = {
        "sample_name": sample_name,
        "point_count": int(points_xyzi.shape[0]),
        "terrain_point_count": terrain_count,
        "vegetation_point_count": vegetation_count,
        "generation_params": generation_params,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return bin_path, npy_path, json_path


def generate_sample(sample_name: str, rng: np.random.Generator, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    terrain_params, height_fn = _make_height_function(
        rng,
        terrain_size_x=args.terrain_size_x,
        terrain_size_y=args.terrain_size_y,
    )

    terrain_pts, terrain_labels = _generate_terrain_points(
        rng,
        height_fn,
        terrain_size_x=args.terrain_size_x,
        terrain_size_y=args.terrain_size_y,
        ground_resolution=args.ground_resolution,
        jitter_xy=args.jitter_xy,
    )

    vegetation_pts, vegetation_labels, vegetation_meta = _generate_vegetation_points(
        rng,
        height_fn,
        terrain_size_x=args.terrain_size_x,
        terrain_size_y=args.terrain_size_y,
        min_vegetation_clumps=args.min_vegetation_clumps,
        max_vegetation_clumps=args.max_vegetation_clumps,
        min_points_per_clump=args.min_points_per_clump,
        max_points_per_clump=args.max_points_per_clump,
    )

    points = np.concatenate([terrain_pts, vegetation_pts], axis=0).astype(np.float32)
    labels = np.concatenate([terrain_labels, vegetation_labels], axis=0).astype(np.int32)

    if args.noise_std > 0.0:
        points[:, :3] += rng.normal(0.0, args.noise_std, size=(points.shape[0], 3)).astype(np.float32)

    _validate(points, labels)

    generation_params = {
        "seed_used": int(rng.bit_generator.state["state"]["state"] if isinstance(rng.bit_generator.state, dict) and "state" in rng.bit_generator.state else 0),
        "terrain_size_x": float(args.terrain_size_x),
        "terrain_size_y": float(args.terrain_size_y),
        "ground_resolution": float(args.ground_resolution),
        "jitter_xy": float(args.jitter_xy),
        "noise_std": float(args.noise_std),
        "min_vegetation_clumps": int(args.min_vegetation_clumps),
        "max_vegetation_clumps": int(args.max_vegetation_clumps),
        "min_points_per_clump": int(args.min_points_per_clump),
        "max_points_per_clump": int(args.max_points_per_clump),
        "terrain_height_params": terrain_params,
        "vegetation": vegetation_meta,
    }
    return points, labels, generation_params


def preview_sample(points_xyzi: np.ndarray, labels: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    xyz = points_xyzi[:, :3]
    terrain_mask = labels == LABEL_TERRAIN
    veg_mask = labels == LABEL_VEGETATION

    # Downsample for fast interactive preview.
    max_show = 50000
    if xyz.shape[0] > max_show:
        idx = np.linspace(0, xyz.shape[0] - 1, max_show, dtype=np.int64)
        xyz = xyz[idx]
        terrain_mask = terrain_mask[idx]
        veg_mask = veg_mask[idx]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if np.any(terrain_mask):
        ax.scatter(
            xyz[terrain_mask, 0],
            xyz[terrain_mask, 1],
            xyz[terrain_mask, 2],
            s=1,
            c="#b3a369",
            alpha=0.65,
            label="terrain (0)",
        )
    if np.any(veg_mask):
        ax.scatter(
            xyz[veg_mask, 0],
            xyz[veg_mask, 1],
            xyz[veg_mask, 2],
            s=2,
            c="#2f8f2f",
            alpha=0.85,
            label="vegetation (1)",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Procedural Terrain + Vegetation (Preview)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate procedural binary segmentation point cloud samples.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=1234, help="Global random seed.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/proceedural_generation"),
        help="Output root directory.",
    )

    parser.add_argument("--terrain-size-x", type=float, default=60.0, help="Terrain width in meters.")
    parser.add_argument("--terrain-size-y", type=float, default=60.0, help="Terrain depth in meters.")
    parser.add_argument("--ground-resolution", type=float, default=0.45, help="Spacing for terrain support points.")

    parser.add_argument("--min-vegetation-clumps", type=int, default=18)
    parser.add_argument("--max-vegetation-clumps", type=int, default=40)
    parser.add_argument("--min-points-per-clump", type=int, default=120)
    parser.add_argument("--max-points-per-clump", type=int, default=420)

    parser.add_argument("--noise-std", type=float, default=0.015, help="Gaussian coordinate noise stddev.")
    parser.add_argument("--jitter-xy", type=float, default=0.08, help="Terrain XY jitter stddev.")

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate one sample and show matplotlib preview (still saved to disk).",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    if args.ground_resolution <= 0.0:
        raise ValueError("--ground-resolution must be > 0")
    if args.min_vegetation_clumps < 0:
        raise ValueError("--min-vegetation-clumps must be >= 0")
    if args.max_vegetation_clumps < args.min_vegetation_clumps:
        raise ValueError("--max-vegetation-clumps must be >= --min-vegetation-clumps")
    if args.min_points_per_clump <= 0:
        raise ValueError("--min-points-per-clump must be > 0")
    if args.max_points_per_clump < args.min_points_per_clump:
        raise ValueError("--max-points-per-clump must be >= --min-points-per-clump")
    if args.noise_std < 0.0:
        raise ValueError("--noise-std must be >= 0")
    if args.jitter_xy < 0.0:
        raise ValueError("--jitter-xy must be >= 0")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args)

    out_root = Path(args.out_root)
    _mkdir(out_root / "pointclouds")
    _mkdir(out_root / "labels")

    global_rng = np.random.default_rng(args.seed)
    total = 1 if args.preview else args.num_samples

    first_points = None
    first_labels = None

    for i in range(total):
        sample_name = f"procedural_{i:03d}"
        sample_seed = int(global_rng.integers(0, 2**31 - 1))
        sample_rng = np.random.default_rng(sample_seed)

        points_xyzi, labels, generation_params = generate_sample(sample_name, sample_rng, args)
        generation_params["sample_seed"] = sample_seed

        bin_path, npy_path, json_path = _save_sample(
            sample_name=sample_name,
            out_root=out_root,
            points_xyzi=points_xyzi,
            labels=labels,
            generation_params=generation_params,
        )

        terrain_n = int((labels == LABEL_TERRAIN).sum())
        veg_n = int((labels == LABEL_VEGETATION).sum())
        print(
            f"[{i + 1}/{total}] {sample_name} | terrain={terrain_n} vegetation={veg_n} total={points_xyzi.shape[0]}"
        )
        print(f"  pointcloud: {bin_path}")
        print(f"  labels:     {npy_path}")
        print(f"  metadata:   {json_path}")

        if i == 0:
            first_points = points_xyzi
            first_labels = labels

    if args.preview and first_points is not None and first_labels is not None:
        preview_sample(first_points, first_labels)


if __name__ == "__main__":
    main()
