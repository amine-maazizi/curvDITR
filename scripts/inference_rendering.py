from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _build_palette(num_classes: int) -> np.ndarray:
    # Deterministic HSV-like palette without extra dependencies.
    if num_classes <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    idx = np.arange(num_classes, dtype=np.float32)
    hue = (idx * 0.61803398875) % 1.0
    sat = np.full_like(hue, 0.75)
    val = np.full_like(hue, 0.95)

    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))

    rgb = np.empty((num_classes, 3), dtype=np.float32)
    i_mod = i % 6
    for k in range(num_classes):
        m = i_mod[k]
        if m == 0:
            rgb[k] = [val[k], t[k], p[k]]
        elif m == 1:
            rgb[k] = [q[k], val[k], p[k]]
        elif m == 2:
            rgb[k] = [p[k], val[k], t[k]]
        elif m == 3:
            rgb[k] = [p[k], q[k], val[k]]
        elif m == 4:
            rgb[k] = [t[k], p[k], val[k]]
        else:
            rgb[k] = [val[k], p[k], q[k]]
    return rgb


def _resolve_prediction_path(
    sample_name: str,
    pred_root: Path,
    pred_file: str | None = None,
) -> Path | None:
    if pred_file:
        p = Path(pred_file)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            # Preserve explicit relative paths (possibly with directories), then try CWD and repo-root resolution.
            candidates.append(p)
            candidates.append(Path.cwd() / p)
            candidates.append(Path(__file__).resolve().parents[1] / p)
            # If a bare filename is provided, allow pred_root anchoring.
            if p.parent == Path("."):
                candidates.append(pred_root / p)

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None
    else:
        resolved = pred_root / f"{sample_name}_pred.npy"

    return resolved if resolved.is_file() else None


def _resolve_pred_root(pred_root: str | Path) -> Path:
    p = Path(pred_root)
    if p.is_absolute():
        return p

    # Try current working directory first, then repository root relative to this script.
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate

    repo_root_candidate = Path(__file__).resolve().parents[1] / p
    if repo_root_candidate.exists():
        return repo_root_candidate

    return cwd_candidate


def _decode_prediction_array(raw: Any) -> np.ndarray:
    arr = np.asarray(raw)

    # Common formats:
    # - [N] integer class IDs
    # - [N, C] logits/probabilities (argmax over C)
    # - [1, N] or [N, 1] single-channel labels
    if arr.ndim == 1:
        return arr.astype(np.int64)

    if arr.ndim == 2:
        if 1 in arr.shape:
            return arr.reshape(-1).astype(np.int64)
        return np.argmax(arr, axis=1).astype(np.int64)

    return arr.reshape(-1).astype(np.int64)


def load_dditr_inference_colors(
    sample_name: str,
    num_points: int,
    pred_root: str | Path = "data/proceedural_generation/inference",
    pred_file: str | None = None,
) -> dict:
    pred_root_path = _resolve_pred_root(pred_root)
    pred_path = _resolve_prediction_path(sample_name=sample_name, pred_root=pred_root_path, pred_file=pred_file)

    if pred_path is None:
        expected_path = pred_root_path / f"{sample_name}_pred.npy"
        return {
            "available": False,
            "reason": f"Missing prediction file: {expected_path}",
            "pred_path": expected_path,
            "pred": None,
            "colors": None,
            "legend_entries": [],
        }

    print(f"[INFO] Loading prediction from: {pred_path}")
    print(f"[INFO] Prediction exists: {pred_path.exists()}")
    try:
        pred = _decode_prediction_array(np.load(pred_path))
    except Exception as e:
        return {
            "available": False,
            "reason": f"Failed to load prediction file {pred_path}: {e}",
            "pred_path": pred_path,
            "pred": None,
            "colors": None,
            "legend_entries": [],
        }
    print(f"[INFO] Prediction loaded: len={len(pred)} dtype={pred.dtype}")

    if pred.shape[0] != num_points:
        return {
            "available": False,
            "reason": f"Prediction length mismatch for {sample_name}. pred={pred.shape[0]} points={num_points} path={pred_path}",
            "pred_path": pred_path,
            "pred": pred,
            "colors": None,
            "legend_entries": [],
        }

    unique_ids = np.unique(pred)
    non_negative_ids = unique_ids[unique_ids >= 0]
    max_id = int(non_negative_ids.max(initial=-1))
    palette = _build_palette(max_id + 1)

    colors = np.zeros((num_points, 3), dtype=np.float32)
    colors[:] = np.array([0.6, 0.6, 0.6], dtype=np.float32)
    valid = (pred >= 0) & (pred <= max_id)
    if np.any(valid):
        colors[valid] = palette[pred[valid]]

    legend_entries = [(f"pred_{int(cid)}", palette[int(cid)]) for cid in non_negative_ids]

    return {
        "available": True,
        "reason": "",
        "pred_path": pred_path,
        "pred": pred,
        "colors": colors,
        "legend_entries": legend_entries,
    }
