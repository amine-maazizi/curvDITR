from pathlib import Path
import sys
import logging

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    CurvatureGradientLegendPanel,
    LegendPanel,
    normalize_scalar,
    scalar_to_colors,
    suppress_requests_dependency_warning,
)
from scripts.curavture_maps import compute_roughness_from_points
from scripts.inference_rendering import load_dditr_inference_colors

suppress_requests_dependency_warning()

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


LOGGER = logging.getLogger("procedural_viewer")


def load_procedural_sample(
    sample_index: int,
    data_root: Path = Path("data/proceedural_generation"),
) -> dict:
    pointcloud_files = sorted((data_root / "pointclouds").glob("*_pcl.bin"))
    if not pointcloud_files:
        raise SystemExit(f"No pointcloud files found under: {data_root / 'pointclouds'}")
    if sample_index < 0 or sample_index >= len(pointcloud_files):
        raise SystemExit(f"index must be in [0, {len(pointcloud_files) - 1}]")

    pc_path = pointcloud_files[sample_index]
    sample_name = pc_path.stem.replace("_pcl", "")
    LOGGER.info("Selected procedural sample index=%d file=%s", sample_index, pc_path)
    gt_path = data_root / "labels" / f"{sample_name}.npy"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing GT for {sample_name}: {gt_path}")

    points_xyzi = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    sem = np.load(gt_path).reshape(-1).astype(np.int64)
    if sem.shape[0] != points_xyzi.shape[0]:
        raise ValueError(
            f"GT length mismatch for {sample_name}. "
            f"gt={sem.shape[0]} points={points_xyzi.shape[0]} path={gt_path}"
        )

    id_to_name = {
        0: "terrain",
        1: "vegetation",
    }
    id_to_color = {
        0: np.array([0.70, 0.60, 0.42], dtype=np.float32),
        1: np.array([0.20, 0.68, 0.25], dtype=np.float32),
    }

    legend_entries = [
        (id_to_name.get(int(lid), f"id_{int(lid)}"), id_to_color.get(int(lid), np.array([0.6, 0.6, 0.6])))
        for lid in np.unique(sem)
    ]

    point_semantic_colors = np.zeros((sem.shape[0], 3), dtype=np.float32)
    for label_id, color in id_to_color.items():
        point_semantic_colors[sem == label_id] = color

    mapped_mask = np.isin(sem, np.array(list(id_to_color.keys()), dtype=np.int64)) if id_to_color else np.zeros_like(sem, dtype=bool)
    unmapped_count = int((~mapped_mask).sum())
    if unmapped_count > 0:
        point_semantic_colors[~mapped_mask] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        LOGGER.warning("Unmapped GT labels: %d points. Applying fallback magenta color.", unmapped_count)
    LOGGER.info(
        "GT loaded: points=%d unique_labels=%d mapped=%d unmapped=%d",
        sem.shape[0],
        int(np.unique(sem).size),
        int(mapped_mask.sum()),
        unmapped_count,
    )

    return {
        "pc_path": pc_path,
        "gt_path": gt_path,
        "points": points_xyzi[:, :3].astype(np.float32),
        "point_semantic_colors": point_semantic_colors,
        "legend_entries": legend_entries,
        "sample_name": sample_name,
    }


def build_point_cloud(points_xyz: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def build_surface_mesh(points_xyz: np.ndarray) -> o3d.geometry.TriangleMesh | None:
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    )
    diagonal = float(np.linalg.norm(bbox.get_extent()))
    voxel_size = max(diagonal / 250.0, 1e-3)
    normal_radius = max(voxel_size * 4.0, 1e-3)

    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    surface_pcd = surface_pcd.voxel_down_sample(voxel_size)
    if len(surface_pcd.points) < 200:
        surface_pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))

    surface_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    try:
        surface_pcd.orient_normals_consistent_tangent_plane(30)
    except RuntimeError:
        pass

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(surface_pcd, depth=8)
    mesh = mesh.crop(bbox)
    densities = np.asarray(densities)

    if len(densities) == len(mesh.vertices) and len(densities) > 0:
        keep = densities > np.percentile(densities, 2.0)
        if np.any(keep):
            mesh.remove_vertices_by_mask(~keep)

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0:
        return None

    mesh.compute_vertex_normals()
    return mesh


def transfer_semantics_to_mesh(
    mesh: o3d.geometry.TriangleMesh,
    points_xyz: np.ndarray,
    point_semantic_colors: np.ndarray,
) -> np.ndarray:
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(src_pcd)

    mesh_vertices = np.asarray(mesh.vertices)
    mesh_colors = np.empty((mesh_vertices.shape[0], 3), dtype=np.float32)
    for idx, vertex in enumerate(mesh_vertices):
        _, nn_idx, _ = tree.search_knn_vector_3d(vertex, 1)
        mesh_colors[idx] = point_semantic_colors[int(nn_idx[0])]
    return mesh_colors


def transfer_scalar_to_mesh_knn(
    mesh: o3d.geometry.TriangleMesh,
    points_xyz: np.ndarray,
    point_scalar: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(src_pcd)

    mesh_vertices = np.asarray(mesh.vertices)
    values = np.empty(mesh_vertices.shape[0], dtype=np.float32)
    for idx, vertex in enumerate(mesh_vertices):
        _, nn_idx, nn_d2 = tree.search_knn_vector_3d(vertex, k)
        nn_idx = np.asarray(nn_idx, dtype=np.int64)
        nn_d2 = np.asarray(nn_d2, dtype=np.float64)
        w = 1.0 / (np.sqrt(nn_d2) + 1e-8)
        w /= w.sum()
        values[idx] = float((point_scalar[nn_idx] * w).sum())
    return values


def compute_mesh_curvature_scalar(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    mesh.compute_adjacency_list()
    vertices = np.asarray(mesh.vertices)
    scalar = np.zeros(vertices.shape[0], dtype=np.float32)

    for idx, nbrs in enumerate(mesh.adjacency_list):
        nbrs = list(nbrs)
        if len(nbrs) < 3:
            continue
        nbr_vertices = vertices[np.asarray(nbrs, dtype=np.int64)]
        mean_offset = nbr_vertices.mean(axis=0) - vertices[idx]
        local_scale = np.mean(np.linalg.norm(nbr_vertices - vertices[idx], axis=1)) + 1e-12
        scalar[idx] = float(np.linalg.norm(mean_offset) / local_scale)

    return scalar


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sample_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    pred_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/proceedural_generation/inference")
    sample = load_procedural_sample(sample_index)

    points_xyz = sample["points"]
    point_semantic_colors = sample["point_semantic_colors"]
    sample_name = sample["sample_name"]
    pc_path = sample["pc_path"]
    gt_path = sample["gt_path"]

    pred_path = pred_root / f"{sample_name}_pred.npy"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction for {sample_name}: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT for {sample_name}: {gt_path}")

    points_full = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    print(f"[INFO] Loading prediction from: {pred_path}")
    print(f"[INFO] Prediction exists: {pred_path.exists()}")
    try:
        pred = np.load(pred_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load prediction file {pred_path}: {e}") from e
    gt = np.load(gt_path)

    if pred.ndim == 2 and 1 not in pred.shape:
        pred = np.argmax(pred, axis=1)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    print(f"[INFO] Prediction loaded: len={len(pred)} dtype={pred.dtype}")

    if len(pred) != len(points_full):
        raise ValueError(
            f"Prediction length mismatch for {sample_name}. "
            f"pred={len(pred)} points={len(points_full)} path={pred_path}"
        )
    if len(gt) != len(points_full):
        raise ValueError(
            f"GT length mismatch for {sample_name}. "
            f"gt={len(gt)} points={len(points_full)} path={gt_path}"
        )

    print(f"[INFO] Sample: {sample_name}")
    print(f"[INFO] Points: {len(points_full)}")
    print(f"[INFO] Prediction: {pred_path}")
    print(f"[INFO] GT: {gt_path}")

    inference = load_dditr_inference_colors(
        sample_name=sample_name,
        num_points=points_xyz.shape[0],
        pred_root=pred_root,
        pred_file=str(pred_path),
    )
    if not bool(inference["available"]):
        raise ValueError(inference.get("reason", "Failed to load prediction colors."))
    inference_available = True
    point_inference_colors = inference["colors"]
    if inference_available:
        LOGGER.info("Inference loaded from: %s", inference.get("pred_path"))
    else:
        LOGGER.warning("Inference unavailable: %s", inference.get("reason", "unknown reason"))

    point_roughness = compute_roughness_from_points(points_xyz, k=30)
    point_roughness_norm, point_rough_lo, point_rough_hi = normalize_scalar(point_roughness)
    point_roughness_colors = scalar_to_colors(point_roughness_norm)

    pcd = build_point_cloud(points_xyz, point_semantic_colors)
    mesh = build_surface_mesh(points_xyz)
    mesh_available = mesh is not None

    mesh_semantic_colors = None
    mesh_inference_colors = None
    mesh_scalar_colors = None
    mesh_scalar_name = "curvature"
    mesh_scalar_lo = point_rough_lo
    mesh_scalar_hi = point_rough_hi

    if mesh_available and mesh is not None:
        mesh_semantic_colors = transfer_semantics_to_mesh(mesh, points_xyz, point_semantic_colors)
        if inference_available and point_inference_colors is not None:
            mesh_inference_colors = transfer_semantics_to_mesh(mesh, points_xyz, point_inference_colors)
        mesh_curvature = compute_mesh_curvature_scalar(mesh)
        if float(mesh_curvature.max(initial=0.0)) > 0.0:
            mesh_curv_norm, mesh_scalar_lo, mesh_scalar_hi = normalize_scalar(mesh_curvature)
            mesh_scalar_colors = scalar_to_colors(mesh_curv_norm)
            mesh_scalar_name = "curvature"
        else:
            fallback_scalar = transfer_scalar_to_mesh_knn(mesh, points_xyz, point_roughness, k=3)
            fallback_norm, mesh_scalar_lo, mesh_scalar_hi = normalize_scalar(fallback_scalar)
            mesh_scalar_colors = scalar_to_colors(fallback_norm)
            mesh_scalar_name = "roughness"

    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Procedural Terrain Semantic Viewer", 1280, 800)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    scene.scene.set_background([0.08, 0.08, 0.08, 1.0])

    point_mat = rendering.MaterialRecord()
    point_mat.shader = "defaultUnlit"
    point_mat.point_size = 2.0

    mesh_mat = rendering.MaterialRecord()
    mesh_mat.shader = "defaultLit"
    mesh_mat.base_color = [1.0, 1.0, 1.0, 1.0]

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0.0, 0.0, 0.0])
    axes_mat = rendering.MaterialRecord()
    axes_mat.shader = "defaultUnlit"
    scene.scene.add_geometry("axes", axes, axes_mat)

    legend_panel = LegendPanel(sample["legend_entries"])
    inference_legend_panel = LegendPanel(inference["legend_entries"]) if inference_available else None
    scalar_legend_panel = CurvatureGradientLegendPanel(point_rough_lo, point_rough_hi, steps=24, title="Roughness")

    controls_panel = gui.Vert(4, gui.Margins(8, 8, 8, 8))
    controls_panel.background_color = gui.Color(0.05, 0.05, 0.05, 0.85)
    style_button = gui.Button("Switch to mesh/sdf")
    mode_button = gui.Button("Mode: Semantic")
    pred_toggle_button = gui.Button("Pred labels: OFF")
    status_label = gui.Label("Style: pointcloud | Colors: ground truth")
    status_label.text_color = gui.Color(0.9, 0.9, 0.9, 1.0)
    if inference_available:
        hint_text = "Use buttons to switch style, semantic/curvature, and GT/Pred labels"
    else:
        hint_text = "Use buttons to switch style and semantic/curvature. No D-DITR file found."
    hint_label = gui.Label(hint_text)
    hint_label.text_color = gui.Color(0.9, 0.9, 0.9, 1.0)
    controls_panel.add_child(style_button)
    controls_panel.add_child(mode_button)
    controls_panel.add_child(pred_toggle_button)
    controls_panel.add_child(status_label)
    controls_panel.add_child(hint_label)

    state = {
        "render_style": "pointcloud",
        "color_mode": "semantic",
        "semantic_source": "gt",
        "mesh_available": mesh_available,
    }

    def apply_active_geometry() -> None:
        if state["render_style"] == "mesh/sdf" and not state["mesh_available"]:
            state["render_style"] = "pointcloud"

        try:
            scene.scene.remove_geometry("point_cloud")
        except Exception:
            pass
        try:
            scene.scene.remove_geometry("mesh_surface")
        except Exception:
            pass

        if state["render_style"] == "pointcloud":
            scalar_title = "Roughness"
            scalar_legend_panel.set_range(point_rough_lo, point_rough_hi)
            if state["color_mode"] == "curvature":
                pcd.colors = o3d.utility.Vector3dVector(point_roughness_colors.astype(np.float64))
                LOGGER.info("Rendering curvature colors on pointcloud")
            elif state["semantic_source"] == "inference" and point_inference_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(point_inference_colors.astype(np.float64))
                LOGGER.info("Rendering D-DITR inference colors on pointcloud")
            else:
                pcd.colors = o3d.utility.Vector3dVector(point_semantic_colors.astype(np.float64))
                LOGGER.info("Rendering GT colors on pointcloud")
            scene.scene.add_geometry("point_cloud", pcd, point_mat)
        else:
            scalar_title = mesh_scalar_name.title()
            scalar_legend_panel.set_range(mesh_scalar_lo, mesh_scalar_hi)
            if state["color_mode"] == "curvature" and mesh_scalar_colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_scalar_colors.astype(np.float64))
                LOGGER.info("Rendering curvature colors on mesh")
            elif state["semantic_source"] == "inference" and mesh_inference_colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_inference_colors.astype(np.float64))
                LOGGER.info("Rendering D-DITR inference colors on mesh")
            elif mesh_semantic_colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_semantic_colors.astype(np.float64))
                LOGGER.info("Rendering GT colors on mesh")
            mesh.compute_vertex_normals()
            scene.scene.add_geometry("mesh_surface", mesh, mesh_mat)

        if state["color_mode"] == "curvature":
            scalar_legend_panel.set_title(scalar_title)
        else:
            scalar_legend_panel.set_title("Roughness")

        mode_names = {"semantic": "Semantic", "curvature": "Curvature"}
        mode_button.text = f"Mode: {mode_names[state['color_mode']]}"
        style_button.text = "Switch to pointcloud" if state["render_style"] == "mesh/sdf" else "Switch to mesh/sdf"
        pred_toggle_button.text = "Pred labels: ON" if state["semantic_source"] == "inference" else "Pred labels: OFF"

        if state["color_mode"] == "curvature":
            status_colors = mesh_scalar_name if state["render_style"] == "mesh/sdf" else "roughness"
        else:
            status_colors = "D-DITR inference" if state["semantic_source"] == "inference" else "ground truth"
        status_label.text = f"Style: {state['render_style']} | Colors: {status_colors}"
        LOGGER.info(
            "Active mode => style=%s color=%s semantic_source=%s",
            state["render_style"],
            state["color_mode"],
            state["semantic_source"],
        )

        legend_panel.widget.visible = state["color_mode"] == "semantic" and state["semantic_source"] == "gt"
        scalar_legend_panel.widget.visible = state["color_mode"] == "curvature"
        if inference_legend_panel is not None:
            inference_legend_panel.widget.visible = (
                state["color_mode"] == "semantic" and state["semantic_source"] == "inference"
            )

    def on_cycle_mode() -> None:
        state["color_mode"] = "curvature" if state["color_mode"] == "semantic" else "semantic"
        apply_active_geometry()

    def on_toggle_pred_labels() -> None:
        if not inference_available:
            return
        state["semantic_source"] = "inference" if state["semantic_source"] == "gt" else "gt"
        apply_active_geometry()

    def on_toggle_style() -> None:
        if state["render_style"] == "pointcloud":
            state["render_style"] = "mesh/sdf"
        else:
            state["render_style"] = "pointcloud"
        apply_active_geometry()

    mode_button.set_on_clicked(on_cycle_mode)
    pred_toggle_button.set_on_clicked(on_toggle_pred_labels)
    style_button.set_on_clicked(on_toggle_style)

    bounds = pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())

    window.add_child(scene)
    window.add_child(legend_panel.widget)
    window.add_child(scalar_legend_panel.widget)
    if inference_legend_panel is not None:
        window.add_child(inference_legend_panel.widget)
    window.add_child(controls_panel)

    apply_active_geometry()

    def on_layout(layout_context):
        r = window.content_rect
        scene.frame = r
        legend_panel.layout(r, layout_context)
        scalar_legend_panel.layout(r, layout_context)
        if inference_legend_panel is not None:
            inference_legend_panel.layout(r, layout_context)

        pref = controls_panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
        controls_panel.frame = gui.Rect(r.x + 10, r.y + r.height - pref.height - 10, pref.width, pref.height)

    window.set_on_layout(on_layout)
    app.run()


if __name__ == "__main__":
    main()