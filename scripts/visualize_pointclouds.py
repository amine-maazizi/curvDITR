from pathlib import Path
import sys
import csv

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import LegendPanel, suppress_requests_dependency_warning

suppress_requests_dependency_warning()

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

pointcloud_files = sorted(Path("data/goose_ex/pointclouds").glob("*_pcl.bin"))
i = int(sys.argv[1])

if i < 0 or i >= len(pointcloud_files):
    raise SystemExit(f"index must be in [0, {len(pointcloud_files) - 1}]")

bin_path = pointcloud_files[i]
label_path = Path("data/goose_ex/labels") / bin_path.name.replace("_pcl.bin", "_goose.label")
mapping_path = Path("data/goose_ex/goose_label_mapping.csv")

points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
labels = np.fromfile(label_path, dtype=np.uint32)
sem = labels & 0xFFFF

id_to_name = {}
id_to_color = {}
with mapping_path.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        label_id = int(row["label_key"])
        id_to_name[label_id] = row["class_name"]
        hex_color = row["hex"].lstrip("#")
        id_to_color[label_id] = np.array(
            [int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)],
            dtype=np.float32,
        ) / 255.0

point_colors = np.zeros((sem.shape[0], 3), dtype=np.float32)
for label_id, color in id_to_color.items():
    point_colors[sem == label_id] = color

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(point_colors.astype(np.float64))

unique_sem = np.unique(sem)
legend_entries = [
    (id_to_name.get(int(lid), f"id_{int(lid)}"), id_to_color.get(int(lid), np.array([0.6, 0.6, 0.6])))
    for lid in unique_sem
]

app = gui.Application.instance
app.initialize()

window = app.create_window("GOOSE Semantic Viewer", 1280, 800)

scene = gui.SceneWidget()
scene.scene = rendering.Open3DScene(window.renderer)

# dark background
scene.scene.set_background([0.08, 0.08, 0.08, 1.0])

mat = rendering.MaterialRecord()
mat.shader = "defaultUnlit"
mat.point_size = 2.0
scene.scene.add_geometry("point_cloud", pcd, mat)

# XYZ axes
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=2.0, origin=[0.0, 0.0, 0.0]
)
axes_mat = rendering.MaterialRecord()
axes_mat.shader = "defaultUnlit"
scene.scene.add_geometry("axes", axes, axes_mat)

legend_panel = LegendPanel(legend_entries)

bounds = pcd.get_axis_aligned_bounding_box()
scene.setup_camera(60, bounds, bounds.get_center())

window.add_child(scene)
window.add_child(legend_panel.widget)

def on_layout(layout_context):
    r = window.content_rect
    scene.frame = r
    legend_panel.layout(r, layout_context)

window.set_on_layout(on_layout)
app.run()