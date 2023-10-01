import os
import glob
import argparse

import numpy as np
import cv2

import open3d as o3d
import os
import IPython
import json


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


parser = argparse.ArgumentParser()
parser.add_argument("--mesh_file", "-m", type=str)

# parser.add_argument('--write', required=True)
args = parser.parse_args()

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
mesh = o3d.io.read_triangle_mesh(args.mesh_file)
# o3d.visualization.draw_geometries([mesh])

pcd = mesh.sample_points_uniformly(number_of_points=2000)
pcd.paint_uniform_color([0.5, 0.5, 0.5])
# o3d.visualization.draw_geometries([pcd])

meshes = [pcd]
keypoint_info = json.load(open(args.mesh_file.replace("_corr.obj", "_info.json")))
tool_keypoints = keypoint_info["keypoints"]
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
for idx in range(len(tool_keypoints)):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
    mesh_sphere.translate(tool_keypoints[idx])
    mesh_sphere.paint_uniform_color(colors[idx])
    meshes.append(mesh_sphere)


viewer = o3d.visualization.Visualizer()
viewer.create_window()
print("meshes:", len(meshes))
for m in meshes:
    viewer.add_geometry(m)
opt = viewer.get_render_option()
opt.show_coordinate_frame = True
viewer.run()
viewer.destroy_window()
