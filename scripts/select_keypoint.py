import os
import glob
import argparse

# from colmap_depth import extract_all_depths
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
parser.add_argument("--dir", type=str)
parser.add_argument("--mesh_file", type=str)

# parser.add_argument('--write', required=True)
args = parser.parse_args()
datapath = args.dir

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
mesh = o3d.io.read_triangle_mesh(args.mesh_file)
pcd = mesh.sample_points_uniformly(number_of_points=50000)


viewer = o3d.visualization.VisualizerWithEditing()
viewer.create_window()
viewer.add_geometry(pcd)
opt = viewer.get_render_option()
opt.show_coordinate_frame = True
# opt.background_color = np.asarray([0.5, 0.5, 0.5])
viewer.run()
viewer.destroy_window()
