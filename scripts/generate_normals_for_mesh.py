import argparse
import logging
from lxml import etree as ET
import numpy as np
import os
import trimesh
import sys

import typing
import numpy as np
from scipy import linalg

import trimesh
import trimesh.remesh
from trimesh.visual.material import SimpleMaterial
import open3d
import IPython
import open3d as o3d
import json
from PIL import Image
from generate_tool_sdf_from_mesh import *

EPS = 10e-10


 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDF from a mesh.")
    parser.add_argument("mesh_file", type=str, help="Path to mesh file.")
    parser.add_argument("--preview", default=False, action="store_true", help="Preview decomp with a trimesh window?")
    parser.add_argument(
        "--scale", type=float, default=1, help="Scale factor to convert the specified mesh's coordinates to meters."
    )
    parser.add_argument("--size", type=float, default=0.15, help="Diameter for the output mesh.")
    parser.add_argument(
        "--density", type=float, default=2, help="Assumed density in kg/m^3 of object, for inertia calculation."
    )
    parser.add_argument(
        "--do_visual_simplification",
        default=False,
        action="store_true",
        help="Do additional visual simplification of mesh. Requires open3d. Probably won't preserve materials.",
    )
    parser.add_argument(
        "--target_tris",
        type=int,
        default=1000,
        help="If we do visual simplification, we decimate to this # of triangles.",
    )
    parser.add_argument("--resolution", type=int, default=100000, help="VHACD voxel resolution.")
    parser.add_argument("--maxhulls", type=int, default=50, help="VHACD max # of convex hulls.")
    parser.add_argument("--minVolumePerCH", type=float, default=0.001, help="VHACD min convex hull volume.")
    parser.add_argument("--maxNumVerticesPerCH", type=int, default=50, help="VHACD voxel resolution.")
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )
    parser.add_argument("--prefix", type=str, default="")
    # manually tuned
    parser.add_argument(
        "--up", type=int, default=-10, help="Scale factor to convert the specified mesh's coordinates to meters."
    )
    parser.add_argument("--forward", type=int, default=-10, help="Diameter for the output mesh.")
    parser.add_argument("--flip_z", default=False, action="store_true", help="Preview decomp with a trimesh window?")
    parser.add_argument(
        "--select_keypoint", default=False, action="store_true", help="Preview decomp with a trimesh window?"
    )
    parser.add_argument("--additional_transform", type=str, default="")
    parser.add_argument("--rename_idx", type=str, default="")
    parser.add_argument("--no_convex_decomp", action="store_true", help="Preview decomp with a trimesh window?")
    # parser.add_argument('--resize', action="store_true", help="Preview decomp with a trimesh window?")

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    input_mesh_path = os.path.abspath(args.mesh_file)

    args.prefix = args.mesh_file.split("/")[2] + "_"  # add prefix
    dir_path, mesh_filename = os.path.split(input_mesh_path)
    mesh_minus_ext, _ = os.path.splitext(mesh_filename)
    sdf_path = os.path.join(dir_path, mesh_minus_ext + ".sdf")

    # Load in and prescale mesh.
    obj = Object3D(input_mesh_path)
    obj.recompute_normals()
    open3d_mesh = obj.meshes[0].as_open3d
    open3d_mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(input_mesh_path.replace(".obj", "_open3d.obj"), open3d_mesh, write_vertex_colors=True)
    obj.meshes[0].export(input_mesh_path, include_texture=True, include_normals=True)
    print("save mesh:", input_mesh_path)
