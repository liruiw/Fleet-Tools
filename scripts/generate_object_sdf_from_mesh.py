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
from generate_tool_sdf_from_mesh import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDF from a mesh.")
    parser.add_argument("mesh_file", type=str, help="Path to mesh file.")
    parser.add_argument("--preview", default=False, action="store_true", help="Preview decomp with a trimesh window?")
    parser.add_argument(
        "--scale", type=float, default=1, help="Scale factor to convert the specified mesh's coordinates to meters."
    )
    parser.add_argument("--size", type=float, default=0.1, help="Diameter for the output mesh.")
    parser.add_argument(
        "--density", type=float, default=200, help="Assumed density in kg/m^3 of object, for inertia calculation."
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

    parser.add_argument(
        "--no_resize",
        default=False,
        action="store_true",
        help="Avoid resizing?",
    )
    parser.add_argument(
        "--no_recenter",
        default=False,
        action="store_true",
        help="Avoid recentering?",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    mesh_path = os.path.abspath(args.mesh_file)

    if len(args.rename_idx) > 0:
        mesh_path = rename_model_normalized_folder(args.mesh_file, args.mesh_file.split("/")[-1], int(args.rename_idx))

    if not os.path.exists(mesh_path):
        logging.error("No mesh found at %s" % mesh_path)
        sys.exit(-1)

    gen_sdf_func = create_sdf_with_mesh if args.no_convex_decomp else create_sdf_with_convex_decomp
    obj = gen_sdf_func(
        mesh_path,
        scale=args.scale,
        size=args.size,
        do_visual_simplification=args.do_visual_simplification,
        target_tris=args.target_tris,
        preview_with_trimesh=args.preview,
        density=args.density,
        resolution=args.resolution,
        maxhulls=args.maxhulls,
        maxNumVerticesPerCH=args.maxNumVerticesPerCH,
        minVolumePerCH=args.minVolumePerCH,
        pca=1,
        prefix=args.prefix,
        flip_z=args.flip_z,
        additional_transform=args.additional_transform,
        resize=not args.no_resize,
        recenter=not args.no_recenter,
    )
    input_mesh_path = mesh_path.replace(".obj", "_corr.obj")
    obj = Object3D(input_mesh_path)
    obj.recompute_normals()
    open3d_mesh = obj.meshes[0].as_open3d
    open3d_mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(input_mesh_path.replace(".obj", "_open3d.obj"), open3d_mesh, write_vertex_colors=True)
    obj.meshes[0].export(input_mesh_path, include_texture=True, include_normals=True)

    if args.select_keypoint:
        mesh = o3d.io.read_triangle_mesh(input_mesh_path)
        pcd = mesh.sample_points_uniformly(number_of_points=10000)

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")

        # vis = o3d.visualization.draw_geometries_with_editing([pcd])
        picked_points = vis.get_picked_points()
        picked_points = np.unique(picked_points)
        xyz = np.asarray(pcd.points)
        picked_points = xyz[picked_points]

        pointcloudfile = mesh_path.replace(".obj", "_points.npy")
        np.save(pointcloudfile, xyz)
        keypoint_description_file = mesh_path.replace(".obj", "_info.json")
        keypoint_info = {"keypoints": picked_points.tolist(), "extents": obj.extents.tolist()}
        with open(keypoint_description_file, "w") as f:
            json.dump(keypoint_info, f, indent=4, sort_keys=True)
