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

EPS = 10e-10


def compute_vertex_normals(vertices, faces):
    normals = np.ones_like(vertices)
    triangles = vertices[faces]
    triangle_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    triangle_normals /= linalg.norm(triangle_normals, axis=1)[:, None] + EPS
    normals[faces[:, 0]] += triangle_normals
    normals[faces[:, 1]] += triangle_normals
    normals[faces[:, 2]] += triangle_normals
    normals /= linalg.norm(normals, axis=1)[:, None] + 0
    return normals


def are_trimesh_normals_corrupt(trimesh):
    corrupt_normals = linalg.norm(trimesh.vertex_normals, axis=1) == 0.0
    return corrupt_normals.sum() > 0


def subdivide_mesh(mesh):
    attributes = {}
    if hasattr(mesh.visual, "uv"):
        attributes = {"uv": mesh.visual.uv}
    vertices, faces, attributes = trimesh.remesh.subdivide(mesh.vertices, mesh.faces, attributes=attributes)
    mesh.vertices = vertices
    mesh.faces = faces
    if "uv" in attributes:
        mesh.visual.uv = attributes["uv"]
    return mesh


class Object3D(object):
    """Represents a graspable object."""

    def __init__(self, path, load_materials=False):
        scene = trimesh.load(
            str(path),
            ignore_mtl=not load_materials,
            skip_materials=True,
            force="mesh",
            default_material=SimpleMaterial(color=(1.0, 0.0, 1.0)),
        )
        # if isinstance(scene, trimesh.Trimesh):
        #     scene = trimesh.Scene(scene)

        self.meshes: typing.List[trimesh.Trimesh] = [scene]  # list(scene.dump())
        self.recompute_normals()
        self.path = path
        self.scale = 1.0
        # self.fix()

    def fix(self):
        for idx, mesh in enumerate(self.meshes):
            trimesh.repair.fill_holes(self.meshes[idx])
            trimesh.repair.fix_inversion(self.meshes[idx])
            trimesh.repair.fix_normals(self.meshes[idx])
            self.meshes[idx].remove_unreferenced_vertices()
            self.meshes[idx].remove_degenerate_faces()
            self.meshes[idx].remove_duplicate_faces()

    def to_scene(self):
        return trimesh.Scene(self.meshes)

    def are_normals_corrupt(self):
        for mesh in self.meshes:
            if are_trimesh_normals_corrupt(mesh):
                return True

        return False

    def recompute_normals(self):
        for mesh in self.meshes:
            mesh.vertex_normals = compute_vertex_normals(mesh.vertices, mesh.faces)

        return self

    def rescale(self, scale=1.0):
        """Set scale of object mesh.
        :param scale
        """
        self.scale = scale
        for mesh in self.meshes:
            mesh.apply_scale(self.scale)

        return self

    def resize(self, size, ref="diameter"):
        """Set longest of all three lengths in Cartesian space.
        :param size
        """
        if ref == "diameter":
            ref_scale = self.bounding_diameter
        else:
            ref_scale = self.bounding_size

        self.scale = size / ref_scale
        for mesh in self.meshes:
            mesh.apply_scale(self.scale)

        return self

    @property
    def bounding_size(self):
        return max(self.extents)

    @property
    def bounding_diameter(self):
        centroid = self.bounds.mean(axis=0)
        max_radius = linalg.norm(self.vertices - centroid, axis=1).max()
        return max_radius * 2

    @property
    def bounding_radius(self):
        return self.bounding_diameter / 2.0

    @property
    def extents(self):
        min_dim = np.min(self.vertices, axis=0)
        max_dim = np.max(self.vertices, axis=0)
        return max_dim - min_dim

    @property
    def bounds(self):
        min_dim = np.min(self.vertices, axis=0)
        max_dim = np.max(self.vertices, axis=0)
        return np.stack((min_dim, max_dim), axis=0)

    def recenter(self, method="bounds"):
        if method == "mean":
            # Center the mesh.
            vertex_mean = np.mean(self.vertices, 0)
            translation = -vertex_mean
        elif method == "bounds":
            center = self.bounds.mean(axis=0)
            translation = -center
        else:
            raise ValueError(f"Unknown method {method!r}")

        for mesh in self.meshes:
            mesh.apply_translation(translation)

        return self

    def normalize_direction(self, flip_z=False):
        # make sure the longest dimension facing upward (+z) and the second longest dimension points at positive x

        up = np.zeros(3)
        forward = np.zeros(3)

        sorted_idx = np.argsort(self.extents)
        up[sorted_idx[-1]] = 1
        if flip_z:
            up[sorted_idx[-1]] = -1

        forward[sorted_idx[-2]] = 1

        side = np.cross(forward, up)
        # R = np.stack([side, up, -forward], axis=-1)
        R = np.stack([forward, side, up], axis=-1).T  # transpose
        transform = np.eye(4)
        transform[:3, :3] = R
        print("apply rotation", R)
        self.meshes[0] = self.meshes[0].apply_transform(transform)
        return self

    def apply_transform(self, trans):
        # make sure the longest dimension facing upward (+z) and the second longest dimension points at positive x

        self.meshes[0] = self.meshes[0].apply_transform(trans)
        return self

    @property
    def vertices(self):
        return np.concatenate([mesh.vertices for mesh in self.meshes])


def do_visual_mesh_simplification(input_mesh_path, target_tris=1000):
    """
    Given an obj path, simplifies the geometry to make it easier to render
    by creating a mesh alongside it with a "_simple_vis.<ext>" postfix.
    - Looks for a texture file at a fixed relative path

    Args:
    - input_mesh_path: String path to mesh file. Only 'obj' format is tested,
        but others might work.
    - target_tris: Currently unusued, but would be target for mesh decimation.
    Returns:
    - output obj file path
    """
    import open3d
    import cv2
    import imutils

    # TODO(gizatt) What gives, open3d is trashing my models...
    logging.warning("As of writing, this is sometimes creating terrible models.")

    mesh_minus_ext, mesh_ext = os.path.splitext(input_mesh_path)
    output_mesh_path = mesh_minus_ext + "_simple_vis" + mesh_ext

    mesh = open3d.io.read_triangle_mesh(input_mesh_path)
    mesh.compute_vertex_normals()
    simplified_mesh = mesh.simplify_quadric_decimation(target_tris)
    simplified_mesh.compute_vertex_normals()
    open3d.io.write_triangle_mesh(output_mesh_path, simplified_mesh)
    return output_mesh_path


def do_collision_mesh_simplification_coacd(mesh, mesh_name, mesh_dir, preview_with_trimesh=False, **kwargs):
    """
    Given a mesh, performs a convex decomposition of it with
    coacd, saving all the parts in a subfolder named
    `<mesh_filename>_parts`.

    Args:
    - input_mesh_path: String path to mesh file to decompose. Only
        'obj' format is currently tested, but other formats supported
        by trimesh might work.
    - preview_with_trimesh: Whether to open (and block on) a window to preview
    the decomposition.
    - A set of control kwargs, plus any additional kwargs, are passed to the convex
      decomposition routine 'vhacd'; you can run `testVHACD --help` to see options.

    Returns:
    - List of generated mesh file parts, in obj format.
    """
    # IPython.embed()

    # Create a subdir for the convex decomp parts.

    mesh_parts_folder = mesh_name + "_parts"
    out_dir = os.path.join(mesh_dir, mesh_parts_folder)
    mesh_full_path = os.path.join(mesh_dir, mesh_name)
    os.makedirs(out_dir, exist_ok=True)

    if len(os.listdir(out_dir)) > 0:
        print("WARNING: {} is not empty".format(out_dir))
        d = input("** DELETE? [y/n]")
        if d == 'y':
            for fn in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, fn))
                print("\tRemoved {}...".format(fn))
        else:
            raise ValueError("{} should be empty".format(out_dir))

    if preview_with_trimesh:
        logging.info("Showing mesh before decomp. Close window to proceed.")
        # mesh.set_camera(angles=(1, 0, 0), distance=1, center=(0,0,0))
        # mesh.show(angles=(0, 1, 0), distance=1, center=(0,0,0))
        scene = trimesh.scene.scene.Scene()
        scene.add_geometry(mesh)
        scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
        # scene.viewer.toggle_axis()
        scene.show()
    try:
        convex_pieces = []
        os.system(f"coacd -i {mesh_full_path}_corr.obj -o {out_dir}")
        logging.info("Performing convex decomposition with CoACD.")

        for file in sorted(os.listdir(out_dir)):
            part = trimesh.load(os.path.join(out_dir, file))
            convex_pieces += [part]

    except Exception as e:
        logging.error("Problem performing decomposition: %s", e)

    if preview_with_trimesh:
        # Display the convex decomp, giving each a random colors
        # to make them easier to distinguish.
        for part in convex_pieces:
            this_color = trimesh.visual.random_color()
            part.visual.face_colors[:] = this_color
        scene = trimesh.scene.scene.Scene()
        for part in convex_pieces:
            scene.add_geometry(part)

        logging.info("Showing mesh convex decomp into %d parts. Close window to proceed." % (len(convex_pieces)))
        scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))

        # viewer = trimesh.viewer.SceneViewer(scene)
        # viewer.init_gl()
        # viewer.on_draw()
        scene.show()

    # rewrite the mesh with new names
    os.system(f"rm {out_dir}/*")
    out_paths = []
    for k, part in enumerate(convex_pieces):
        piece_name = "%s_convex_piece_%03d.obj" % (mesh_name, k)
        full_path = os.path.join(out_dir, piece_name)

        trimesh.exchange.export.export_mesh(part, full_path)
        out_paths.append(full_path)
    return out_paths


def create_sdf_with_mesh(
    input_mesh_path,
    scale=1.0,
    do_visual_simplification=False,
    size=0.1,
    target_tris=1000,
    preview_with_trimesh=False,
    density=2000,
    recenter=True,
    resize=True,
    bound_type="extents",
    prefix="",
    flip_z=False,
    additional_transform="",
    **kwargs,
):
    """
    Given an input mesh file, produces an SDF if the same directory that:
    - Uses the mesh as its sole visual geometry.
    - Performs a convex decomposition of the mesh, and uses those pieces
    as the collision geometry.
    - Inserts inertia for the object, calculated from the original
    mesh assuming a constant density.

    The SDF is saved as `<mesh_file>.sdf` next to the mesh file.

    Args:
    - input_mesh_path: Path to the mesh file
    - preview_with_trimesh: Whether to show 3D previews of pre/post decomposition.
    - density: Assumed density of the object, in kg/m^3.
    - kwargs: Passed through to do_collision_mesh_simplification as convex decomp args.
    """
    # Get mesh name.
    dir_path, mesh_filename = os.path.split(input_mesh_path)
    mesh_minus_ext, _ = os.path.splitext(mesh_filename)
    sdf_path = os.path.join(dir_path, mesh_minus_ext + ".sdf")

    # Load in and prescale mesh.
    obj = Object3D(input_mesh_path)  # trimesh.load(input_mesh_path, skip_materials=True, force="mesh")

    if resize and size > 0:
        if bound_type == "diameter":
            object_scale = size / obj.bounding_diameter
        elif bound_type == "extents":
            object_scale = size / obj.bounding_size
        else:
            raise ValueError(f"Unkown size_type {bound_type!r}")

        obj.rescale(object_scale)

    if recenter:
        obj.recenter("bounds")
    # scale = object_scale
    mesh = obj.meshes[0]
    input_mesh_path = input_mesh_path.replace(".obj", "_corr.obj")
    mesh = mesh.process(validate=True)
    # add texture

    if not hasattr(mesh.visual, "uv"):
        # load template mesh and assign materials
        # template_mesh = trimesh.load("assets/template_mesh/textured_simple.obj")
        from trimesh.visual import TextureVisuals  # template_mesh.visual #

        # assign random texture coordinates
        open3d_mesh = mesh.as_open3d
        open3d_mesh.paint_uniform_color((0.8, 0.8, 0.8))
        v_uv = np.random.rand(len(open3d_mesh.triangles) * 3, 2)
        open3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
        o3d.io.write_triangle_mesh(input_mesh_path, open3d_mesh)
    else:
        mesh.export(input_mesh_path, include_texture=True)

    # old version that writes with trimesh
    # Generate SDF file and the robot and link elements.
    robot_name = prefix + mesh_minus_ext
    root_item = ET.Element("sdf", version="1.5", nsmap={"drake": "drake.mit.edu"})
    model_item = ET.SubElement(root_item, "model", name=robot_name)
    link_name = "{}_body_link".format(robot_name)
    link_item = ET.SubElement(model_item, "link", name=link_name)
    pose_item = ET.SubElement(link_item, "pose")
    pose_item.text = "0 0 0 0 0 0"

    # Set up object inertia.
    mass, I = calc_mesh_inertia(mesh, density=density)
    inertial_item = ET.SubElement(link_item, "inertial")
    mass_item = ET.SubElement(inertial_item, "mass")
    mass_item.text = "{:.4E}".format(mass)
    inertia_item = ET.SubElement(inertial_item, "inertia")
    for i in range(3):
        for j in range(i, 3):
            item = ET.SubElement(inertia_item, "i" + "xyz"[i] + "xyz"[j])
            item.text = "{:.4E}".format(I[i, j])

    # Set up object visual geometry.
    visual_mesh_filename = input_mesh_path
    logging.warn("%s -> %s", visual_mesh_filename, sdf_path)
    visual_mesh_filename = os.path.relpath(visual_mesh_filename, dir_path)
    visual_item = ET.SubElement(link_item, "visual", name="visual")
    geometry_item = ET.SubElement(visual_item, "geometry")
    mesh_item = ET.SubElement(geometry_item, "mesh")
    uri_item = ET.SubElement(mesh_item, "uri")
    uri_item.text = visual_mesh_filename
    scale_item = ET.SubElement(mesh_item, "scale")
    scale_item.text = "{:.4E} {:.4E} {:.4E}".format(scale, scale, scale)

    # Set up object collision geometry.
    collision_paths = [input_mesh_path]
    for k, new_model_path in enumerate(collision_paths):
        new_model_path = os.path.relpath(new_model_path, dir_path)
        # Create a new XML subtree for the collection of meshes
        # we just created. I *think* each convex piece needs
        # to be in its own collision tag, otherwise Drake
        # seems to be ignoring them...
        collision_item = ET.SubElement(link_item, "collision", name="collision_%04d" % k)
        geometry_item = ET.SubElement(collision_item, "geometry")
        mesh_item = ET.SubElement(geometry_item, "mesh")
        uri_item = ET.SubElement(mesh_item, "uri")
        uri_item.text = new_model_path

    logging.info("Writing SDF to %s" % sdf_path)
    ET.ElementTree(root_item).write(sdf_path, pretty_print=True)
    return obj


def do_collision_mesh_simplification(mesh, mesh_name, mesh_dir, preview_with_trimesh=False, **kwargs):
    """
    Given a mesh, performs a convex decomposition of it with
    trimesh _ vhacd, saving all the parts in a subfolder named
    `<mesh_filename>_parts`.

    Args:
    - input_mesh_path: String path to mesh file to decompose. Only
        'obj' format is currently tested, but other formats supported
        by trimesh might work.
    - preview_with_trimesh: Whether to open (and block on) a window to preview
    the decomposition.
    - A set of control kwargs, plus any additional kwargs, are passed to the convex
      decomposition routine 'vhacd'; you can run `testVHACD --help` to see options.

    Returns:
    - List of generated mesh file parts, in obj format.
    """

    # Create a subdir for the convex decomp parts.
    mesh_parts_folder = mesh_name + "_parts"
    out_dir = os.path.join(mesh_dir, mesh_parts_folder)
    os.makedirs(out_dir, exist_ok=True)

    if preview_with_trimesh:
        logging.info("Showing mesh before decomp. Close window to proceed.")
        # mesh.set_camera(angles=(1, 0, 0), distance=1, center=(0,0,0))
        # mesh.show(angles=(0, 1, 0), distance=1, center=(0,0,0))
        scene = trimesh.scene.scene.Scene()
        scene.add_geometry(mesh)
        scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))
        # scene.viewer.toggle_axis()
        scene.show()
    try:
        convex_pieces = []
        logging.info("Performing convex decomposition. If this runs too long, try decreasing --resolution.")
        convex_pieces_new = trimesh.decomposition.convex_decomposition(mesh, **kwargs)
        if not isinstance(convex_pieces_new, list):
            convex_pieces_new = [convex_pieces_new]
        convex_pieces += convex_pieces_new
    except Exception as e:
        logging.error("Problem performing decomposition: %s", e)

    if preview_with_trimesh:
        # Display the convex decomp, giving each a random colors
        # to make them easier to distinguish.
        for part in convex_pieces:
            this_color = trimesh.visual.random_color()
            part.visual.face_colors[:] = this_color
        scene = trimesh.scene.scene.Scene()
        for part in convex_pieces:
            scene.add_geometry(part)

        logging.info("Showing mesh convex decomp into %d parts. Close window to proceed." % (len(convex_pieces)))
        scene.set_camera(angles=(1, 0, 0), distance=0.3, center=(0, 0, 0))

        # viewer = trimesh.viewer.SceneViewer(scene)
        # viewer.init_gl()
        # viewer.on_draw()
        scene.show()

    out_paths = []
    for k, part in enumerate(convex_pieces):
        piece_name = "%s_convex_piece_%03d.obj" % (mesh_name, k)
        full_path = os.path.join(out_dir, piece_name)
        trimesh.exchange.export.export_mesh(part, full_path)
        out_paths.append(full_path)
    return out_paths


def calc_mesh_inertia(mesh, density=2000):
    """
    Given a mesh, calculates its total mass and inertia assuming
    a fixed density.
    Args:
    - mesh: A trimesh mesh.
    - density: Density of object in kg/m^3, used for inertia calculation.
    Returns: (mass, inertia)
    - out_paths: List of generated mesh file parts, in obj format.
    - inertia: total inertia of the input mesh.
    """
    mesh.density = density
    I = mesh.moment_inertia
    return mesh.mass, mesh.moment_inertia


def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX


def create_sdf_with_convex_decomp(
    input_mesh_path,
    scale=1.0,
    do_visual_simplification=False,
    size=0.1,
    target_tris=1000,
    preview_with_trimesh=False,
    density=2000,
    recenter=True,
    resize=True,
    bound_type="extents",
    prefix="",
    flip_z=False,
    additional_transform="",
    **kwargs,
):
    """
    Given an input mesh file, produces an SDF if the same directory that:
    - Uses the mesh as its sole visual geometry.
    - Performs a convex decomposition of the mesh, and uses those pieces
    as the collision geometry.
    - Inserts inertia for the object, calculated from the original
    mesh assuming a constant density.

    The SDF is saved as `<mesh_file>.sdf` next to the mesh file.

    Args:
    - input_mesh_path: Path to the mesh file
    - preview_with_trimesh: Whether to show 3D previews of pre/post decomposition.
    - density: Assumed density of the object, in kg/m^3.
    - kwargs: Passed through to do_collision_mesh_simplification as convex decomp args.
    """
    # Get mesh name.
    dir_path, mesh_filename = os.path.split(input_mesh_path)
    mesh_minus_ext, _ = os.path.splitext(mesh_filename)
    sdf_path = os.path.join(dir_path, mesh_minus_ext + ".sdf")

    # Load in and prescale mesh.
    obj = Object3D(input_mesh_path)  # trimesh.load(input_mesh_path, skip_materials=True, force="mesh")

    if resize and size > 0:
        if bound_type == "diameter":
            object_scale = size / obj.bounding_diameter
        elif bound_type == "extents":
            object_scale = size / obj.bounding_size
        else:
            raise ValueError(f"Unkown size_type {bound_type!r}")
        if scale != 1:
            object_scale = scale
        obj.rescale(object_scale)
    # obj.normalize_direction(flip_z)
    if recenter:
        obj.recenter("bounds")

    # one off rotate
    if len(additional_transform) > 0:
        obj.apply_transform(eval(additional_transform)(np.pi / 2))

    if flip_z:
        obj.apply_transform(rotX(np.pi))

    #
    mesh = obj.meshes[0]
    input_mesh_path = input_mesh_path.replace(".obj", "_corr.obj")
    mesh = mesh.process(validate=True)
    # trimesh.exchange.export.export_obj(mesh, input_mesh_path, include_texture=True)
    # add texture
    # IPython.embed()
    # if not hasattr(mesh.visual, 'uv'):
    # load template mesh and assign materials
    # template_mesh = trimesh.load("assets/template_mesh/textured_simple.obj")
    from trimesh.visual import TextureVisuals  # template_mesh.visual #

    # assign random texture coordinates
    open3d_mesh = mesh.as_open3d
    open3d_mesh.paint_uniform_color((0.8, 0.8, 0.8))
    v_uv = np.random.rand(len(open3d_mesh.triangles) * 3, 2)
    open3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
    o3d.io.write_triangle_mesh(input_mesh_path, open3d_mesh)
    # else:
    #     mesh.export(input_mesh_path, include_texture=True)

    # old version that writes with trimesh
    # Generate SDF file and the robot and link elements.
    robot_name = prefix + mesh_minus_ext
    root_item = ET.Element("sdf", version="1.5", nsmap={"drake": "drake.mit.edu"})
    model_item = ET.SubElement(root_item, "model", name=robot_name)
    link_name = "{}_body_link".format(robot_name)
    link_item = ET.SubElement(model_item, "link", name=link_name)
    pose_item = ET.SubElement(link_item, "pose")
    pose_item.text = "0 0 0 0 0 0"

    # Set up object inertia.
    mass, I = calc_mesh_inertia(mesh, density=density)
    # all 0s for off diagonal
    I[0, 1] = I[0, 2] = I[1, 0] = I[1, 2] = I[2, 0] = I[2, 1] = 0
    I = np.abs(I)  # all positive diagonal
    inertial_item = ET.SubElement(link_item, "inertial")
    mass_item = ET.SubElement(inertial_item, "mass")
    mass_item.text = "{:.4E}".format(mass)
    inertia_item = ET.SubElement(inertial_item, "inertia")
    for i in range(3):
        for j in range(i, 3):
            item = ET.SubElement(inertia_item, "i" + "xyz"[i] + "xyz"[j])
            item.text = "{:.4E}".format(I[i, j])

    # Set up object visual geometry.
    if do_visual_simplification:
        visual_mesh_filename = do_visual_mesh_simplification(input_mesh_path, target_tris=target_tris)
    else:
        visual_mesh_filename = input_mesh_path

    logging.warn("%s -> %s", visual_mesh_filename, sdf_path)
    visual_mesh_filename = os.path.relpath(visual_mesh_filename, dir_path)
    visual_item = ET.SubElement(link_item, "visual", name="visual")
    geometry_item = ET.SubElement(visual_item, "geometry")
    mesh_item = ET.SubElement(geometry_item, "mesh")
    uri_item = ET.SubElement(mesh_item, "uri")
    uri_item.text = visual_mesh_filename
    # scale_item = ET.SubElement(mesh_item, "scale") # don't need the scale, already applied
    # scale_item.text = "{:.4E} {:.4E} {:.4E}".format(scale, scale, scale)

    # Set up object collision geometry.
    # do_collision_mesh_simplification
    collision_paths = do_collision_mesh_simplification_coacd(
        mesh,
        mesh_name=mesh_minus_ext,
        mesh_dir=dir_path,
        preview_with_trimesh=preview_with_trimesh,
        **kwargs,
    )
    for k, new_model_path in enumerate(collision_paths):
        new_model_path = os.path.relpath(new_model_path, dir_path)
        collision_item = ET.SubElement(link_item, "collision", name="collision_%04d" % k)
        geometry_item = ET.SubElement(collision_item, "geometry")
        mesh_item = ET.SubElement(geometry_item, "mesh")
        uri_item = ET.SubElement(mesh_item, "uri")
        uri_item.text = new_model_path
        ET.SubElement(mesh_item, "{drake.mit.edu}declare_convex")

    logging.info("Writing SDF to %s" % sdf_path)
    ET.ElementTree(root_item).write(sdf_path, pretty_print=True)
    return obj


def rename_model_normalized_folder(path, obj_name, idx):
    print(
        "rename object folder:",
        f"{path}/model_normalized.obj",
        os.path.exists(f"{path}/model_normalized.obj"),
    )
    if os.path.exists(f"{path}/model_normalized.obj"):
        new_model_path = path.replace(obj_name, f"model{idx}")  #
        os.system(f"mkdir {new_model_path}  ")
        os.system(f"cp {path}/model_normalized.obj {new_model_path}/model{idx}.obj ")  # {path}{obj_name}.obj
        # better rename it as modelx
        # os.system(f'rm {path}/model_normalized* ')
        # print(new_model_path)
        return f"{new_model_path}/model{idx}.obj"
    return f"{path}/{obj_name}.obj"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDF from a mesh.")
    parser.add_argument("mesh_file", type=str, help="Path to mesh file.")
    parser.add_argument(
        "--preview",
        default=False,
        action="store_true",
        help="Preview decomp with a trimesh window?",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="Scale factor to convert the specified mesh's coordinates to meters.",
    )
    parser.add_argument("--size", type=float, default=0.15, help="Diameter for the output mesh.")
    parser.add_argument(
        "--density",
        type=float,
        default=2,
        help="Assumed density in kg/m^3 of object, for inertia calculation.",
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
    parser.add_argument("--maxhulls", type=int, default=500, help="VHACD max # of convex hulls.")
    parser.add_argument(
        "--minVolumePerCH",
        type=float,
        default=0.001,
        help="VHACD min convex hull volume.",
    )
    parser.add_argument("--maxNumVerticesPerCH", type=int, default=500, help="VHACD voxel resolution.")
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
        "--up",
        type=int,
        default=-10,
        help="Scale factor to convert the specified mesh's coordinates to meters.",
    )
    parser.add_argument("--forward", type=int, default=-10, help="Diameter for the output mesh.")
    parser.add_argument(
        "--flip_z",
        default=False,
        action="store_true",
        help="Preview decomp with a trimesh window?",
    )
    parser.add_argument(
        "--select_keypoint",
        default=False,
        action="store_true",
        help="Preview decomp with a trimesh window?",
    )
    parser.add_argument("--additional_transform", type=str, default="")
    parser.add_argument("--rename_idx", type=str, default="")
    parser.add_argument(
        "--no_convex_decomp",
        action="store_true",
        help="Preview decomp with a trimesh window?",
    )
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
    # parser.add_argument('--resize', action="store_true", help="Preview decomp with a trimesh window?")

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    mesh_path = os.path.abspath(args.mesh_file)

    if len(args.rename_idx) > 0:
        mesh_path = rename_model_normalized_folder(args.mesh_file, args.mesh_file.split("/")[-1], int(args.rename_idx))

    args.prefix = args.mesh_file.split("/")[2] + "_"  # add prefix
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


    # if args.preview and not args.no_convex_decomp:
    #     # remaining one degree of freedom for the heuristicsq
    #     import cv2

    #     cv2.imshow("flip?", np.zeros((64, 64, 3)))
    #     k = cv2.waitKey(0)

    #     if k == ord("f"):
    #         args.flip_z = True
    #         print("flip")
    #         obj = create_sdf_with_convex_decomp(
    #             mesh_path,
    #             scale=args.scale,
    #             size=args.size,
    #             do_visual_simplification=args.do_visual_simplification,
    #             target_tris=args.target_tris,
    #             preview_with_trimesh=False,
    #             density=args.density,
    #             resolution=args.resolution,
    #             maxhulls=args.maxhulls,
    #             maxNumVerticesPerCH=args.maxNumVerticesPerCH,
    #             minVolumePerCH=args.minVolumePerCH,
    #             pca=1,
    #             prefix=args.prefix,
    #             flip_z=args.flip_z,
    #             additional_transform=args.additional_transform,
    #         )

    if args.select_keypoint:
        mesh = o3d.io.read_triangle_mesh(mesh_path.replace(".obj", "_corr.obj"))
        pcd = mesh.sample_points_uniformly(number_of_points=10000)

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")

        # vis = o3d.visualization.draw_geometries_with_editing([pcd])
        picked_points = vis.get_picked_points()
        xyz = np.asarray(pcd.points)

        picked_points = xyz[picked_points]
        # save as json  try two-point method
        pointcloudfile = mesh_path.replace(".obj", "_points.npy")
        np.save(pointcloudfile, xyz)
        keypoint_description_file = mesh_path.replace(".obj", "_info.json")
        keypoint_info = {
            "keypoints": picked_points.tolist(),
            "extents": obj.extents.tolist(),
        }

        with open(keypoint_description_file, "w") as f:
            json.dump(keypoint_info, f, indent=4, sort_keys=True)

        # also save sampled points
