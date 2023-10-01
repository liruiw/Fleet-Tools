import numpy as np
import os
import warnings

from core.utils.robot_internal_controller import RobotInternalController
from pydrake.all import *
import warnings

warnings.simplefilter("ignore")

import IPython
from core.utils import *
from env.meshcat_cpp_utils import *
import time
import json
from colored import fg
import gym

screw_joint_ = None
bolt_joint = None
extra_force_system = None
TRAIN_TEST_SPLIT = 0.5


class ExternalForceSystem(LeafSystem):
    """
    This class provides external force for rigid bodies in the MBP to change the force behavior.
    """

    def __init__(self, plant, target_idx):
        LeafSystem.__init__(self)

        self.nv = 0  # plant.num_velocities()
        self.target_body_index = target_idx
        forces_cls = Value[List[ExternallyAppliedSpatialForce]]

        self.DeclareAbstractOutputPort("spatial_forces_vector", lambda: forces_cls(), self.DoCalcAbstractOutput)

        self.DeclareVectorOutputPort("generalized_forces", self.nv, self.DoCalcVectorOutput)

        self.wrench = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def DoCalcAbstractOutput(self, context, y_data):
        test_force = ExternallyAppliedSpatialForce()
        test_force.body_index = self.target_body_index
        test_force.p_BoBq_B = np.array([0, 0, 1e-3])  # the origin of the torque  np.zeros(3) #
        test_force.F_Bq_W = SpatialForce(tau=self.wrench[3:], f=self.wrench[:3])
        y_data.set_value([test_force])

    def DoCalcVectorOutput(self, context, y_data):
        y_data.SetFromVector([0] * self.nv)


def AddFloatingRpyJoint(plant, frame, instance):
    inertia = UnitInertia.SolidSphere(1.0)
    x_body = plant.AddRigidBody("x", instance, SpatialInertia(mass=0, p_PScm_E=[0.0, 0.0, 0.0], G_SP_E=inertia))
    plant.AddJoint(PrismaticJoint("x", plant.world_frame(), x_body.body_frame(), [1, 0, 0]))
    y_body = plant.AddRigidBody("y", instance, SpatialInertia(mass=0, p_PScm_E=[0.0, 0.0, 0.0], G_SP_E=inertia))
    plant.AddJoint(PrismaticJoint("y", x_body.body_frame(), y_body.body_frame(), [0, 1, 0]))
    z_body = plant.AddRigidBody("z", instance, SpatialInertia(mass=0, p_PScm_E=[0.0, 0.0, 0.0], G_SP_E=inertia))
    plant.AddJoint(PrismaticJoint("z", y_body.body_frame(), z_body.body_frame(), [0, 0, 1]))
    plant.AddJoint(BallRpyJoint("ball", z_body.body_frame(), frame))


def FindResource(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def AddFranka(plant, collision_model="no_collision"):
    """Add the default franka panda robot to the scene. its origin is also the origin of the scene"""

    franka_combined_path = FindResource("models/panda_arm_hand.urdf")
    parser = Parser(plant)
    franka = parser.AddModelFromFile(franka_combined_path)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"))

    # Set default positions:
    q0 = [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]
    index = 0
    for joint_index in plant.GetJointIndices(franka):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1
    return franka


def AddTable(plant, blender=False):
    """Add the default table to the scene"""
    parser = Parser(plant)
    if blender:
        sdf_path = FindResource("models/table/table.sdf")
    else:
        sdf_path = FindResource("models/table.sdf")
    table_pose = RigidTransform(RollPitchYaw(0, 0, 0), [0.96, 0, 0.15])
    table = parser.AddModelFromFile(sdf_path)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("top_center", table), table_pose)
    return table


def AddCameraBox(plant, X_WC, name="camera0", parent_frame=None):
    if not parent_frame:
        parent_frame = plant.world_frame()

    camera = Parser(plant).AddModelFromFile(FindResource("models/camera_box.sdf"), name)
    plant.WeldFrames(parent_frame, plant.GetFrameByName("base", camera), X_WC)


def AddShape(plant, shape, name, mass=1, mu=1, color=[0.5, 0.5, 0.9, 1.0], collision=True):
    instance = plant.AddModelInstance(name)
    if isinstance(shape, Box):
        inertia = UnitInertia.SolidBox(shape.width(), shape.depth(), shape.height())
    elif isinstance(shape, Cylinder):
        inertia = UnitInertia.SolidCylinder(shape.radius(), shape.length())
    elif isinstance(shape, Sphere):
        inertia = UnitInertia.SolidSphere(shape.radius())
    else:
        raise RuntimeError(f"need to write the unit inertia for shapes of type {shape}")
    body = plant.AddRigidBody(
        name,
        instance,
        SpatialInertia(mass=mass, p_PScm_E=np.array([0.0, 0.0, 0.0]), G_SP_E=inertia),
    )

    if plant.geometry_source_is_registered():
        """add small spheres"""
        if collision:
            if isinstance(shape, Box):
                plant.RegisterCollisionGeometry(
                    body,
                    RigidTransform(),
                    Box(
                        shape.width() - 0.001,
                        shape.depth() - 0.001,
                        shape.height() - 0.001,
                    ),
                    name,
                    CoulombFriction(mu, mu),
                )
                i = 0
                for x in [-shape.width() / 2.0, shape.width() / 2.0]:
                    for y in [-shape.depth() / 2.0, shape.depth() / 2.0]:
                        for z in [-shape.height() / 2.0, shape.height() / 2.0]:
                            plant.RegisterCollisionGeometry(
                                body,
                                RigidTransform([x, y, z]),
                                Sphere(radius=1e-7),
                                f"contact_sphere{i}",
                                CoulombFriction(mu, mu),
                            )
                            i += 1
            else:
                plant.RegisterCollisionGeometry(body, RigidTransform(), shape, name, CoulombFriction(mu, mu))
        plant.RegisterVisualGeometry(body, RigidTransform(), shape, name, color)
    return instance


def LoadMeshFrom(parent_path, name, fixed_idx, info):
    object_paths = sorted(
        [os.path.join(parent_path, name, o, o + ".sdf") for o in os.listdir(os.path.join(parent_path, name))]
    )
    if fixed_idx != -1:
        object_paths = [n for n in object_paths if f"model{fixed_idx}" in n]
        if len(object_paths) == 0:
            print("no matching idx tool", fixed_idx, object_paths)
            object_paths = sorted(
                [os.path.join(parent_path, name, o, o + ".sdf") for o in os.listdir(os.path.join(parent_path, name))]
            )

    split_idx = int(TRAIN_TEST_SPLIT * len(object_paths))
    idx = np.random.randint(0, max(1, split_idx))

    path = object_paths[idx]
    idx = idx

    fixedpath = [p for p in object_paths if str(idx) in p]
    if len(fixedpath) > 0:
        path = fixedpath[0]

    if len(info) > 0:

        matched_paths = [p for p in object_paths if p.split("/")[-1][:-4] in info]
        if len(matched_paths) > 0:
            path = matched_paths[0]
        else:
            print(object_paths)
            # raise ValueError("ERROR: matching instance not found: {}".format(info))
            print("ERROR: matching instance not found: {}".format(info))
    return path


def AddTools(
    plant,
    tool_name,
    instance_info={},
    fixed_idx=-1,
    train=False,
    actuated=False,
    load_all_objs=False,
    blender_render=False,
):
    """
    Adding the tool to the robot
    """
    parser = Parser(plant)
    # obj_path = "assets/tools"  # objects_sdf
    obj_path = "assets/tools"  # objects_sdf
    info = "" if len(instance_info) == 0 else instance_info["tool_name"]
    path = LoadMeshFrom(obj_path, tool_name, fixed_idx, info)
    model_name = path.split("/")[-1][:-4]

    tool_info_path = path.replace(".sdf", "_info.json")
    tool_info = json.load(open(tool_info_path))
    object_name = tool_name + "_" + model_name + "_body_link"
    if blender_render:
        path = path.replace(".sdf", "_blender.sdf")

    tool_obj = parser.AddModelFromFile(path, object_name)
    return [tool_obj, object_name, tool_info, path]


def AddToolObjects(
    builder, plant, scene_graph, tool_name, instance_info={}, fixed_idx=-1, train=False, load_all_objs=False
):
    """
    Adding the object that the tool needs to interact with
    """
    if tool_name == "wrench":
        return AddWrenchObject(
            builder,
            plant,
            scene_graph,
            tool_name,
            instance_info,
            fixed_idx=fixed_idx,
            train=train,
        )

    if tool_name == "knife":
        return AddKnifeObject(
            builder,
            plant,
            scene_graph,
            tool_name,
            instance_info,
            fixed_idx=fixed_idx,
            train=train,
        )

    if tool_name == "hammer":
        return AddHammerObject(
            builder,
            plant,
            scene_graph,
            tool_name,
            instance_info,
            fixed_idx=fixed_idx,
            train=train,
        )

    if tool_name == "spatula":
        return AddSpatulaObject(
            builder,
            plant,
            scene_graph,
            tool_name,
            instance_info,
            fixed_idx=fixed_idx,
            train=train,
        )

    if tool_name == "scoop" or tool_name == "whisk":
        return AddBowlObject(
            builder,
            plant,
            scene_graph,
            tool_name,
            instance_info,
            fixed_idx=fixed_idx,
            train=train,
        )

    parser = Parser(plant)
    # obj_path = "assets/tool_obj"  # objects_sdf
    obj_path = "assets/tool_obj"  # objects_sdf
    info = "" if len(instance_info) == 0 else instance_info["object_name"]
    path = LoadMeshFrom(obj_path, tool_name, fixed_idx, info)
    print(f"Using object at {path}")

    model_name = path.split("/")[-1][:-4]
    tool_obj_keypoint_path = path.replace(".sdf", "_info.json")

    tool_obj_info = json.load(open(tool_obj_keypoint_path))  # ['keypoints']
    object_name = model_name + "_body_link"
    tool_obj = parser.AddModelFromFile(path, object_name)
    return [tool_obj, object_name, tool_obj_info, path]


def AddBowlObject(
    builder, plant, scene_graph, tool_name, instance_info={}, fixed_idx=-1, train=False, load_all_objs=False
):
    """add a welded bowl to the table"""
    parser = Parser(plant)
    # obj_path = "assets/tool_obj"  # objects_sdf
    obj_path = "assets/tool_obj"  # objects_sdf
    info = "" if len(instance_info) == 0 else instance_info["object_name"]
    path = LoadMeshFrom(obj_path, tool_name, fixed_idx, info)
    print(f"Using object at {path}")

    initial_rot = RollPitchYaw(0, 0, np.random.uniform(-np.pi, np.pi))  # -np.pi/2
    tf = RigidTransform(initial_rot, [np.random.uniform(0.2, 0.4), np.random.uniform(-0.2, 0.2), 0.02])

    if len(instance_info) > 0:
        object_pose = instance_info["object_pose"][0]
        tf = RigidTransform(object_pose)

    model_name = path.split("/")[-1][:-4]
    tool_obj_keypoint_path = path.replace(".sdf", "_info.json")
    tool_obj_info = json.load(open(tool_obj_keypoint_path))  # ['keypoints']
    object_name = model_name + "_body_link"
    tool_obj = parser.AddModelFromFile(path, object_name)
    tool_body_frame = plant.GetBodyByName(object_name).body_frame()

    plant.WeldFrames(plant.world_frame(), tool_body_frame, tf)
    # self.set_object_pose(plant_context, tf)
    return [tool_obj, object_name, tool_obj_info, path]


def AddSpatulaObject(
    builder, plant, scene_graph, tool_name, instance_info={}, fixed_idx=-1, train=False, load_all_objs=False
):
    """Add four small spheres under spatula for scooping"""
    parser = Parser(plant)
    obj_path = "assets/tool_obj"  # objects_sdf
    info = "" if len(instance_info) == 0 else instance_info["object_name"]
    path = LoadMeshFrom(obj_path, tool_name, fixed_idx, info)
    print(f"Using object at {path}")

    model_name = path.split("/")[-1][:-4]
    tool_obj_keypoint_path = path.replace(".sdf", "_info.json")
    tool_obj_info = json.load(open(tool_obj_keypoint_path))

    object_name = model_name + "_body_link"
    tool_obj = parser.AddModelFromFile(path, object_name)

    body_frame_id = plant.GetBodyFromFrameId(plant.GetBodyFrameIdOrThrow(plant.GetBodyIndices(tool_obj)[0]))

    # add small spheres for easy scooping
    i = 0
    for x in [-0.02, 0.02]:
        for y in [-0.02, 0.02]:
            plant.RegisterCollisionGeometry(
                body_frame_id,
                RigidTransform([x, y, -0.02]),
                Sphere(radius=1e-5),
                f"contact_sphere{i}",
                CoulombFriction(1, 1),
            )
            i += 1

    return [tool_obj, object_name, tool_obj_info, path]


def AddHammerObject(
    builder, plant, scene_graph, tool_name, instance_info={}, fixed_idx=-1, train=False, load_all_objs=False
):
    """
    Adding a bolt with prismatic joint for hammering
    """
    parser = Parser(plant)
    obj_path = "assets/tool_obj"  # objects_sdf
    info = "" if len(instance_info) == 0 else instance_info["object_name"]
    path = LoadMeshFrom(obj_path, tool_name, fixed_idx, info)
    print(f"Using object at {path}")

    model_name = path.split("/")[-1][:-4]
    tool_obj_keypoint_path = path.replace(".sdf", "_info.json")
    tool_obj_info = json.load(open(tool_obj_keypoint_path))  # ['keypoints']

    object_name = model_name + "_body_link"
    tool_obj = parser.AddModelFromFile(path, object_name)

    # build the bolt-nut combo
    # child rotates and translates on the axis of Z of the parent
    bolt = tool_obj

    # get table
    bolt_body_frame_id = plant.GetBodyFrameIdOrThrow(plant.GetBodyIndices(bolt)[0])

    try:
        table_body_frame = plant.GetBodyByName("link").body_frame()
    except:
        table_body_frame = plant.GetBodyByName("top_center").body_frame()
    bolt_body_frame = plant.GetBodyFromFrameId(bolt_body_frame_id).body_frame()
    global bolt_joint, extra_force_system

    # transparent
    box_instance = AddShape(
        plant,
        Box(0.1, 0.1, 0.03),
        "box",
        1,
        1,
        color=[0.1, 0.5, 0.1, 0.0],
        collision=True,
    )
    box_body_frame = plant.GetBodyByName("box").body_frame()
    bolt_pose = RigidTransform(RollPitchYaw(0, 0, 0), [0.56, 0.0, 0.2])  # 0.13

    plant.WeldFrames(plant.world_frame(), box_body_frame, bolt_pose)
    bolt_joint = PrismaticJoint(
        "bolt_joint",
        box_body_frame,
        bolt_body_frame,
        [0, 0, -1],
        damping=1,
        pos_upper_limit=-0.01,
        pos_lower_limit=-0.025,
    )
    bolt_joint.set_default_translation(-0.025)
    bolt_joint = plant.AddJoint(bolt_joint)
    bolt_body_id = plant.GetBodyIndices(bolt)[0]

    # maybe add a board first
    extra_force_system = builder.AddSystem(ExternalForceSystem(plant, bolt_body_id))
    return [tool_obj, object_name, tool_obj_info, path]


def AddKnifeObject(
    builder, plant, scene_graph, tool_name, instance_info={}, fixed_idx=-1, train=False, load_all_objs=False
):
    """Add two separate parts for knife splitting"""
    parser = Parser(plant)
    obj_path = "assets/tool_obj"  # objects_sdf
    info = "" if len(instance_info) == 0 else instance_info["object_name"]
    path = LoadMeshFrom(obj_path, tool_name, fixed_idx, info)
    print(f"Using object at {path}")

    model_name = path.split("/")[-1][:-4]
    tool_obj_keypoint_path = path.replace(".sdf", "_info.json")
    tool_obj_info = json.load(open(tool_obj_keypoint_path))

    object_name = model_name + "_body_link"
    object_name2 = model_name + "_body_link_2"

    tool_obj = parser.AddModelFromFile(path, object_name)  # add two parts

    copy_path = path.replace(".sdf", "_copy.sdf")
    if not os.path.exists(copy_path):
        copy_file_content = open(path).read()
        copy_file_content = copy_file_content.replace("body_link", "body_link_2")
        with open(copy_path, "w") as f:
            f.write(copy_file_content)

    # manually create the copy
    tool_obj2 = parser.AddModelFromFile(copy_path, object_name2)
    # add small spheres for easy splitting
    body_frame_id = plant.GetBodyFromFrameId(plant.GetBodyFrameIdOrThrow(plant.GetBodyIndices(tool_obj)[0]))

    i = 0
    for x in [-0.02, 0.02]:
        for y in [-0.02, 0.02]:
            plant.RegisterCollisionGeometry(
                body_frame_id,
                RigidTransform([x, y, -0.04]),
                Sphere(radius=1e-4),
                f"contact_sphere{i}",
                CoulombFriction(0.5, 0.5),
            )
            i += 1

    body_frame_id = plant.GetBodyFromFrameId(plant.GetBodyFrameIdOrThrow(plant.GetBodyIndices(tool_obj2)[0]))

    # add small spheres for easy splitting
    for x in [-0.02, 0.02]:
        for y in [-0.02, 0.02]:
            plant.RegisterCollisionGeometry(
                body_frame_id,
                RigidTransform([x, y, -0.04]),
                Sphere(radius=1e-4),
                f"contact_sphere{i}",
                CoulombFriction(0.5, 0.5),
            )
            i += 1

    return [tool_obj, object_name, tool_obj_info, path]


def AddWrenchObject(
    builder, plant, scene_graph, tool_name, instance_info={}, fixed_idx=-1, train=False, load_all_objs=False
):
    """A way to handle the screw joint special case"""
    parser = Parser(plant)
    obj_path = "assets/tool_obj"  # objects_sdf
    # obj_path = "assets/objects_scaled"  # objects_sdf
    info = "" if len(instance_info) == 0 else instance_info["object_name"]
    path = LoadMeshFrom(obj_path, tool_name, fixed_idx, info)
    print(f"Using object at {path}")

    model_name = path.split("/")[-1][:-4]
    tool_obj_keypoint_path = path.replace(".sdf", "_nut_info.json")
    tool_obj_info = json.load(open(tool_obj_keypoint_path))  # ['keypoints']

    object_name = model_name + "_body_link"
    tool_obj = parser.AddModelFromFile(path, object_name)

    # build the bolt-nut combo
    # child rotates and translates on the axis of Z of the parent
    bolt = tool_obj
    nut_name = object_name.replace("_body", "_nut_body")
    nut_path = path.replace(".sdf", "_nut.sdf")

    nut = parser.AddModelFromFile(nut_path, nut_name)
    nut_body_id = plant.GetBodyIndices(nut)[0]
    nut_body_frame_id = plant.GetBodyFrameIdOrThrow(nut_body_id)
    bolt_body_frame_id = plant.GetBodyFrameIdOrThrow(plant.GetBodyIndices(bolt)[0])

    # body_frame
    nut_body_frame = plant.GetBodyFromFrameId(nut_body_frame_id).body_frame()
    bolt_body_frame = plant.GetBodyFromFrameId(bolt_body_frame_id).body_frame()
    # the origin of the nut must be at the screw
    screw_joint = ScrewJoint(
        "nut_bolt_joint",
        bolt_body_frame,
        nut_body_frame,
        screw_pitch=5e-3,
        damping=5e-2,
    )

    # weld the bolt to the table, at a fixed pose for now.
    plant.AddJoint(screw_joint)
    x_random_range = np.random.uniform(0.35, 0.5)
    y_random_range = np.random.uniform(-0.15, 0.15)
    bolt_pose = RigidTransform(RollPitchYaw(0, 0, 0), [x_random_range, y_random_range, 0.18])

    # bolt_pose = RigidTransform(RollPitchYaw(0, 0, 0), [0.4, 0, 0.18])
    plant.WeldFrames(plant.world_frame(), bolt_body_frame, bolt_pose)
    tool_obj = nut
    global screw_joint_, extra_force_system
    screw_joint_ = screw_joint
    extra_force_system = builder.AddSystem(ExternalForceSystem(plant, nut_body_id))
    # extra_force_system = builder.AddSystem(ExternalForceSystem(plant, nut_body_id))

    return [tool_obj, object_name, tool_obj_info, path]
