import numpy as np
import os
import sys
import torch

from core.utils.robot_internal_controller import RobotInternalController
from core.utils.spatialstiffness_controller import PoseController
from pydrake.all import *
import warnings

warnings.simplefilter("ignore")
import IPython
from core.utils.utils import *
from env.meshcat_cpp_utils import *
import time
from colored import fg
from env.env_object_related import *

try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
except:
    pass
from scipy import interpolate

g = RandomGenerator()
#  global variables
meshcat = None
visualizer = None
blender_color_cam = None
robot_plant = None  # for computing FK and IK'

# drake camera coordinate system has z facing forward and y facing downward
OVERHEAD_CAM_EXTR = (
    np.array(
        [
            [-0.21092498, -0.05561141, -0.97591907, 1.51906],
            [-0.13466501, -0.9872077, 0.08535977, 0.161601],
            [-0.9681818, 0.14942665, 0.20073785, 0.634862],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    .dot(rotY(np.pi / 6))
    .dot(rotZ(-np.pi / 2))
)


HAND_CAM_EXTR = (
    np.array(
        [
            [0.00266189, -0.00885279, 0.99995727, 0.1260556],
            [0.02328945, 0.99969013, 0.00878842, -0.036485],
            [-0.99972522, 0.02326506, 0.00286725, 0.057091],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    .dot(rotY(-np.pi / 2))
    .dot(rotZ(-np.pi / 2))
    .dot(rotX(np.pi / 10))
)

CAM_INTR = np.array(
    [
        614.88671875,
        0.0,
        323.3525390625,
        0.0,
        615.4398193359375,
        239.89163208007812,
        0.0,
        0.0,
        1.0,
    ]
).reshape(3, 3)


def reset_global_variables():
    # hack to reset all global variables
    from core.utils.utils import (
        vr_has_init,
        vr_frame_origin,
        rec_action_pose,
        oculus_reader,
    )

    vr_has_init = False
    rec_action_pose = None
    coordinate_origin = None
    oculus_reader = None


def color_print(str, color="white"):
    print(fg(color) + str)


def AddRgbdSensors(
    builder,
    plant,
    scene_graph,
    also_add_point_clouds=True,
    model_instance_prefix="panda",
    depth_camera=None,
    renderer=None,
    width=640,
    height=480,
    relative_pose=None,
    attach_body_name="",
):
    """
    Adds a RgbdSensor to every body in the plant with a name starting with body_prefix.
    """
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display

        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    if not renderer:
        renderer = "my_renderer"

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(
            renderer, MakeRenderEngineVtk(RenderEngineVtkParams(default_diffuse=(0.5, 0.5, 0.5, 1)))
        )

    if not depth_camera:
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer,
                CameraInfo(width=width, height=height, intrinsic_matrix=CAM_INTR),
                ClippingRange(near=0.02, far=3.0),
                RigidTransform(),
            ),
            #  overlaped with the rgbd sensor pose
            DepthRange(0.2, 3.0),
        )

    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)

        if model_name.startswith(model_instance_prefix):
            if len(attach_body_name) == 0:
                body_index = plant.GetBodyIndices(model_instance_index)[0]
                model_name = "_".join([model_name, "base", "cam"])

            else:
                body_index = plant.GetBodyByName(attach_body_name).index()
                model_name = "_".join([attach_body_name, "cam"])

            if relative_pose is not None:
                X_PB = RigidTransform(relative_pose)
            else:
                X_PB = RigidTransform()
            rgbd = builder.AddSystem(
                RgbdSensor(
                    parent_id=plant.GetBodyFrameIdOrThrow(body_index),
                    X_PB=X_PB,
                    depth_camera=depth_camera,
                    show_window=False,
                )
            )

            rgbd.set_name(model_name)
            builder.Connect(scene_graph.get_query_output_port(), rgbd.query_object_input_port())

            # Export the camera outputs
            builder.ExportOutput(rgbd.color_image_output_port(), f"{model_name}_rgb_image")
            builder.ExportOutput(rgbd.depth_image_32F_output_port(), f"{model_name}_depth_image")
            builder.ExportOutput(rgbd.label_image_output_port(), f"{model_name}_label_image")

            if also_add_point_clouds:
                # Add a system to convert the camera output into a point cloud
                to_point_cloud = builder.AddSystem(
                    DepthImageToPointCloud(
                        camera_info=rgbd.depth_camera_info(),
                        fields=BaseField.kXYZs | BaseField.kRGBs,
                    )
                )
                builder.Connect(
                    rgbd.depth_image_32F_output_port(),
                    to_point_cloud.depth_image_input_port(),
                )
                builder.Connect(
                    rgbd.color_image_output_port(),
                    to_point_cloud.color_image_input_port(),
                )

                class ExtractBodyPose(LeafSystem):
                    def __init__(self, body_index):
                        LeafSystem.__init__(self)
                        self.body_index = body_index
                        self.DeclareAbstractInputPort("poses", plant.get_body_poses_output_port().Allocate())
                        self.DeclareAbstractOutputPort(
                            "pose",
                            lambda: AbstractValue.Make(RigidTransform()),
                            self.CalcOutput,
                        )

                    def CalcOutput(self, context, output):
                        poses = self.EvalAbstractInput(context, 0).get_value()
                        pose = poses[int(self.body_index)]
                        output.get_mutable_value().set(pose.rotation(), pose.translation())
                        #
                        new_output = RigidTransform(output.get_mutable_value().GetAsMatrix4().dot(relative_pose))
                        output.get_mutable_value().set(new_output.rotation(), new_output.translation())

                camera_pose = builder.AddSystem(ExtractBodyPose(body_index))
                builder.Connect(plant.get_body_poses_output_port(), camera_pose.get_input_port())
                builder.Connect(
                    camera_pose.get_output_port(),
                    to_point_cloud.GetInputPort("camera_pose"),
                )
                builder.ExportOutput(camera_pose.get_output_port(), f"{model_name}_camera_pose")
                # Export the point cloud output.
                builder.ExportOutput(
                    to_point_cloud.point_cloud_output_port(),
                    f"{model_name}_point_cloud",
                )


def UpdateObjectPhysicalProperties(
    scene_graph,
    inspector,
    source_id,
    oid,
    obj_name,
    range_lo=0.1,
    range_hi=0.9,
    randomize_physics=False,
    frictions_hyper_param=None,
    hydroelastic_contact=False,
    change_mass=False,
    compliant_model=False,
):
    """change physics parameters such as friction, mass, and hydroelastic contacts"""
    if not inspector.BelongsToSource(oid, source_id):
        return
    obj_gids = []
    for geometry_id in inspector.GetAllGeometryIds():
        name = inspector.GetName(geometry_id)
        if obj_name in name and "collision" in name:
            obj_gids.append(geometry_id)

    for gid in obj_gids:
        props = inspector.GetProximityProperties(gid)

        if randomize_physics:
            static_friction = np.random.uniform(range_lo, range_hi)  # simple for now
            friction = CoulombFriction(static_friction, static_friction)
            if props is not None:
                props.UpdateProperty("material", "coulomb_friction", friction)
            try:
                scene_graph.AssignRole(source_id, gid, props, RoleAssign.kReplace)
            except:
                pass

        if hydroelastic_contact:
            try:
                # low friction and less bumpy
                dissipation = 5.0
                mu_static = 0.8  # 0.6
                mu_dynamic = 0.6  # 0.2
                if frictions_hyper_param is not None:
                    dissipation = frictions_hyper_param[0]
                    mu_static = frictions_hyper_param[1]  # 0.6
                    mu_dynamic = frictions_hyper_param[2]  # 0.2

                friction = CoulombFriction(0.7 * mu_static, 0.7 * mu_dynamic)
                if props is not None:
                    if not props.HasProperty("hydroelastic", "resolution_hint"):
                        if compliant_model:
                            AddCompliantHydroelasticProperties(
                                resolution_hint=0.005,
                                hydroelastic_modulus=5e6,
                                properties=props,
                            )  # 0.005
                        else:
                            AddRigidHydroelasticProperties(resolution_hint=0.005, properties=props)

                    if not props.HasProperty("material", "coulomb_friction"):
                        AddContactMaterial(dissipation=dissipation, friction=friction, properties=props)
                scene_graph.AssignRole(source_id, gid, props, RoleAssign.kReplace)
            except:
                print("adding complaint hydroelastic_contact fails")


def UpdateObjectVisualProperties(
    scene_graph,
    inspector,
    source_id,
    oid,
    obj_name,
    range_lo=0.1,
    range_hi=0.9,
    color=None,
    alpha=-1,
    randomize_textures=False,
):
    """change color, transparency, and texture"""

    if not inspector.BelongsToSource(oid, source_id):
        return

    # visual
    obj_gids = []
    for geometry_id in inspector.GetAllGeometryIds():
        name = inspector.GetName(geometry_id)
        if obj_name in name and "visual" in name:
            obj_gids.append(geometry_id)

    for gid in obj_gids:
        if color is not None:
            props = inspector.GetIllustrationProperties(gid)
            props = SetColor(props, color)

        if alpha > 0:
            props = inspector.GetIllustrationProperties(gid)
            props = SetTransparency(props, alpha)

        if randomize_textures:
            props = inspector.GetPerceptionProperties(gid)
            props = SetTextures(props)


def SetTextures(props):
    """update texture in drake"""
    texture_dir = "assets/textures/new_textures"
    files = os.listdir(texture_dir)
    texture_path = os.path.join(texture_dir, np.random.choice(files, 1)[0])

    if props is not None and props.HasProperty("phong", "diffuse_map"):
        c = props.GetProperty("phong", "diffuse_map")
        props.UpdateProperty("phong", "diffuse_map", texture_path)
    else:
        props.AddProperty("phong", "diffuse_map", texture_path)
    return props


def SetTransparency(props, alpha):
    """set transparency on meshcat"""
    if props is not None and props.HasProperty("phong", "diffuse"):
        c = props.GetProperty("phong", "diffuse")
        new_color = Rgba(c.r(), c.g(), c.b(), alpha)
        props.UpdateProperty("phong", "diffuse", new_color)
    else:
        new_color = Rgba(1.0, 1.0, 1.0, alpha)
        props.AddProperty("phong", "diffuse", new_color)
    return props


def SetColor(props, color):
    if props is not None and props.HasProperty("phong", "diffuse"):
        new_color = Rgba(color[0], color[1], color[2], color[3])
        props.UpdateProperty("phong", "diffuse", new_color)
    return props


def AddTriad(
    source_id,
    frame_id,
    scene_graph,
    length=0.25,
    radius=0.01,
    opacity=1.0,
    X_FT=RigidTransform(),
    name="frame",
):
    """
    Adds illustration geometry representing the coordinate frame, with the
    x-axis drawn in red, the y-axis in green and the z-axis in blue. The axes
    point in +x, +y and +z directions, respectively.
    """
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    geom = GeometryInstance(X_FT.multiply(X_TG), Cylinder(radius, length), name + " x-axis")
    geom.set_illustration_properties(MakePhongIllustrationProperties([1, 0, 0, opacity]))
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    geom = GeometryInstance(X_FT.multiply(X_TG), Cylinder(radius, length), name + " y-axis")
    geom.set_illustration_properties(MakePhongIllustrationProperties([0, 1, 0, opacity]))
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    geom = GeometryInstance(X_FT.multiply(X_TG), Cylinder(radius, length), name + " z-axis")
    geom.set_illustration_properties(MakePhongIllustrationProperties([0, 0, 1, opacity]))
    scene_graph.RegisterGeometry(source_id, frame_id, geom)


def visualize_rendered_image(state):
    """visualize the observation with pyplot"""
    print("visualize data shape:", state.shape)
    plt.switch_backend("TKAgg")

    fig = plt.figure(figsize=(25.6, 9.6))
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title("color")
    plt.imshow((state[..., :3] * 255).astype(np.uint8))

    if state.shape[0] == 4:
        ax = fig.add_subplot(1, 3, 2)
        ax.set_title("depth")
        plt.imshow(state[..., -1], cmap=plt.get_cmap("magma"))

    if state.shape[0] >= 5:
        ax = fig.add_subplot(1, 3, 2)
        ax.set_title("depth")
        plt.imshow(state[..., -2], cmap=plt.get_cmap("magma"))

        ax = fig.add_subplot(1, 3, 3)
        ax.set_title("mask")
        mask = (state[..., -1]).astype(np.uint8)
        mask[mask == 255] = 0
        mask[mask == 254] = 0

        plt.imshow(mask)
    plt.show()


def AddMeshcatTriad(meshcat, path, length=0.25, radius=0.01, opacity=1.0, X_PT=RigidTransform()):
    """add meshcat visualization of a coord"""
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(path + "/x-axis", Cylinder(radius, length), Rgba(1, 0, 0, opacity))

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    meshcat.SetTransform(path + "/y-axis", X_TG)
    meshcat.SetObject(path + "/y-axis", Cylinder(radius, length), Rgba(0, 1, 0, opacity))

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    meshcat.SetTransform(path + "/z-axis", X_TG)
    meshcat.SetObject(path + "/z-axis", Cylinder(radius, length), Rgba(0, 0, 1, opacity))


def vis_tool_pose(meshcat, X_TG, path, mesh_path=None, color=[0.1, 0.1, 0.1, 0.3]):
    """render the tool at the pose tf in meshcat"""
    if type(X_TG) is not RigidTransform:
        X_TG = RigidTransform(X_TG)
    if meshcat is None:
        return

    if mesh_path is not None:
        mesh = ReadObjToTriangleSurfaceMesh(mesh_path.replace(".sdf", "_corr.obj"))
        meshcat.Delete(path)  # delete the old one
        meshcat.SetObject(path, mesh, Rgba(*color))
    meshcat.SetProperty(path, "color", color)
    meshcat.SetTransform(path, X_TG)


def vis_hand_pose(meshcat, X_TG, path, load=False, color=[0.1, 0.1, 0.1, 0.3]):
    """render the tool at the pose tf in meshcat"""
    if type(X_TG) is not RigidTransform:
        X_TG = RigidTransform(X_TG)
    if meshcat is None:
        return

    if load:
        mesh = ReadObjToTriangleSurfaceMesh("env/models/hand_finger.obj")
        meshcat.Delete(path)  # delete the old one
        meshcat.SetObject(path, mesh, Rgba(*color))
    meshcat.SetProperty(path, "color", color)
    meshcat.SetTransform(path, X_TG)


def AddWrenchVis(meshcat, path, point, wrench, length=0.03, radius=0.001, opacity=1.0):
    """compute the vector visualization of the wrench"""
    torque_line_vertices = np.array([point, point + wrench[:3]] * 100).T
    force_line_vertices = np.array([point, point + wrench[:3]] * 100).T

    meshcat.SetLine(path + "_force", force_line_vertices, radius, Rgba(1, 0, 0, opacity))
    meshcat.SetLine(path + "_wrench", torque_line_vertices, radius, Rgba(0, 1, 0, opacity))


def add_table_collision_free_constraint(ik, plant, frame, bb_size=[0.12, 0.08, 0.08], table_height=0.1):
    # apprxoimate a link with a bounding box and add as position constraints
    min_height = -0.01 + table_height
    max_num = 100
    y_bound = 1

    ik.AddPositionConstraint(
        frame,
        [0, 0, 0],
        plant.world_frame(),
        [0, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [0, 0, -bb_size[2]],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [0, 0, bb_size[2]],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [0, -bb_size[1], 0],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [0, bb_size[1], 0],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [bb_size[0], 0, 0],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )
    ik.AddPositionConstraint(
        frame,
        [-bb_size[0], 0, 0],
        plant.world_frame(),
        [-0.3, -y_bound, min_height],
        [max_num, y_bound, max_num],
    )


def compute_manipulability(plant, plant_context):
    """compute the manipulability matrix"""
    J = plant.CalcJacobianSpatialVelocity(
        plant_context,
        JacobianWrtVariable.kQDot,
        plant.GetBodyByName("panda_hand").body_frame(),
        [0, 0, 0],
        plant.world_frame(),
        plant.world_frame(),
    )[:, :7]
    return np.linalg.det(J @ J.T), np.linalg.cond(J @ J.T)


def refine_ik_traj(plant, pose_traj, q0, rot_tol=3e-2):
    """run a small trajopt on the trajectory with the solved IK from end-effector traj"""
    # make sure the beginning and the end do not get updated
    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(9, len(pose_traj))
    gripper_frame = plant.GetFrameByName("panda_hand")
    plant_contexts = [plant.CreateDefaultContext() for i in range(len(pose_traj))]

    for idx, pose in enumerate(pose_traj):
        pose = RigidTransform(pose)
        prog.AddConstraint(
            PositionConstraint(
                plant,
                plant.world_frame(),
                pose.translation(),
                pose.translation(),
                gripper_frame,
                [0, 0, 0],
                plant_contexts[idx],
            ),
            q[:, idx],
        )
        prog.AddConstraint(
            PositionConstraint(
                plant,
                plant.world_frame(),
                [0.05, -1, 0],
                [1, 1, 1],
                gripper_frame,
                [0, 0, 0],
                plant_contexts[idx],
            ),
            q[:, idx],
        )
        prog.AddConstraint(
            OrientationConstraint(
                plant,
                gripper_frame,
                RotationMatrix(),
                plant.world_frame(),
                pose.rotation(),
                rot_tol,
                plant_contexts[idx],
            ),
            q[:, idx],
        )

    # add some other constraints
    prog.AddConstraint(np.sum((q[:, 0] - q0[:, 0]) ** 2) == 0)
    prog.AddConstraint(np.sum((q[:, -1] - q0[:, -1]) ** 2) == 0)

    # Add smoothness cost

    weight = np.ones((9, 1))
    weight[0] = 10.0
    weight[-1] = 10.0
    prog.AddQuadraticCost(np.sum(weight * (q[:, 1:] - q[:, :-1]) ** 2))

    # add linear constraint
    try:
        solver = SnoptSolver()
    except:
        solver = IpoptSolver()
    prog.SetInitialGuess(q, q0)
    result = solver.Solve(prog)  #
    if result.is_success():
        return result
    else:
        return None


def refine_feasible_ik_traj(plant, pose_traj, q0, rot_tol=5e-2):
    """run a small trajopt on the trajectory with the solved IK from end-effector traj"""
    # make sure the beginning and the end do not get updated
    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(9, len(pose_traj))
    gripper_frame = plant.GetFrameByName("panda_hand")
    plant_contexts = [plant.CreateDefaultContext() for i in range(len(pose_traj))]

    for idx, pose in enumerate(pose_traj):
        pose = RigidTransform(pose)
        prog.AddConstraint(
            PositionConstraint(
                plant,
                plant.world_frame(),
                pose.translation(),
                pose.translation(),
                gripper_frame,
                [0, 0, 0],
                plant_contexts[idx],
            ),
            q[:, idx],
        )
        prog.AddConstraint(
            OrientationConstraint(
                plant,
                gripper_frame,
                RotationMatrix(),
                plant.world_frame(),
                pose.rotation(),
                rot_tol,
                plant_contexts[idx],
            ),
            q[:, idx],
        )

    prog.AddQuadraticCost(np.sum((q - q0) ** 2))  # np.sum((q[:, 1:] - q[:, :-1]) ** 2)
    # add linear constraint
    try:
        solver = SnoptSolver()
        prog.SetInitialGuess(q, q0)
        result = solver.Solve(prog)
    except:
        solver = IpoptSolver()
        prog.SetInitialGuess(q, q0)
        result = solver.Solve(prog)
    #
    if result.is_success():
        return result
    else:
        return None


def interpolate_waypoints(waypoints, n, m, mode="cubic"):  # linear
    """
    Interpolate the waypoints using interpolation.
    """
    data = np.zeros([n, m])
    x = np.linspace(0, 1, waypoints.shape[0])
    for i in range(waypoints.shape[1]):
        y = waypoints[:, i]

        t = np.linspace(0, 1, n + 2)
        if mode == "linear":  # clamped cubic spline
            f = interpolate.interp1d(x, y, "linear")
        if mode == "cubic":  # clamped cubic spline
            f = interpolate.CubicSpline(x, y, bc_type="clamped")
        elif mode == "quintic":  # seems overkill to me
            pass
        data[:, i] = f(t[1:-1])  #

    return data


def solve_ik_traj_with_standoff(
    plant, endpoint_pose, endpoint_joints, q_traj, waypoint_times, keytimes, keyposes, rot_tol=0.03
):
    """run a small trajopt on the trajectory with the solved IK from end-effector traj"""
    # make sure the beginning and the end do not get updated
    waypoint_num = len(waypoint_times)
    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(9, waypoint_num)
    gripper_frame = robot_plant.GetFrameByName("panda_hand")
    plant_contexts = [robot_plant.CreateDefaultContext() for i in range(waypoint_num)]
    q0 = np.array([q_traj.value(t) for t in waypoint_times])

    for idx, time in enumerate(waypoint_times):
        if time in keytimes:  # standoff
            keypoint_idx = keytimes.index(time)
            pose = RigidTransform(keyposes[keypoint_idx])
            prog.AddConstraint(
                OrientationConstraint(
                    robot_plant,
                    gripper_frame,
                    RotationMatrix(),
                    robot_plant.world_frame(),
                    pose.rotation(),
                    rot_tol,
                    plant_contexts[idx],
                ),
                q[:, idx],
            )

            prog.AddConstraint(
                PositionConstraint(
                    robot_plant,
                    robot_plant.world_frame(),
                    pose.translation(),
                    pose.translation(),
                    gripper_frame,
                    [0, 0, 0],
                    plant_contexts[idx],
                ),
                q[:, idx],
            )

        # table constraint
        prog.AddConstraint(
            PositionConstraint(
                robot_plant,
                robot_plant.world_frame(),
                [0.05, -0.7, 0.06],
                [1, 0.7, 1],
                gripper_frame,
                [0, 0, 0],
                plant_contexts[idx],
            ),
            q[:, idx],
        )

    # add some other constraints
    prog.AddConstraint(np.sum((q[:, 0] - endpoint_joints[:, 0]) ** 2) == 0)

    # Add smoothness cost
    weight = np.ones((9, 1))
    weight[0] = 10.0
    weight[-1] = 10.0
    prog.AddQuadraticCost(np.sum(weight * (q[:, 1:] - q[:, :-1]) ** 2))
    prog.SetInitialGuess(q, q0.squeeze().T)

    # add linear constraint
    try:
        solver = SnoptSolver()
        result = solver.Solve(prog)
    except:
        solver = IpoptSolver()
        result = solver.Solve(prog)

    if result.is_success():
        return result
    else:
        return None


def smoothness(q):
    return np.sum((q[:, 1:] - q[:, :-1]) ** 2)


def solve_ik_kpam(
    cost_and_constraints,
    plant,
    gripper_frame,
    keypoints_in_hand,
    keypoint_obj_in_world,
    p0,
    q0,
    centering_joint,
    rot_tol=np.pi / 10,
    add_table_col=True,
    add_gripper_faceup=False,
    timeout=False,
    consider_collision=False,
    table_height=0.15,
):
    """
    the simple case for the kpam problem
    always assume the head tool keyponint match the head object keypoint
    the head and tail orthogonal to the 0,0,1
    and the tail and side orthogonal to 0,0,1
    minimize cost and both joint space and the pose space
    """

    constraint_dicts, cost_dicts = cost_and_constraints
    ik_context = robot_plant.CreateDefaultContext()
    ik = InverseKinematics(robot_plant, ik_context)

    # separate out axis
    for constraint in constraint_dicts:
        if "axis_from_keypoint_name" in constraint:
            from_idx = constraint["axis_from_keypoint_idx"]
            to_idx = constraint["axis_to_keypoint_idx"]
            target_axis = constraint["target_axis"]
            vec = keypoints_in_hand[to_idx] - keypoints_in_hand[from_idx]
            tol = constraint["tolerance"]
            tgt = np.arccos(constraint["target_inner_product"])
            lower_bound = max(tgt - tol, 0)
            upper_bound = min(tgt + tol, np.pi)

            ik.AddAngleBetweenVectorsConstraint(
                gripper_frame, vec, robot_plant.world_frame(), target_axis, lower_bound, upper_bound
            )
        else:
            idx = constraint["keypoint_idx"]
            target_position = constraint["target_position"]

            target_point = keypoints_in_hand[idx]
            tol = constraint["tolerance"]

            ik.AddPositionConstraint(
                gripper_frame, target_point, robot_plant.world_frame(), target_position - tol, target_position + tol
            )

    """solving IK to match tool head keypoint and the object keypoint"""
    # make sure the arm does not go backward
    ik.AddPositionConstraint(gripper_frame, [0, 0, 0], robot_plant.world_frame(), [0.05, -1, 0], [1, 1, 1])

    if add_gripper_faceup:
        ik.AddAngleBetweenVectorsConstraint(
            gripper_frame, [1, 0, 0], robot_plant.world_frame(), [0, 0, -1], np.pi / 12, np.pi
        )

    # not touching table constraints add elbow
    if add_table_col:
        add_table_collision_free_constraint(ik, robot_plant, gripper_frame, [0.03, 0.04, 0.08])
        add_table_collision_free_constraint(
            ik,
            robot_plant,
            robot_plant.GetFrameByName("panda_link6"),
            [0.03, 0.03, 0.03],
        )
        add_table_collision_free_constraint(
            ik,
            robot_plant,
            robot_plant.GetFrameByName("panda_link7"),
            [0.03, 0.03, 0.03],
        )

    if consider_collision:
        ik.AddMinimumDistanceConstraint(0.01)  # 0.03

    prog = ik.get_mutable_prog()
    q = ik.q()
    solver = SnoptSolver()
    # if timeout:
    #    solver.SetSolverOption(solver.id(), "Major Iterations Limit", 1000)

    # as well as pose space costs experiments
    joint_cost_mat = np.identity(len(q))
    joint_cost_mat[0, 0] = 10  # 000
    prog.AddQuadraticErrorCost(joint_cost_mat, centering_joint, q)
    ik.AddPositionCost(gripper_frame, [0, 0, 0], robot_plant.world_frame(), p0.translation(), np.eye(3))
    ik.AddOrientationCost(gripper_frame, RotationMatrix(), robot_plant.world_frame(), p0.rotation(), 1)

    prog.SetInitialGuess(q, q0)
    result = solver.Solve(ik.prog())  #
    if result.is_success():
        return result
    else:
        return None


def solve_ik(
    plant,
    gripper_frame,
    pose,
    q0,
    rot_tol=np.pi / 10,
    add_table_col=True,
    add_gripper_faceup=False,
    timeout=False,
    consider_collision=False,
    table_height=0.15,
    filter_manipulability=False,
):
    ik_context = robot_plant.CreateDefaultContext()
    ik = InverseKinematics(robot_plant, ik_context)

    """solving IK with environment and rotation constraints"""
    ik.AddPositionConstraint(
        gripper_frame,
        [0, 0, 0],
        robot_plant.world_frame(),
        pose.translation(),
        pose.translation(),
    )
    # make sure the arm does not go backward
    ik.AddPositionConstraint(gripper_frame, [0, 0, 0], robot_plant.world_frame(), [0.05, -1, 0], [1, 1, 1])
    ik.AddOrientationConstraint(gripper_frame, RotationMatrix(), robot_plant.world_frame(), pose.rotation(), rot_tol)
    if add_gripper_faceup:
        ik.AddAngleBetweenVectorsConstraint(
            gripper_frame, [1, 0, 0], robot_plant.world_frame(), [0, 0, -1], np.pi / 12, np.pi
        )

    # not touching table constraints add elbow
    if add_table_col:
        add_table_collision_free_constraint(ik, robot_plant, gripper_frame, [0.03, 0.04, 0.08])
        add_table_collision_free_constraint(
            ik,
            robot_plant,
            robot_plant.GetFrameByName("panda_link6"),
            [0.03, 0.03, 0.03],
        )
        add_table_collision_free_constraint(
            ik,
            robot_plant,
            robot_plant.GetFrameByName("panda_link7"),
            [0.03, 0.03, 0.03],
        )

    if consider_collision:
        ik.AddMinimumDistanceConstraint(0.01)  # 0.03

    prog = ik.get_mutable_prog()
    q = ik.q()

    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
    prog.SetInitialGuess(q, q0)
    try:
        solver = SnoptSolver()
        result = solver.Solve(ik.prog())
    except:
        solver = IpoptSolver()
        result = solver.Solve(ik.prog())

    if result.is_success():
        if result is not None and filter_manipulability:
            det_J, cond_J = compute_manipulability(robot_plant, ik_context)
            if det_J < 0.001 or cond_J > 180:
                print(f"not enough maneuverability det_J: {det_J:.3f} cond_J: {cond_J:.3f}")
                result = None

        return result
    else:
        return None


def add_tool_to_controller_plant(plant, tool_class_name, instance_info, task_config, tool_rel_pose):
    """add the tool to the controller plant"""
    tool_info = AddTools(plant, tool_class_name, instance_info, train=task_config.train)
    plant.WeldFrames(
        plant.GetFrameByName("panda_hand"),
        plant.GetFrameByName(tool_info[1], tool_info[0]),
        tool_rel_pose,
    )


def transform_point(pose, pc):
    """transform points with the given pose
    param pose: 4 x 4
    param pc: 3 x N
    """
    return pose[:3, :3] @ pc + pose[:3, [3]]


def downsample_point(pc, num_pt):
    """param pc: 3 x N
    return 3 x num_pt
    """
    if type(pc) is not torch.Tensor:
        pc = torch.from_numpy(pc).cuda().float()
    if len(pc.shape) == 2:
        pc = pc[None].cuda().float()
    if pc.shape[1] == 0:
        return torch.zeros((pc.shape[-1], 0)).float()

    try:
        # to mimic what real-world pointcloud does
        xyz = gather_operation(
            pc.transpose(1, 2).contiguous(),
            furthest_point_sample(pc[..., :3].contiguous(), num_pt),
        ).contiguous()

    except:
        # farthest point sampling is not available
        print(pc.shape)
        print("Use naive sampling. Consider installing [PointNet++](https://github.com/liruiw/Pointnet2_PyTorch)")
        pc = pc.detach().cpu().numpy()
        sample_indexes = np.random.choice(pc.shape[1], num_pt)  # ALLOW replace
        pc = pc[0, sample_indexes].T
        print(np.unique(sample_indexes))
        return torch.from_numpy(pc).float()

    return xyz[0].detach()


def finalize_builder(
    plant,
    builder,
    scene_graph,
    franka,
    time_step,
    tool_class_name,
    task_config=None,
    tool_name=None,
    object_name=None,
    tool_rel_pose=None,
    object_info=None,
):
    """add the controller / sensors and related ports to the plant"""
    from env.env_object_related import extra_force_system

    add_contact_visualizer = task_config.env.vis_contact
    use_impedance_controller = task_config.use_impedance_controller
    use_meshcat = task_config.use_meshcat

    s = time.time()
    num_positions = plant.num_positions(franka)
    demux = builder.AddSystem(Demultiplexer(2 * num_positions, num_positions))
    builder.Connect(plant.get_state_output_port(franka), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), "panda_position_measured")
    builder.ExportOutput(demux.get_output_port(1), "panda_velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(franka), "panda_state_estimated")

    multibody_plant_config = MultibodyPlantConfig(
        time_step=time_step,
        discrete_contact_solver=task_config.sim.discrete_contact_solver_type,
    )
    controller_builder = DiagramBuilder()
    controller_plant, controller_scene_graph = AddMultibodyPlant(multibody_plant_config, controller_builder)
    controller_panda = AddFranka(controller_plant)

    # tool is added to the controller plant in the real world. This would cause external torque.
    if tool_rel_pose is not None and "mug" not in task_config.task_name:  # hack around mug
        add_tool_to_controller_plant(
            controller_plant,
            tool_class_name,
            {"tool_name": tool_name},
            task_config,
            tool_rel_pose=tool_rel_pose,
        )

    # the controller plant needs to feel the extra torque
    controller_plant.Finalize()  # match two plants
    controller_diagram = controller_builder.Build()

    # I need a PassThrough system so that I can export the input port.
    panda_positions = builder.AddSystem(PassThrough([0] * num_positions))
    panda_kp = [task_config.env.controller_gains[0]] * num_positions  # [120, 100, 100, 80, 30, 20, 50, 50, 50]
    panda_kd = [task_config.env.controller_gains[2]] * num_positions  # critically damped otherwise
    adder = builder.AddSystem(Adder(2, num_positions))

    if use_impedance_controller:
        # https://github.com/frankaemika/franka_ros/blob/develop/franka_control/config/default_controllers.yaml#L31
        # https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/ca7a4fb762f9581e39cc2aab644f18a83d6ab0ba/isaacgymenvs/tasks/factory/factory_control.py
        impedance_controller = RobotInternalController(controller_plant, joint_stiffness=panda_kp, kd=panda_kd)
        franka_controller = builder.AddSystem(impedance_controller)
        builder.Connect(
            panda_positions.get_output_port(),
            franka_controller.get_input_port_desired_state(),
        )
        panda_controller_torque = builder.AddSystem(PassThrough([0] * num_positions))
        builder.Connect(
            panda_controller_torque.get_output_port(),
            franka_controller.get_torque_feedforward_port_control(),
        )

        builder.Connect(
            plant.get_state_output_port(franka),
            franka_controller.get_input_port_estimated_state(),
        )
        builder.Connect(franka_controller.get_output_port_control(), adder.get_input_port(0))

    elif task_config.pose_controller or task_config.use_controller_portswitch:
        # use eric's controller
        # input port : X_TPdes, V_TPdes
        # output port: torques_output
        franka_controller = PoseController(
            controller_plant,
            controller_panda,
            controller_plant.GetFrameByName("panda_link0"),
            controller_plant.GetFrameByName("panda_hand"),
        )
        franka_controller = builder.AddSystem(franka_controller)
        desired_eff_pose = builder.AddSystem(PassThrough(AbstractValue.Make(RigidTransform())))
        desired_eff_velocity = builder.AddSystem(PassThrough(AbstractValue.Make(SpatialVelocity())))
        builder.Connect(
            plant.get_state_output_port(franka),
            franka_controller.get_plant_state_input_port(),
        )

        builder.Connect(
            desired_eff_pose.get_output_port(),
            franka_controller.get_state_desired_port(),
        )
        builder.Connect(
            desired_eff_velocity.get_output_port(),
            franka_controller.get_velocity_desired_port(),
        )

        builder.ExportInput(desired_eff_pose.get_input_port(), "desired_eff_pose")
        builder.ExportInput(desired_eff_velocity.get_input_port(), "desired_eff_velocity")

        if task_config.use_controller_portswitch:
            # add the inverse dynamics controller together
            controller_port_switch = builder.AddSystem(PortSwitch(vector_size=9))
            osc_port = controller_port_switch.DeclareInputPort(name="operational_space_controller")
            inv_dyn_port = controller_port_switch.DeclareInputPort(name="inverse_dynamics_controller")

            inverse_dynamic_controller = InverseDynamicsController(
                controller_plant,
                kp=panda_kp,  # 100
                ki=[1] * num_positions,  # 1
                kd=panda_kd,  # 1 for less oscillating motion and 20 for reaching
                has_reference_acceleration=False,
            )

            # joint position controller. The controller block
            inv_dyn_franka_controller = builder.AddSystem(inverse_dynamic_controller)

            # Add discrete derivative to command velocities. The IK block
            desired_state_from_position = builder.AddSystem(
                StateInterpolatorWithDiscreteDerivative(num_positions, time_step, suppress_initial_transient=True)
            )
            desired_state_from_position.set_name("desired_state_from_position")
            builder.Connect(
                desired_state_from_position.get_output_port(),
                inv_dyn_franka_controller.get_input_port_desired_state(),
            )

            builder.Connect(
                panda_positions.get_output_port(),
                desired_state_from_position.get_input_port(),
            )
            builder.Connect(
                plant.get_state_output_port(franka),
                inv_dyn_franka_controller.get_input_port_estimated_state(),
            )
            builder.Connect(inv_dyn_franka_controller.get_output_port(), inv_dyn_port)  # 2
            builder.Connect(franka_controller.get_output_port(), osc_port)  # 1
            builder.Connect(controller_port_switch.get_output_port(), adder.get_input_port(0))

            builder.ExportInput(
                controller_port_switch.get_port_selector_input_port(),
                "controller_switch_port",
            )
        else:
            builder.Connect(franka_controller.get_output_port(), adder.get_input_port(0))
    else:
        # too much oscillation for learner if high kd, too much oscillation for the expert if high kp tuning required
        inverse_dynamic_controller = InverseDynamicsController(
            controller_plant,
            kp=panda_kp,  # 100
            ki=[1] * num_positions,  # 1
            kd=panda_kd,  # 1 for less oscillating motion and 20 for reaching
            has_reference_acceleration=False,
        )

        # joint position controller. The controller block
        franka_controller = builder.AddSystem(inverse_dynamic_controller)

        # Add discrete derivative to command velocities. The IK block
        desired_state_from_position = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(num_positions, time_step, suppress_initial_transient=True)
        )
        desired_state_from_position.set_name("desired_state_from_position")

        builder.Connect(
            desired_state_from_position.get_output_port(),
            franka_controller.get_input_port_desired_state(),
        )
        builder.Connect(
            panda_positions.get_output_port(),
            desired_state_from_position.get_input_port(),
        )
        builder.Connect(
            plant.get_state_output_port(franka),
            franka_controller.get_input_port_estimated_state(),
        )
        builder.Connect(franka_controller.get_output_port_control(), adder.get_input_port(0))

    franka_controller.set_name("panda_controller")
    # Add in the feed-forward torque
    # Use a PassThrough to make the port optional (it will provide zero values if not connected).

    torque_passthrough = builder.AddSystem(PassThrough([0] * num_positions))
    builder.Connect(torque_passthrough.get_output_port(), adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(), "feedforward_torque")
    builder.Connect(adder.get_output_port(), plant.get_actuation_input_port(franka))
    builder.ExportInput(panda_positions.get_input_port(), "panda_joint_commanded")

    # Export commanded torques.
    builder.ExportOutput(adder.get_output_port(), "panda_torque_commanded")
    builder.ExportOutput(adder.get_output_port(), "panda_torque_measured")
    builder.ExportOutput(
        plant.get_generalized_contact_forces_output_port(franka),
        "panda_torque_external",
    )

    # Inverse Kinematics
    params = DifferentialInverseKinematicsParameters(num_positions, num_positions)
    ik_timestep = time_step * 160  # differentialIK...
    velocity_limit_scale = 2.1750  # from ROS
    franka_velocity_limits = np.ones(9) * velocity_limit_scale

    # differential IK is now separated by another plant
    global robot_plant
    robot_builder = DiagramBuilder()
    robot_plant, _ = AddMultibodyPlant(multibody_plant_config, robot_builder)
    robot_panda = AddFranka(robot_plant)
    robot_plant.Finalize()

    differential_ik = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            robot_plant,
            robot_plant.GetFrameByName("panda_hand"),
            ik_timestep,
            params,
        )
    )
    differential_ik.get_mutable_parameters().set_nominal_joint_position(
        [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
    )
    builder.ExportInput(differential_ik.get_input_port(0), "panda_hand_pose")
    builder.ExportOutput(differential_ik.get_output_port(), "differential_ik_output")

    if task_config.task_name == "wrench" or task_config.task_name == "hammer":
        builder.Connect(
            extra_force_system.get_output_port(0),
            plant.get_applied_spatial_force_input_port(),
        )
        builder.Connect(
            extra_force_system.get_output_port(1),
            plant.get_actuation_input_port(object_info[0]),
        )

    wrist_cam_view = HAND_CAM_EXTR.copy()
    overhead_cam_view = OVERHEAD_CAM_EXTR.copy()

    if (
        hasattr(task_config.env, "randomize_camera_extrinsics")
        and task_config.env.randomize_camera_extrinsics
        and task_config.train
    ):
        translation_noise = 0.02
        rotation_noise = 0.05
        augment_noise = np.concatenate(
            (
                np.random.uniform(-1, 1, size=(3,)) * translation_noise,
                np.random.uniform(-1, 1, size=(3,)) * rotation_noise,
            ),
            axis=-1,
        )
        augment_noise_pose = unpack_action(augment_noise)
        wrist_cam_view = wrist_cam_view @ augment_noise_pose

        # one way to choose the overhead camera distribution is to shift it to some sphere that is x axis some distance away from the table center
        target_center = np.array([0.45, 0.0, 0.3])
        overhead_cam_view = randomize_sphere_lookat(
            target_center, near=1.5, far=2.5, x_near=0.7, x_far=1.5, theta_low=np.pi / 8, theta_high=np.pi / 4
        )

    # Overhead Cameras
    AddRgbdSensors(
        builder,
        plant,
        scene_graph,
        model_instance_prefix="panda",
        relative_pose=overhead_cam_view,
        attach_body_name="panda_link0",
    )

    # Wrist Cameras
    AddRgbdSensors(
        builder,
        plant,
        scene_graph,
        model_instance_prefix="panda",
        relative_pose=wrist_cam_view,
        attach_body_name="panda_hand",
    )
    if not task_config.blender_render:
        AddTriad(
            plant.get_source_id(),
            plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("panda_link0").index()),
            scene_graph,
            length=0.05,
            radius=0.005,
            X_FT=RigidTransform(overhead_cam_view),
            name="overhead_cam_coord",
        )

        AddTriad(
            plant.get_source_id(),
            plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("panda_hand").index()),
            scene_graph,
            length=0.05,
            radius=0.005,
            X_FT=RigidTransform(HAND_CAM_EXTR),
            name="hand_cam_coord",
        )

    # create a dummy body and attach a sensor to it.
    AddRgbdSensors(
        builder,
        plant,
        scene_graph,
        model_instance_prefix="dummy",
        relative_pose=np.eye(4),
        attach_body_name="dummy_cam_body",
    )
    ##########################################

    # add the tool frame and the object frame
    if tool_name is not None and not task_config.blender_render:
        AddTriad(
            plant.get_source_id(),
            plant.GetBodyFrameIdOrThrow(plant.GetBodyByName(tool_name).index()),
            scene_graph,
            length=0.05,
            radius=0.005,
            name="tool_coord",
        )

    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(plant.get_contact_results_output_port(), "contact_results")
    builder.ExportOutput(plant.get_state_output_port(), "plant_continuous_state")

    global meshcat, visualizer
    if use_meshcat:
        if not meshcat:
            meshcat = StartMeshcat()
            if task_config.verbose:
                print("meshcat url: ", meshcat.web_url())
            meshcat.ResetRenderMode()
            meshcat.SetProperty("/Background", "visible", False)
            meshcat.SetProperty("/Grid", "visible", False)
            meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.5)
            meshcat.SetRealtimeRate(task_config.sim.realtime_rate)

        visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration, delete_on_initialization_event=False),
        )
        collision_visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
        )

        if not task_config.vis_collision_geometry:
            meshcat.SetProperty("collision", "visible", False)

        if add_contact_visualizer:
            cparams = ContactVisualizerParams()
            cparams.force_threshold = 1e-2
            cparams.newtons_per_meter = 1e4  # 1e-6 .0 # 1.0
            cparams.newton_meters_per_meter = 1e1
            cparams.radius = 0.005
            contact_visualizer = ContactVisualizer.AddToBuilder(builder, plant, meshcat, cparams)

    state_logger = LogVectorOutput(differential_ik.get_output_port(), builder, 10.0)
    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return [
        builder,
        state_logger,
        diagram,
        plant,
        franka_controller,
        controller_plant,
        controller_diagram,
        differential_ik,
        scene_graph,
    ]


def domain_randomization(scene_graph, inspector, plant, tool_info, tool_obj_info, task_config):
    """domain randomzation for the textures and physics of the object and tools"""
    tool_name, object_name = tool_info[1], tool_obj_info[1]
    tool_id, object_id = tool_info[0], tool_obj_info[0]

    randomize_physics = task_config.env.randomize_physics
    hydroelastic_contact = task_config.sim.hydroelastic_contact
    randomize_textures = task_config.env.randomize_textures
    compliant_hydroelastic_model = task_config.sim.compliant_hydroelastic_model
    frictions_hyper_param = [
        task_config.sim.friction_dissipation,
        task_config.sim.friction_mu_static,
        task_config.sim.friction_mu_dynamic,
    ]

    UpdateObjectPhysicalProperties(
        scene_graph,
        inspector,
        plant.get_source_id(),
        plant.GetBodyFrameIdOrThrow(plant.GetBodyByName(tool_name).index()),
        tool_name,
        randomize_physics=randomize_physics,
        hydroelastic_contact=hydroelastic_contact,
        compliant_model=compliant_hydroelastic_model,
        frictions_hyper_param=frictions_hyper_param,
    )

    UpdateObjectVisualProperties(
        scene_graph,
        inspector,
        plant.get_source_id(),
        plant.GetBodyFrameIdOrThrow(plant.GetBodyByName(tool_name).index()),
        tool_name,
        color=(1, 0, 0, 1),
        randomize_textures=randomize_textures,
    )

    UpdateObjectPhysicalProperties(
        scene_graph,
        inspector,
        plant.get_source_id(),
        plant.GetBodyFrameIdOrThrow(plant.GetBodyByName(object_name).index()),
        object_name,
        randomize_physics=randomize_physics,
        hydroelastic_contact=hydroelastic_contact,
        compliant_model=compliant_hydroelastic_model,
        frictions_hyper_param=frictions_hyper_param,
    )

    UpdateObjectVisualProperties(
        scene_graph,
        inspector,
        plant.get_source_id(),
        plant.GetBodyFrameIdOrThrow(plant.GetBodyByName(object_name).index()),
        object_name,
        randomize_textures=randomize_textures,
        color=(1, 0, 0, 1),
        alpha=0.3,
    )

    if task_config.task_name == "wrench":
        nut_object_name = object_name.replace("body", "nut_body")
        UpdateObjectPhysicalProperties(
            scene_graph,
            inspector,
            plant.get_source_id(),
            plant.GetBodyFrameIdOrThrow(plant.GetBodyByName(nut_object_name).index()),
            nut_object_name,
            randomize_physics=randomize_physics,
            hydroelastic_contact=hydroelastic_contact,
            compliant_model=compliant_hydroelastic_model,
            frictions_hyper_param=frictions_hyper_param,
        )

        UpdateObjectVisualProperties(
            scene_graph,
            inspector,
            plant.get_source_id(),
            plant.GetBodyFrameIdOrThrow(plant.GetBodyByName(nut_object_name).index()),
            nut_object_name,
            randomize_textures=randomize_textures,
            color=(1, 0, 0, 1),
            alpha=0.3,
        )

    # table
    UpdateObjectPhysicalProperties(
        scene_graph,
        inspector,
        plant.get_source_id(),
        plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("link").index()),
        "::surface",
        randomize_physics=randomize_physics,
        hydroelastic_contact=hydroelastic_contact,
        compliant_model=compliant_hydroelastic_model,
        frictions_hyper_param=[0.5, 0.1, 0.1],  # low friction for the table
    )

    UpdateObjectVisualProperties(
        scene_graph,
        inspector,
        plant.get_source_id(),
        plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("link").index()),
        "::visual1",
        randomize_textures=randomize_textures,
    )


def update_tool_pose(tool_class_name, tool_info, randomize_tool_pose, task_config):
    """
    Update the tool pose based on domain randomization.

    Args:
        tool_class_name (str): The name of the tool class.
        tool_info (list): Information about the tool.
        randomize_tool_pose (bool): Flag indicating whether to randomize the tool pose.
        task_config (TaskConfig): The task configuration.

    Returns:
        RigidTransform: The updated tool pose.

    """
    tool_rel_pose = [[0, 0, 0], [0, 0, 0.15]]
    # domain randomization
    tool_name = tool_info[1]

    if randomize_tool_pose:
        # pitch augmentation
        # roll pitch yaw
        tool_rel_pose[0][1] = np.random.uniform(
            task_config.env.tool_rotation_y_range[0],
            task_config.env.tool_rotation_y_range[1],
        )
        tool_rel_pose[0][0] = np.random.uniform(
            task_config.env.tool_rotation_z_range[0],
            task_config.env.tool_rotation_z_range[1],
        )
        tool_rel_pose[0][2] = np.random.uniform(
            task_config.env.tool_rotation_x_range[0],
            task_config.env.tool_rotation_x_range[1],
        )

        # add perturbation / to make the tool tilted down
        # translation in z
        tool_rel_pose[1][2] = np.random.uniform(
            task_config.env.tool_translation_z_range[0],
            task_config.env.tool_translation_z_range[1],
        )
        tool_rel_pose[1][0] = np.random.uniform(
            task_config.env.tool_translation_x_range[0],
            task_config.env.tool_translation_x_range[1],
        )
    return RigidTransform(RollPitchYaw(*tool_rel_pose[0]), tool_rel_pose[1])


def MakeManipulationStation(time_step=0.005, task_config=None, instance_info={}):
    """
    Creates a manipulation station for performing tasks.

    Args:
        time_step (float, optional): The time step for the simulation. Defaults to 0.005.
        task_config (object, optional): The task configuration object. Defaults to None.
        instance_info (dict, optional): Additional instance information. Defaults to {}.

    Returns:
        list: A list containing the finalized builder, model inspector, and other relevant objects.
    """
    builder = DiagramBuilder()
    multibody_plant_config = MultibodyPlantConfig(time_step=time_step)
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    franka = AddFranka(plant)
    AddTable(plant, task_config.blender_render)
    plant.Finalize()
    res = finalize_builder(plant, builder, scene_graph, franka, time_step, "", task_config)
    inspector = scene_graph.model_inspector()
    res.append(inspector)

    global meshcat
    vis_hand_pose(meshcat, np.eye(4), "hand_goal_pose", load=True)
    return res


def MakeManipulationStationTools(time_step=0.005, tool_class_name="pole", task_config=None, instance_info={}):
    """make the manipulation station with tools"""
    reset_global_variables()
    s = time.time()
    builder = DiagramBuilder()
    multibody_plant_config = MultibodyPlantConfig(
        time_step=time_step,
        contact_surface_representation=task_config.sim.contact_surface_representation,
        contact_model=task_config.sim.contact_model,
        discrete_contact_solver=task_config.sim.discrete_contact_solver_type,
    )
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    franka = AddFranka(plant)
    AddTable(plant, task_config.blender_render)

    # tools
    tool_obj_info = AddToolObjects(
        builder,
        plant,
        scene_graph,
        task_config.task_name,  # actually the task determines which object to use
        instance_info,
        train=task_config.train,
        fixed_idx=task_config.tool_fix_idx,
    )  #
    tool_info = AddTools(
        plant, tool_class_name, instance_info, train=task_config.train, fixed_idx=task_config.tool_fix_idx
    )
    tool_rel_pose = update_tool_pose(tool_class_name, tool_info, task_config.env.randomize_tool_pose, task_config)
    plant.WeldFrames(
        plant.GetFrameByName("panda_hand"),
        plant.GetFrameByName(tool_info[1], tool_info[0]),
        tool_rel_pose,
    )
    if not task_config.blender_render:
        AddCameraBox(
            plant,
            RigidTransform(OVERHEAD_CAM_EXTR),
            name="overhead_cam",
            parent_frame=plant.world_frame(),
        )

    # add a dummy camera
    AddShape(
        plant,
        Box(1e-3, 1e-3, 1e-3),
        "dummy_cam_body",
        1,
        0,  # mass to be 0
        color=[0.1, 0.5, 0.1, 0.0],
        collision=False,
    )

    plant.Finalize()
    tool_name, object_name = tool_info[1], tool_obj_info[1]
    inspector = scene_graph.model_inspector()
    domain_randomization(scene_graph, inspector, plant, tool_info, tool_obj_info, task_config)
    res = finalize_builder(
        plant,
        builder,
        scene_graph,
        franka,
        time_step,
        tool_class_name,
        task_config,
        tool_name=tool_name,
        object_name=object_name,
        tool_rel_pose=tool_rel_pose,
        object_info=tool_obj_info,
    )
    res.append(inspector)

    global meshcat
    if not task_config.blender_render:
        vis_tool_pose(meshcat, np.eye(4), "tool_goal_pose", mesh_path=tool_info[-1])
        vis_hand_pose(meshcat, np.eye(4), "hand_goal_pose", load=True)

    if task_config.verbose:
        print(f"object name: {object_name} tool name: {tool_name}")
        print(f"make manipulation time: {(time.time() - s):.3f}")
    tool_info.append(tool_rel_pose.GetAsMatrix4())

    return res + [tool_obj_info, tool_info]
