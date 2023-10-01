# From Russ's code in manipulation repo
from functools import partial
import os
import sys
import time

from IPython.display import display, HTML, Javascript
import numpy as np


from pydrake.geometry import Cylinder, Rgba, Sphere
from pydrake.perception import PointCloud, Fields, BaseField
from pydrake.all import BoundingBoxConstraint

# imports for the pose sliders
from collections import namedtuple
from pydrake.all import AbstractValue
from pydrake.all import RollPitchYaw, RigidTransform, RotationMatrix
from pydrake.all import LeafSystem, PublishEvent

# imports for the joint sliders
from pydrake.all import JointIndex
from pydrake.all import PublishEvent, VectorSystem

from pydrake.all import *

# Some GUI code that will be moved into Drake.
import IPython


class DrawContactWrench(LeafSystem):
    def __init__(self, meshcat):
        LeafSystem.__init__(self)
        self.DeclareAbstractInputPort("contact_results", AbstractValue.Make(ContactResults()))
        self.DeclareForcedPublishEvent(self.Publish)
        self.meshcat = meshcat

    def Publish(self, context):
        results = self.get_input_port().Eval(context)

        for i in range(results.num_point_pair_contacts()):
            info = results.point_pair_contact_info(i)
            self.meshcat.SetObject(
                f"contact_{i}", MeshcatCone(height, height * mu, height * mu), rgba=Rgba(0.1, 0.9, 0.1, 1.0)
            )
            p_WC = info.contact_point()
            R_WC = RotationMatrix.MakeFromOneVector(info.point_pair().nhat_BA_W, 2)
            X_WC = RigidTransform(R_WC, p_WC)
            self.meshcat.SetTransform(f"contact_{i}", X_WC)

            X_WB = RigidTransform()  # box center is at world origin
            self.meshcat.SetObject(
                f"box_center/contact_{i}/translational",
                MeshcatCone(height, height * mu, height * mu),
                rgba=Rgba(0.7, 0.1, 0.1, 1.0),
            )
            self.meshcat.SetTransform("box_center", X_WB)
            R_BC = X_WB.inverse().rotation() @ R_WC
            self.meshcat.SetTransform(f"box_center/contact_{i}", RigidTransform(R_BC))
            p_CB_C = -(R_WC @ (X_WB.inverse() @ p_WC))
            A_CCtau = VectorToSkewSymmetric(p_CB_C)
            # Three.js does not fully support non-uniform scaling
            # https://github.com/mrdoob/three.js/issues/15079.
            # So I cannot simply draw the meshcatcone and transform it.
            # Instead I will create the vertices myself.
            cloud.mutable_xyzs()[:] = A_CCtau @ vertices
            self.meshcat.SetObject(
                f"box_center/contact_{i}/rotational_cloud", cloud, point_size=0.002, rgba=Rgba(0, 1, 0, 1)
            )
            # TODO(russt): Figure out colors.  The surface mesh example works ok in meshcat_manual_test...
            self.meshcat.SetTriangleMesh(
                f"box_center/contact_{i}/rotational", A_CCtau @ vertices, faces, rgba=Rgba(0, 0, 1, 1)
            )

        clear_output(wait=True)


class MeshcatSliders(LeafSystem):
    """
    A system that outputs the ``value``s from meshcat sliders.

    .. pydrake_system::

      name: MeshcatSliderSystem
      output_ports:
      - slider_group_0
      - ...
      - slider_group_{N-1}
    """

    def __init__(self, meshcat, slider_names):
        """
        An output port is created for each element in the list `slider_names`.
        Each element of `slider_names` must itself be an iterable collection
        (list, tuple, set, ...) of strings, with the names of sliders that have
        *already* been added to Meshcat via Meshcat.AddSlider().

        The same slider may be used in multiple ports.
        """
        LeafSystem.__init__(self)

        self._meshcat = meshcat
        self._sliders = slider_names
        for i, slider_iterable in enumerate(self._sliders):
            port = self.DeclareVectorOutputPort(
                f"slider_group_{i}",
                len(slider_iterable),
                partial(self.DoCalcOutput, port_index=i),
            )
            port.disable_caching_by_default()

    def DoCalcOutput(self, context, output, port_index):
        for i, slider in enumerate(self._sliders[port_index]):
            output[i] = self._meshcat.GetSliderValue(slider)


class MeshcatPoseSliders(LeafSystem):
    """
    Provides a set of ipywidget sliders (to be used in a Jupyter notebook) with
    one slider for each of roll, pitch, yaw, x, y, and z.  This can be used,
    for instance, as an interface to teleoperate the end-effector of a robot.

    .. pydrake_system::

        name: PoseSliders
        output_ports:
        - pose
    """

    # TODO(russt): Use namedtuple defaults parameter once we are Python >= 3.7.
    Visible = namedtuple("Visible", ("roll", "pitch", "yaw", "x", "y", "z"))
    Visible.__new__.__defaults__ = (True, True, True, True, True, True)
    MinRange = namedtuple("MinRange", ("roll", "pitch", "yaw", "x", "y", "z"))
    MinRange.__new__.__defaults__ = (-np.pi, -np.pi, -np.pi, -1.0, -1.0, -1.0)
    MaxRange = namedtuple("MaxRange", ("roll", "pitch", "yaw", "x", "y", "z"))
    MaxRange.__new__.__defaults__ = (np.pi, np.pi, np.pi, 1.0, 1.0, 1.0)
    Value = namedtuple("Value", ("roll", "pitch", "yaw", "x", "y", "z"))
    Value.__new__.__defaults__ = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __init__(
        self,
        meshcat,
        visible=Visible(),
        min_range=MinRange(),
        max_range=MaxRange(),
        value=Value(),
    ):
        """
        Args:
            meshcat: A Meshcat instance.
            visible: An object with boolean elements for 'roll', 'pitch',
                     'yaw', 'x', 'y', 'z'; the intention is for this to be the
                     PoseSliders.Visible() namedtuple.  Defaults to all true.
            min_range, max_range, value: Objects with float values for 'roll',
                      'pitch', 'yaw', 'x', 'y', 'z'; the intention is for the
                      caller to use the PoseSliders.MinRange, MaxRange, and
                      Value namedtuples.  See those tuples for default values.
        """
        LeafSystem.__init__(self)
        port = self.DeclareAbstractOutputPort("pose", lambda: AbstractValue.Make(RigidTransform()), self.DoCalcOutput)

        # The widgets themselves have undeclared state.  For now, we accept it,
        # and simply disable caching on the output port.
        # TODO(russt): consider implementing the more elaborate methods seen
        # in, e.g., LcmMessageSubscriber.
        port.disable_caching_by_default()

        self._meshcat = meshcat
        self._visible = visible
        self._value = list(value)

        for i in range(6):
            if visible[i]:
                meshcat.AddSlider(
                    min=min_range[i],
                    max=max_range[i],
                    value=value[i],
                    step=0.01,
                    name=value._fields[i],
                )

    def __del__(self):
        for s in ["roll", "pitch", "yaw", "x", "y", "z"]:
            if visible[s]:
                self._meshcat.DeleteSlider(s)

    def SetPose(self, pose):
        """
        Sets the current value of the sliders.

        Args:
            pose: Any viable argument for the RigidTransform
                  constructor.
        """
        tf = RigidTransform(pose)
        self.SetRpy(RollPitchYaw(tf.rotation()))
        self.SetXyz(tf.translation())

    def SetRpy(self, rpy):
        """
        Sets the current value of the sliders for roll, pitch, and yaw.

        Args:
            rpy: An instance of drake.math.RollPitchYaw
        """
        self._value[0] = rpy.roll_angle()
        self._value[1] = rpy.pitch_angle()
        self._value[2] = rpy.yaw_angle()
        for i in range(3):
            if self._visible[i]:
                self._meshcat.SetSliderValue(self._visible._fields[i], self._value[i])

    def SetXyz(self, xyz):
        """
        Sets the current value of the sliders for x, y, and z.

        Args:
            xyz: A 3 element iterable object with x, y, z.
        """
        self._value[3:] = xyz
        for i in range(3, 6):
            if self._visible[i]:
                self._meshcat.SetSliderValue(self._visible._fields[i], self._value[i])

    def _update_values(self):
        changed = False
        for i in range(6):
            if self._visible[i]:
                old_value = self._value[i]
                self._value[i] = self._meshcat.GetSliderValue(self._visible._fields[i])
                changed = changed or self._value[i] != old_value
        return changed

    def _get_transform(self):
        return RigidTransform(
            RollPitchYaw(self._value[0], self._value[1], self._value[2]),
            self._value[3:],
        )

    def DoCalcOutput(self, context, output):
        """Constructs the output values from the sliders."""
        self._update_values()
        output.set_value(self._get_transform())

    def Run(self, publishing_system, root_context, callback):
        # Calls callback(root_context, pose), then publishing_system.Publish()
        # each time the sliders change value.
        if not running_as_notebook:
            return

        publishing_context = publishing_system.GetMyContextFromRoot(root_context)

        print("Press the 'Stop PoseSliders' button in Meshcat to continue.")
        self._meshcat.AddButton("Stop PoseSliders")
        while self._meshcat.GetButtonClicks("Stop PoseSliders") < 1:
            if self._update_values():
                callback(root_context, self._get_transform())
                publishing_system.Publish(publishing_context)
            time.sleep(0.1)

        self._meshcat.DeleteButton("Stop PoseSliders")


class WsgButton(LeafSystem):
    def __init__(self, meshcat):
        LeafSystem.__init__(self)
        port = self.DeclareVectorOutputPort("wsg_position", 1, self.DoCalcOutput)
        port.disable_caching_by_default()
        self._meshcat = meshcat
        self._button = "Open/Close Gripper"
        meshcat.AddButton(self._button)

    def __del__(self):
        self._meshcat.DeleteButton(self._button)

    def DoCalcOutput(self, context, output):
        position = 0.107  # open
        if (self._meshcat.GetButtonClicks(self._button) % 2) == 1:
            position = 0.002  # close
        output.SetAtIndex(0, position)


# TODO(russt): Add floating base support (as in manipulation.jupyter_widgets
# MakeJointSlidersThatPublish)
class MeshcatJointSliders(LeafSystem):
    """
    Adds one slider per joint of the MultibodyPlant.  Any positions that are
    not associated with joints (e.g. floating-base "mobilizers") are held
    constant at the default value obtained from robot.CreateDefaultContext().

    .. pydrake_system::

        name: JointSliders
        output_ports:
        - positions

    In addition to being used inside a Diagram that is being simulated with
    Simulator, this class also offers a `Run` method that runs its own simple
    event loop, querying the slider values and calling `Publish`.  It does not
    simulate any state dynamics.
    """

    def __init__(
        self,
        meshcat,
        plant,
        root_context=None,
        lower_limit=-10.0,
        upper_limit=10.0,
        resolution=0.01,
    ):
        """
        Creates an meshcat slider for each joint in the plant.

        Args:
            meshcat:      A Meshcat instance.
            plant:        A MultibodyPlant. publishing_system: The System whose
                          Publish method will be called.  Can be the entire
                          Diagram, but can also be a subsystem.
            root_context: A mutable root Context of the Diagram containing the
                          ``plant``; we will extract the subcontext's using
                          `GetMyContextFromRoot`.
            lower_limit:  A scalar or vector of length robot.num_positions().
                          The lower limit of the slider will be the maximum
                          value of this number and any limit specified in the
                          Joint.

            upper_limit:  A scalar or vector of length robot.num_positions().
                          The upper limit of the slider will be the minimum
                          value of this number and any limit specified in the
                          Joint.

            resolution:   A scalar or vector of length robot.num_positions()
                          that specifies the step argument of the FloatSlider.
        """
        LeafSystem.__init__(self)

        def _broadcast(x, num):
            x = np.array(x)
            assert len(x.shape) <= 1
            return np.array(x) * np.ones(num)

        lower_limit = _broadcast(lower_limit, plant.num_positions())
        upper_limit = _broadcast(upper_limit, plant.num_positions())
        resolution = _broadcast(resolution, plant.num_positions())

        self._meshcat = meshcat
        self._plant = plant
        plant_context = plant.GetMyContextFromRoot(root_context) if root_context else plant.CreateDefaultContext()

        self._sliders = {}
        self._positions = plant.GetPositions(plant_context)
        slider_num = 0
        for i in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(i))
            low = joint.position_lower_limits()
            upp = joint.position_upper_limits()
            for j in range(joint.num_positions()):
                index = joint.position_start() + j
                description = joint.name()
                if joint.num_positions() > 1:
                    description += "_" + joint.position_suffix(j)
                meshcat.AddSlider(
                    value=self._positions[index],
                    min=max(low[j], lower_limit[slider_num]),
                    max=min(upp[j], upper_limit[slider_num]),
                    step=resolution[slider_num],
                    name=description,
                )
                self._sliders[index] = description
                slider_num += 1

        port = self.DeclareVectorOutputPort("positions", plant.num_positions(), self.DoCalcOutput)
        port.disable_caching_by_default()

    def DoCalcOutput(self, context, output):
        output.SetFromVector(self._positions)
        for i, s in self._sliders.items():
            output[i] = self._meshcat.GetSliderValue(s)

    def Run(self, publishing_system, root_context, callback=None):
        """
        Args:
            publishing_system:  The system to call publish on.  Probably a
                          MeshcatVisualizer.
            root_context: A mutable root Context of the Diagram containing both
                          the ``plant`` and the ``publishing_system``; we will
                          extract the subcontext's using `GetMyContextFromRoot`.
            callback: callback(plant_context) will be called whenever the
                      slider values change.
        """
        if not running_as_notebook:
            return
        print("Press the 'Stop JointSliders' button in Meshcat to continue.")
        self._meshcat.AddButton("Stop JointSliders")

        plant_context = self._plant.GetMyContextFromRoot(root_context)
        publishing_context = publishing_system.GetMyContextFromRoot(root_context)

        publishing_system.Publish(publishing_context)
        while self._meshcat.GetButtonClicks("Stop JointSliders") < 1:
            old_positions = self._plant.GetPositions(plant_context)
            positions = self._positions
            for i, s in self._sliders.items():
                positions[i] = self._meshcat.GetSliderValue(s)
            if not np.array_equal(positions, old_positions):
                self._plant.SetPositions(plant_context, positions)
                if callback:
                    callback(plant_context)
                publishing_system.Publish(publishing_context)
            time.sleep(0.1)

        self._meshcat.DeleteButton("Stop JointSliders")


def AddMeshcatVector(meshcat, path, color, length=0.25, radius=0.01, opacity=1.0, X_PT=RigidTransform()):

    meshcat.SetTransform(path, X_PT)
    meshcat.SetObject(path, Cylinder(radius, length), color)


def AddMeshcatTriad(meshcat, path, length=0.25, radius=0.01, opacity=1.0, X_PT=RigidTransform()):
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


def draw_open3d_point_cloud(meshcat, path, pcd, normals_scale=0.0, point_size=0.001, sphere_vis=False):
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # assert pcd.has_colors()  # TODO(russt): handle this case better

    if sphere_vis:
        for idx, pt in enumerate(pts):
            meshcat_geom = Sphere(point_size)
            color = colors[idx]
            point_path = path + f"_point_{idx}"
            meshcat.SetObject(point_path, Sphere(point_size), Rgba(color[0], color[1], color[2], 1.0))
            transform = np.eye(4)
            transform[:3, 3] = pt
            meshcat.SetTransform(point_path, RigidTransform(pt))

    else:
        cloud = PointCloud(pts.shape[0], Fields(BaseField.kXYZs | BaseField.kRGBs))
        cloud.mutable_xyzs()[:] = pts.T
        cloud.mutable_rgbs()[:] = 255 * np.asarray(pcd.colors).T
        meshcat.SetObject(path, cloud, point_size=point_size)

    if pcd.has_normals() and normals_scale > 0.0:
        assert "need to implement LineSegments in meshcat c++"
        normals = np.asarray(pcd.normals)
        vertices = np.hstack((pts, pts + normals_scale * normals)).reshape(-1, 3).T
        meshcat["normals"].set_object(g.Points(g.PointsGeometry(vertices), g.MeshBasicMaterial(color=0x000000)))
        # .set_object(meshcat_geometry.PointCloud(cloud.T, color.T))


# Writing System.
class MeshcatWriter(LeafSystem):
    def __init__(
        self,
        meshcat,
        drawing_body_index,
        canvas_body_index,
        rgba,
        line_width=0.003,
        draw_threshold=0.01,
    ):
        LeafSystem.__init__(self)
        self._meshcat = meshcat
        self._drawing_body_index = drawing_body_index
        self._canvas_body_index = canvas_body_index
        self._rgba = rgba
        self._line_width = line_width
        self._draw_threshold = draw_threshold

        self.DeclareAbstractInputPort("contact_results", AbstractValue.Make(ContactResults()))
        self.DeclarePeriodicDiscreteUpdateEvent(0.01, 0, self.MaybeDraw)

        self._p_WLastDraw_index = self.DeclareDiscreteState(3)
        self._was_in_contact_index = self.DeclareDiscreteState([0])
        self._num_drawn_index = self.DeclareDiscreteState([0])

    def MaybeDraw(self, context, discrete_state):
        results = self.get_input_port().Eval(context)

        # The point on the canvas that most deeply penetrates the chalk.
        p_WDraw = None
        for i in range(results.num_point_pair_contacts()):
            info = results.point_pair_contact_info(i)
            if info.bodyA_index() == self._drawing_body_index and info.bodyB_index() == self._canvas_body_index:
                p_WDraw = info.point_pair().p_WCb
                break
            elif info.bodyB_index() == self._drawing_body_index and info.bodyA_index() == self._canvas_body_index:
                p_WDraw = info.point_pair().p_WCa
                break

        if p_WDraw is not None:
            p_WLastDraw = context.get_discrete_state(self._p_WLastDraw_index).get_value()
            was_in_contact = context.get_discrete_state(self._was_in_contact_index)[0]
            num_drawn = context.get_discrete_state(self._num_drawn_index)[0]

            length = np.linalg.norm(p_WDraw - p_WLastDraw)
            if was_in_contact and length > self._draw_threshold:
                self._meshcat.SetObject(f"writer/{num_drawn}", Capsule(self._line_width, length), self._rgba)
                p_WMidpoint = (p_WDraw + p_WLastDraw) / 2
                X_WCapsule = RigidTransform(
                    RotationMatrix.MakeFromOneVector(p_WDraw - p_WLastDraw, 2),
                    p_WMidpoint,
                )
                self._meshcat.SetTransform(f"writer/{num_drawn}", X_WCapsule)
                discrete_state.set_value(self._num_drawn_index, [num_drawn + 1])
                discrete_state.set_value(self._p_WLastDraw_index, p_WDraw)
            elif not was_in_contact:
                discrete_state.set_value(self._p_WLastDraw_index, p_WDraw)

            discrete_state.set_value(self._was_in_contact_index, [1])
        else:
            discrete_state.set_value(self._was_in_contact_index, [0])


def plot_surface(
    meshcat,
    path,
    X,
    Y,
    Z,
    rgba=Rgba(0.87, 0.6, 0.6, 1.0),
    wireframe=False,
    wireframe_line_width=1.0,
):
    (rows, cols) = Z.shape
    assert np.array_equal(X.shape, Y.shape)
    assert np.array_equal(X.shape, Z.shape)

    vertices = np.empty((rows * cols, 3), dtype=np.float32)
    vertices[:, 0] = X.reshape((-1))
    vertices[:, 1] = Y.reshape((-1))
    vertices[:, 2] = Z.reshape((-1))

    # Vectorized faces code from https://stackoverflow.com/questions/44934631/making-grid-triangular-mesh-quickly-with-numpy  # noqa
    faces = np.empty((rows - 1, cols - 1, 2, 3), dtype=np.uint32)
    r = np.arange(rows * cols).reshape(rows, cols)
    faces[:, :, 0, 0] = r[:-1, :-1]
    faces[:, :, 1, 0] = r[:-1, 1:]
    faces[:, :, 0, 1] = r[:-1, 1:]
    faces[:, :, 1, 1] = r[1:, 1:]
    faces[:, :, :, 2] = r[1:, :-1, None]
    faces.shape = (-1, 3)

    # TODO(Russ): support per vertex / Colormap colors.
    meshcat.SetTriangleMesh(path, vertices.T, faces.T, rgba, wireframe, wireframe_line_width)


def plot_mathematical_program(meshcat, path, prog, X, Y, result=None, point_size=0.05):
    assert prog.num_vars() == 2
    assert X.size == Y.size

    N = X.size
    values = np.vstack((X.reshape(-1), Y.reshape(-1)))
    costs = prog.GetAllCosts()

    # Vectorized multiply for the quadratic form.
    # Z = (D*np.matmul(Q,D)).sum(0).reshape(nx, ny)

    if costs:
        Z = prog.EvalBindingVectorized(costs[0], values)
        for b in costs[1:]:
            Z = Z + prog.EvalBindingVectorized(b, values)

    cv = f"{path}/constraints"
    for binding in prog.GetAllConstraints():
        if isinstance(binding.evaluator(), BoundingBoxConstraint):
            c = binding.evaluator()
            var_indices = [int(prog.decision_variable_index()[v.get_id()]) for v in binding.variables()]
            satisfied = np.array(c.CheckSatisfiedVectorized(values[var_indices, :], 0.001)).reshape(1, -1)
            if costs:
                Z[~satisfied] = np.nan

            v = f"{cv}/{type(c).__name__}"
            Zc = np.zeros(Z.shape)
            Zc[satisfied] = np.nan
            plot_surface(
                meshcat,
                v,
                X,
                Y,
                Zc.reshape(X.shape),
                rgba=Rgba(1.0, 0.2, 0.2, 1.0),
                wireframe=True,
            )
        else:
            Zc = prog.EvalBindingVectorized(binding, values)
            evaluator = binding.evaluator()
            low = evaluator.lower_bound()
            up = evaluator.upper_bound()
            cvb = f"{cv}/{type(evaluator).__name__}"
            for index in range(Zc.shape[0]):
                # TODO(russt): Plot infeasible points in a different color.
                infeasible = np.logical_or(Zc[index, :] < low[index], Zc[index, :] > up[index])
                plot_surface(
                    meshcat,
                    f"{cvb}/{index}",
                    X,
                    Y,
                    Zc[index, :].reshape(X.shape),
                    rgba=Rgba(1.0, 0.3, 1.0, 1.0),
                    wireframe=True,
                )

    if costs:
        plot_surface(
            meshcat,
            f"{path}/objective",
            X,
            Y,
            Z.reshape(X.shape),
            rgba=Rgba(0.3, 1.0, 0.3, 1.0),
            wireframe=True,
        )

    if result:
        v = f"{path}/solution"
        meshcat.SetObject(v, Sphere(point_size), Rgba(0.3, 1.0, 0.3, 1.0))
        x_solution = result.get_x_val()
        meshcat.SetTransform(v, RigidTransform([x_solution[0], x_solution[1], result.get_optimal_cost()]))