import numpy as np
import time
import numpy as np
from numpy.linalg import inv

from pydrake.common.cpp_param import List
from pydrake.common.value import Value
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialForce, SpatialVelocity
from pydrake.multibody.plant import (
    ExternallyAppliedSpatialForce,
    MultibodyPlant,
)
from pydrake.multibody.tree import (
    Frame,
    JacobianWrtVariable,
    ModelInstanceIndex,
    SpatialInertia,
)
from pydrake.systems.framework import (
    BasicVector,
    DiagramBuilder,
    FixedInputPortValue,
    InputPort,
    LeafSystem,
    OutputPort,
)
from pydrake.systems.primitives import FirstOrderLowPassFilter
from core.utils.utils import *
import dataclasses as dc


def rotation_matrix_to_axang3(rot_mat):
    axangle = mat2axangle(rot_mat.matrix()[:3, :3])
    return axangle[0] * axangle[1]


@dc.dataclass
class PoseFrames:
    frame_T: Frame
    frame_P: Frame

    def __eq__(self, rhs):
        return self.frame_T is rhs.frame_T and self.frame_P is rhs.frame_P

    def AssertValid(self):
        assert self.frame_T is not None
        assert self.frame_P is not None


@dc.dataclass
class PoseFeedback:
    kp_xyz: np.ndarray
    kd_xyz: np.ndarray
    kp_rot: np.ndarray
    kd_rot: np.ndarray

    def __call__(self, X_WP, V_WP, X_WPdes, V_WPdes):
        # Transform to "negative error": desired w.r.t. actual,
        # expressed in world frame (for applying the force).
        # TODO(eric.cousineau): Use ComputePoseDiffInCommonFrame.
        p_PPdes_W = X_WPdes.translation() - X_WP.translation()
        R_WP = X_WP.rotation()
        R_PPdes = R_WP.inverse() @ X_WPdes.rotation()
        axang3_PPdes = rotation_matrix_to_axang3(R_PPdes)
        axang3_PPdes_W = R_WP @ axang3_PPdes
        V_PPdes_W = V_WPdes - V_WP
        v_PPdes_W = V_PPdes_W.translational()
        w_PPdes_W = V_PPdes_W.rotational()
        # Compute wrench components.

        f_P_W = self.kp_xyz * p_PPdes_W + self.kd_xyz * v_PPdes_W
        tau_P_W = self.kp_rot * axang3_PPdes_W + self.kd_rot * w_PPdes_W
        F_P_W_feedback = SpatialForce(tau=tau_P_W, f=f_P_W)
        return F_P_W_feedback


@dc.dataclass
class PoseReferenceInputPorts:
    frames: PoseFrames
    X_TPdes: InputPort
    V_TPdes: InputPort

    def fix(self, plant, context, plant_context=None):
        if plant_context is None:
            plant_context = plant.GetMyContextFromRoot(context)
        X_TP_init = plant.CalcRelativeTransform(
            plant_context,
            self.frames.frame_T,
            self.frames.frame_P,
        )
        V_TP_init = SpatialVelocity.Zero()
        return PosePortValues(
            X_TPdes=fix_port(self.X_TPdes, context, X_TP_init),
            V_TPdes=fix_port(self.V_TPdes, context, V_TP_init),
        )


@dc.dataclass
class PosePortValues:
    X_TPdes: FixedInputPortValue
    V_TPdes: FixedInputPortValue

    def update(self, X_TPdes, V_TPdes=None):
        if V_TPdes is None:
            V_TPdes = SpatialVelocity.Zero()
        self.X_TPdes.GetMutableData().set_value(X_TPdes)
        self.V_TPdes.GetMutableData().set_value(V_TPdes)


@dc.dataclass
class JointFeeback:
    Kp: np.ndarray
    Kd: np.ndarray

    def __call__(self, q, v, q_des, v_des):
        return self.Kp * (q_des - q) + self.Kd * (v_des - v)


@dc.dataclass
class DofSetForDynamics:
    """
    Like DofSet, but for dynamics where we are not constrained to nq = nv = nu.
    """

    q: np.ndarray  # bool
    v: np.ndarray  # bool
    u: np.ndarray  # bool


def get_frame_spatial_velocity(plant, context, frame_T, frame_F, frame_E=None):
    """
    Returns:
        SpatialVelocity of frame F's origin w.r.t. frame T, expressed in E
        (which is frame T if unspecified).
    """
    if frame_E is None:
        frame_E = frame_T
    Jv_TF_E = plant.CalcJacobianSpatialVelocity(
        context,
        with_respect_to=JacobianWrtVariable.kV,
        frame_B=frame_F,
        p_BoBp_B=[0, 0, 0],
        frame_A=frame_T,
        frame_E=frame_E,
    )
    v = plant.GetVelocities(context)
    V_TF_E = SpatialVelocity(Jv_TF_E @ v)
    return V_TF_E


def make_empty_dofset(plant):
    return DofSetForDynamics(
        q=np.zeros(plant.num_positions(), dtype=bool),
        v=np.zeros(plant.num_velocities(), dtype=bool),
        u=np.zeros(plant.num_actuated_dofs(), dtype=bool),
    )


def articulated_model_instance_dofset(plant, instance):
    """
    Returns DofSetForDynamics given a particular model instance.
    """
    dofs = make_empty_dofset(plant)
    joint_indices = plant.GetJointIndices(instance)
    for joint_index in joint_indices:
        joint = plant.get_joint(joint_index)
        start_in_q = joint.position_start()
        end_in_q = start_in_q + joint.num_positions()
        dofs.q[start_in_q:end_in_q] = True
        start_in_v = joint.velocity_start()
        end_in_v = start_in_v + joint.num_velocities()
        dofs.v[start_in_v:end_in_v] = True

    B_full = plant.MakeActuationMatrix()
    B = B_full[dofs.v]
    assert set(B.flat) <= {0.0, 1.0}
    B_rows = B.sum(axis=0)
    assert set(B_rows.flat) <= {0.0, 1.0}
    dofs.u[:] = B_rows.astype(bool)

    nq = plant.num_positions(instance)
    nv = plant.num_velocities(instance)
    nu = plant.num_actuated_dofs(instance)
    assert dofs.q.sum() == nq
    assert dofs.v.sum() == nv
    assert dofs.u.sum() == nu
    return dofs


class PoseController(LeafSystem):
    """
    Leveraging notation in line with Drake and Russ's Manipulation course
    notes:
    https://manipulation.csail.mit.edu/force.html#section3

    For additional operationaln space refs, see:

    - A. D. Ames and M. Powell, “Towards the Unification of Locomotion and
      Manipulation through Control Lyapunov Functions and Quadratic Programs,”
      in Control of Cyber-Physical Systems, 2013.
      http://link.springer.com/10.1007/978-3-319-01159-2_12
      - Easier to digest this notation.

    - O. Khatib, L. Sentis, J. Park, and J. Warren, “Whole-Body Dynamic
      Behavior and Control of Human-Like Robots,” Int. J. Human. Robot. 2004.
      https://www.worldscientific.com/doi/abs/10.1142/S0219843604000058
      https://khatib.stanford.edu/publications/pdfs/Khatib_2004_IJHR.pdf
      - Good for main reference of above paper.

    - O. Khatib, “A unified approach for motion and force control of robot
      manipulators: The operational space formulation,” IEEE Journal on
      Robotics and Automation, 1987.
      https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

    - Robotics Handbook, p.236, Sec. 10.6, Second-Order Redundancy Resolution.

    Derivation:
    https://toyotaresearchinstitute.sharepoint.com/:o:/r/sites/ToyotaResearchInstitute/Shared%20Documents/Dexterous%20Manipulation/Issues/Anzu%20Issue%208783%20-%20Pose%20Tracking/2022-09-14%20Pose%20Tracking?d=w94c38e6f0ac547a78e188b2d23a734b5&csf=1&web=1&e=HkxogD
    """

    def __init__(
        self,
        plant,
        model_instance,
        frame_T,
        frame_P,
    ):
        """frame_T is the task frame (inertial frame) and frame_P is the end effector frame"""
        super().__init__()

        context = plant.CreateDefaultContext()
        ndof = plant.num_positions(model_instance)
        nv = plant.num_velocities(model_instance)
        assert ndof == nv
        nx = 2 * ndof
        nu = plant.num_actuated_dofs(model_instance)
        assert ndof == nu
        dofs = articulated_model_instance_dofset(plant, model_instance)
        assert np.sum(dofs.q) == ndof
        assert np.sum(dofs.v) == ndof
        FIX_FINGER = True  # always command certain torques to the fingers

        kpt_xyz = 400  # 100.0
        kdt_xyz = 40  # 20.0
        kpt_rot = 400  # 100.0
        kdt_rot = 40  # 20.0 # 400, 40, 400, 40
        self.pose_feedback = PoseFeedback(
            kp_xyz=kpt_xyz,
            kd_xyz=kdt_xyz,  # lambda R_TP:
            kp_rot=kpt_rot,
            kd_rot=kdt_rot,  # lambda R_TP:
        )

        # Very stiff.
        kpp = 200.0  # 100.0
        kdp = 20.0
        # # Less stiff.
        # kpp = 50.0
        # kdp = 12.0
        # # Wiggly.
        # kpp = 5.0
        # kdp = 1.0
        self.scaled_joint_feedback = JointFeeback(
            Kp=kpp,
            Kd=kdp,
        )

        self.direct_joint_feedback = JointFeeback(
            Kp=np.diag([160.0, 200.0, 200.0, 160.0, 25.0, 70.0, 16.0]) * 0.4,
            Kd=np.diag([37.5, 50.0, 37.5, 25.0, 5.0, 3.75, 2.5]) * 0.6,
        )

        # Should use initialization event.
        q0 = plant.GetPositions(context, model_instance)
        v0 = np.zeros(nv)

        q_lower = plant.GetPositionLowerLimits()[dofs.q]
        q_upper = plant.GetPositionUpperLimits()[dofs.q]
        v_lower = plant.GetVelocityLowerLimits()[dofs.v]
        v_upper = plant.GetVelocityUpperLimits()[dofs.v]
        u_lower = plant.GetEffortLowerLimits()[dofs.u]
        u_upper = plant.GetEffortUpperLimits()[dofs.u]

        def assert_within_limits(value, *, lower, upper):
            too_low = value < lower
            too_high = value > upper
            assert not (too_low | too_high).any()

        def assert_values_within_limits(q, v, u_des):
            assert_within_limits(q, lower=q_lower, upper=q_upper)
            assert_within_limits(v, lower=v_lower, upper=v_upper)
            assert_within_limits(u_des, lower=u_lower, upper=u_upper)

        def control_math(X_TPdes, V_TPdes):
            q = plant.GetPositions(context, model_instance)
            v = plant.GetVelocities(context, model_instance)
            X_TP = plant.CalcRelativeTransform(context, frame_T, frame_P)
            V_TP = get_frame_spatial_velocity(plant, context, frame_T, frame_P)

            Jv_TP_full = plant.CalcJacobianSpatialVelocity(
                context,
                with_respect_to=JacobianWrtVariable.kV,
                frame_B=frame_P,
                p_BoBp_B=[0, 0, 0],
                frame_A=frame_T,
                frame_E=frame_T,
            )  # J
            Jv_TP = Jv_TP_full[:, dofs.v]
            Jvdot_v_TP = plant.CalcBiasSpatialAcceleration(
                context,
                with_respect_to=JacobianWrtVariable.kV,
                frame_B=frame_P,
                p_BoBp_B=[0, 0, 0],
                frame_A=frame_T,
                frame_E=frame_T,
            )  #

            # Full-plant dynamics.
            M_full = plant.CalcMassMatrix(context)
            C_full = plant.CalcBiasTerm(context)
            tau_g_full = plant.CalcGravityGeneralizedForces(context)

            # Full-order dynamics, but selecting controlled DoFs.
            M = M_full[dofs.v, :][:, dofs.v]
            C = C_full[dofs.v]
            tau_g = tau_g_full[dofs.v]

            # Task-space dynamics.
            Jt = Jv_TP
            Jtdot_v = Jvdot_v_TP.get_coeffs()  # J_dot q_dot
            # Purely kinematic.
            Nt_T = np.eye(ndof) - np.linalg.pinv(Jt) @ Jt  # null space

            # For singularities:
            # WARNING: Jv_TP dropping rank (beyond redundancy) will cause this
            # stuff to explode. Should look at nasa jsc code or Khatib '87 for
            # pointers on how to more intelligently avoid, maybe?
            # Below can fudge things, but may induce a loss of tracking.
            nt = 6
            t_fudge = 0.01 * np.eye(nt)

            # TODO(eric.cousineau): Rewrite w/ more friendly inversions.
            inv = np.linalg.inv
            Mtinv = Jt @ inv(M) @ Jt.T  # eq 18
            Mt = inv(Mtinv + t_fudge)  # regularization
            Jtbar = inv(M) @ Jt.T @ Mt
            Nt_T = np.eye(ndof) - Jt.T @ Jtbar.T
            Ct = Jtbar.T @ C - Mt @ Jtdot_v  # eq 24
            tau_gt = Jtbar.T @ tau_g  # virtual work

            # N.B. This do not currently add M*vdot for task or joint
            # centering.
            # TODO(eric.cousineau): Add in inertial term when computing finite
            # diff of V_TPdes.

            # # Task-space control.
            Ft_feedback = (  # desired behavior when being exerted force
                Mt @ self.pose_feedback(X_TP, V_TP, X_TPdes, V_TPdes).get_coeffs()
            )
            Ft_feedforward = Ct - tau_gt  # cancel out dynamics
            # - Use task-space inverse dynamics.
            # N.B. Simulation seems to go unstable if not using task-space
            # inertial scaling.
            Ft = Ft_feedback + Ft_feedforward

            # Posture regularization.
            # N.B. Because Jp = I, we don't need any projected dynamics.
            Jp = np.eye(ndof)
            Mp = M
            Cp = C
            tau_gp = tau_g

            # - Use task-space inverse dynamics.
            Fp_feedback = Mp @ self.scaled_joint_feedback(q, v, q0, v0)
            Fp_feedforward = Cp - tau_gp
            Fp = Fp_feedback + Fp_feedforward

            # Combined joint-space control.
            ut = Jt.T @ Ft
            up = Nt_T @ Jp.T @ Fp  # null space part in the joint space eq 55
            # up = Jp.T @ Fp  # joint-space alone.

            u = ut + up
            # https://github.com/rachelholladay/franka_ros_interface/blob/fdbc28bf78e1f7eaa1ff627e1a34be72fafbdedc/franka_ros_controllers/src/cartesian_impedance_controller.cpp#L192

            # # Inverse Dynamics. (Feedback linearization?)
            # vd = scaled_joint_feedback(q, v, q0, v0)
            # u = M @ vd + C - tau_g

            # # Very simple joint feedback.
            # u = direct_joint_feedback(q, v, q0, v0) - tau_g

            # TODO(eric.cousineau): Implement as QP for joint position,
            # velocity, accel, and torque limits. May need CBF?
            # assert_values_within_limits(q, v, u)

            if FIX_FINGER:
                # print("finger torque:", u[-2:])
                u[-2:] = -100
            return u

        self.plant_state_input = self.DeclareVectorInputPort("plant_state", nx)
        X_TPdes_input = self.DeclareAbstractInputPort("X_TPdes", Value[RigidTransform]())
        V_TPdes_input = self.DeclareAbstractInputPort("V_TPdes", Value[SpatialVelocity]())
        self.pose_inputs = PoseReferenceInputPorts(
            frames=PoseFrames(frame_T=frame_T, frame_P=frame_P),
            X_TPdes=X_TPdes_input,
            V_TPdes=V_TPdes_input,
        )

        def control_calc(sys_context, output):
            x = self.plant_state_input.Eval(sys_context)
            plant.SetPositionsAndVelocities(context, model_instance, x)
            X_TPdes = self.pose_inputs.X_TPdes.Eval(sys_context)
            V_TPdes = self.pose_inputs.V_TPdes.Eval(sys_context)
            u = control_math(X_TPdes, V_TPdes)
            output.set_value(u)

        self.torques_output = self.DeclareVectorOutputPort("torques_output", size=nu, calc=control_calc)

        self._plant = plant
        self._model_instance = model_instance
        self.V_TPdes_input = V_TPdes_input
        self.X_TPdes_input = X_TPdes_input

    def set_joint_stiffness(self, mode="stiff"):
        if mode == "very_stiff":
            kpp = 400.0
            kdp = 40.0

        if mode == "stiff":
            kpp = 100.0
            kdp = 20.0
        # # Less stiff.

        if mode == "normal":
            kpp = 50.0
            kdp = 12.0

        # # Wiggly.
        if mode == "compliant":
            kpp = 5.0
            kdp = 1.0

        self.scaled_joint_feedback = JointFeeback(Kp=kpp, Kd=kdp)

    def makeJointFeeback(self, kpp, kdp):
        self.scaled_joint_feedback = JointFeeback(Kp=kpp, Kd=kdp)

    def set_pose_stiffness(self, parameters=[100, 20, 100, 20]):
        (kpt_xyz, kdt_xyz, kpt_rot, kdt_rot) = parameters
        self.pose_feedback = PoseFeedback(
            kp_xyz=kpt_xyz,
            kd_xyz=kdt_xyz,  # lambda R_TP:
            kp_rot=kpt_rot,
            kd_rot=kdt_rot,  # lambda R_TP:
        )

    def get_plant_state_input_port(self):
        return self.plant_state_input

    def get_velocity_desired_port(self):
        return self.V_TPdes_input

    def get_state_desired_port(self):
        return self.X_TPdes_input

    @staticmethod
    def AddToBuilder(builder, controller):
        plant = controller._plant
        model_instance = controller._model_instance

        builder.AddSystem(controller)
        builder.Connect(
            plant.get_state_output_port(model_instance),
            controller.plant_state_input,
        )
        builder.Connect(
            controller.torques_output,
            plant.get_actuation_input_port(model_instance),
        )
