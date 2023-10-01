# https://github.com/pangtao22/robotics_utilities/blob/master/iiwa_controller/robot_internal_controller.py

import numpy as np

from pydrake.systems.framework import LeafSystem, BasicVector, PortDataType
from pydrake.multibody.tree import MultibodyForces, JacobianWrtVariable
from core.utils import *


class RobotInternalController(LeafSystem):
    def __init__(
        self,
        plant_robot,
        joint_stiffness,
        damping_ratio=1,
        kd=None,
        controller_mode="inverse_dynamics",
        name="robot_internal_controller",
    ):
        """
        Impedance controller implements the controller in Ott2008 for IIWA.
        Inverse dynamics controller makes use of drake's
            InverseDynamicsController.
        :param plant_robot:
        :param joint_stiffness: (nq,) numpy array which defines the stiffness
            of all joints.
        """
        LeafSystem.__init__(self)
        self.set_name(name)
        self.plant = plant_robot
        self.context = plant_robot.CreateDefaultContext()

        self.nq = plant_robot.num_positions()
        self.nv = plant_robot.num_velocities()
        self.robot_state_input_port = self.DeclareInputPort(
            "robot_state", PortDataType.kVectorValued, self.nq + self.nv
        )
        self.tau_feedforward_input_port = self.DeclareInputPort("tau_feedforward", PortDataType.kVectorValued, self.nq)
        self.joint_angle_commanded_input_port = self.DeclareInputPort(
            "q_robot_commanded", PortDataType.kVectorValued, self.nq
        )
        self.joint_torque_output_port = self.DeclareVectorOutputPort(
            "joint_torques", BasicVector(self.nv), self.CalcJointTorques
        )

        # control rate
        self.t = 0
        self.control_period = 1e-3  # 1000Hz.
        self.DeclareDiscreteState(self.nv)
        self.DeclarePeriodicDiscreteUpdate(period_sec=self.control_period)

        # joint velocity estimator
        self.q_prev = None
        self.w_cutoff = 2 * np.pi * 4000
        self.velocity_estimator = LowPassFilter(self.nv, self.control_period, self.w_cutoff)
        self.acceleration_estimator = LowPassFilter(self.nv, self.control_period, self.w_cutoff)

        # damping coefficient filter
        self.Kv_filter = LowPassFilter(self.nq, self.control_period, 2 * np.pi)

        # controller gains
        self.Kp = joint_stiffness
        self.damping_ratio = damping_ratio
        self.Kv = 2 * self.damping_ratio * np.sqrt(self.Kp)
        if kd is not None:
            self.Kv = np.array(kd)
        self.controller_mode = controller_mode

        # logs
        self.Kv_log = []
        self.tau_stiffness_log = []
        self.tau_damping_log = []
        self.sample_times = []
        self.cartesian_impedance_values = np.array([20, 20, 20, 200, 200, 200])
        self.joint_impedance_values = np.array([120, 100, 100, 80, 20, 50, 20, 20])
        self.KI = np.ones(9)
        self.integral_errors = np.zeros(9)

        # https://github.com/rachelholladay/franka_ros_interface/blob/master/franka_ros_controllers/config/ros_controllers.yaml
        self.joint_min_limit = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0, 0])
        self.joint_max_limit = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
        self.joint_null_space = (self.joint_min_limit + self.joint_max_limit) / 2
        self.effort_limit = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])

    def get_input_port_estimated_state(self):
        return self.robot_state_input_port

    def get_input_port_desired_state(self):
        return self.joint_angle_commanded_input_port

    def get_output_port_control(self):
        return self.joint_torque_output_port

    def get_torque_feedforward_port_control(self):
        return self.tau_feedforward_input_port

    # adding hacks for cartesian force control
    def set_desired_cartesian_force(self):
        pass

    def set_cartesian_controller_parameters(self, set_point, stiffness=None):
        if stiffness is not None:
            self.Kp = stiffness
        self.Kv = 2 * self.damping_ratio * np.sqrt(self.Kp)
        self.cartesian_setpoint = set_point  # check frame

    def compute_cartesian_impedance_control(self, context, v, tau):
        # https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/ca7a4fb762f9581e39cc2aab644f18a83d6ab0ba/isaacgymenvs/tasks/factory/factory_control.py#L270
        # https://github.com/rachelholladay/franka_ros_interface/blob/master/franka_ros_controllers/src/cartesian_impedance_controller.cpp
        # could be tricky to choose the coordinate frame
        x = self.robot_state_input_port.Eval(context)
        q = x[: self.nq]
        v = x[self.nq :]

        self.joint_jacobian = self.plant.CalcJacobianSpatialVelocity(
            self.context,
            JacobianWrtVariable.kQDot,
            self.plant.GetBodyByName("panda_hand").body_frame(),
            [0, 0, 0],
            self.plant.world_frame(),
            self.plant.world_frame(),
        )[:, :7]

        # compute error
        self.curr_ee = np.array(self.plant.GetFrameByName("panda_hand").CalcPoseInWorld(self.context).GetAsMatrix4())
        self.cartesian_error = se3_inverse(self.curr_ee) @ self.cartesian_setpoint
        # print("self.cartesian_error:", self.cartesian_error)
        axangle_error = mat2axangle(self.cartesian_error[:3, :3])
        axangle_error = axangle_error[0] * axangle_error[1]
        cartesian_error = np.concatenate((axangle_error, self.cartesian_error[:3, 3]))
        Kp = self.cartesian_impedance_values
        tau_stiffness = -(Kp * cartesian_error)
        Kv = 2 * self.damping_ratio * np.sqrt(Kp)
        tau_damping = -Kv * (self.joint_jacobian @ v[:7])

        # also add null space
        tau_task = self.joint_jacobian.T @ (tau_stiffness + tau_damping)

        # tau_nullspace = ( np.eye(7) -  self.joint_jacobian.T * np.linalg.pinv(self.joint_jacobian.T)) *
        #                (self.Kp * (self.joint_null_space - x) -
        #                 (2.0 * sqrt(self.Kp)) * v);

        tau_task = np.concatenate((tau_task, np.zeros(2)))

        # jacobian inverse gives
        tau += tau_task  #
        self.tau_stiffness_log.append(tau_stiffness.copy())
        self.tau_damping_log.append(tau_damping.copy())
        return tau

    def switch_controller(self, controller_mode):
        self.controller_mode = controller_mode

    # TODO(liruiw) consider cartesian impedance control and joint velocity control
    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        LeafSystem.DoCalcDiscreteVariableUpdates(self, context, events, discrete_state)

        # read input ports
        x = self.robot_state_input_port.Eval(context)
        q_cmd = self.joint_angle_commanded_input_port.Eval(context)
        tau_ff = self.tau_feedforward_input_port.Eval(context)

        # TODO: could even implement a hybrid policy in here.
        q = x[: self.nq]
        v = x[self.nq :]

        # estimate velocity
        if self.q_prev is None:
            self.velocity_estimator.update(np.zeros(self.nv))
            self.acceleration_estimator.update(np.zeros(self.nv))
            v_diff = np.zeros(self.nv)
        else:
            # low pass filter velocity.
            v_diff = (q - self.q_prev) / self.control_period
            print(f"v_diff: {np.linalg.norm(v_diff):.3f}")
            self.velocity_estimator.update(v_diff)
            self.acceleration_estimator.update((v_diff - self.v_prev) / self.control_period)

        self.q_prev = q
        self.v_prev = v_diff
        v_est = self.velocity_estimator.get_current_state()
        a_est = self.acceleration_estimator.get_current_state()

        # log the P and D parts of desired acceleration
        self.sample_times.append(context.get_time())

        # update plant context
        self.plant.SetPositions(self.context, q)  # why is this called ?
        # self.plant.SetVelocities(self.context, v_est)

        # gravity compenstation
        tau_g = self.plant.CalcGravityGeneralizedForces(self.context)  # compensate for gravity
        tau = -tau_g

        # TODO(liruiw) think about controller switch in the context of drake simulation
        if self.controller_mode == "impedance":
            M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)

            # m = np.sort(np.linalg.eig(M)[0])[::-1]
            m = M.diagonal()
            Kv = 2 * self.damping_ratio * np.sqrt(self.Kp * m)
            self.Kv_filter.update(Kv)
            Kv = self.Kv_filter.get_current_state()
            tau_stiffness = self.Kp * (q_cmd - q)
            tau_damping = -Kv * v
            tau += tau_damping + tau_stiffness

            self.Kv_log.append(Kv)
            self.tau_stiffness_log.append(tau_stiffness.copy())
            self.tau_damping_log.append(tau_damping.copy())

        elif self.controller_mode == "inverse_dynamics":
            qDDt_d = self.Kp * (q_cmd - q) + self.Kv * (-v_est)
            # self.integral_errors  += (q_cmd - q) * self.control_period
            # qDDt_d = qDDt_d + self.integral_errors * self.KI
            # print("ki term:", (self.integral_errors * self.KI).sum())
            tau += self.plant.CalcInverseDynamics(
                context=self.context,
                known_vdot=qDDt_d,
                external_forces=MultibodyForces(self.plant),
            )

        elif self.controller_mode == "cartesian_impedance":
            # self.plant_controller.erset_cartesian_controller_parameters()
            qDDt_d = self.Kp * (q_cmd - q) + self.Kv * (-v_est)
            tau_inv_dyn = self.plant.CalcInverseDynamics(
                context=self.context,
                known_vdot=qDDt_d,
                external_forces=MultibodyForces(self.plant),
            )
            tau_old = tau.copy()
            tau = self.compute_cartesian_impedance_control(context, v, tau)
            # https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_multibody_plant.html#afcb4de9b58cb375d056da7e8ec30937a
            C = self.plant.CalcBiasTerm(self.context)
            M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)
            tau += C @ v + M @ a_est
            tau -= MultibodyForces(self.plant).generalized_forces()
            # if  np.linalg.norm(v) > 2:
            #     print("inverse dynamics control:", tau_inv_dyn)
            #     print("cartesian_impedance control:", tau - tau_inv_dyn)  #100x greater
            #     print("tau update:", tau - self.prev_tau)  #100x greater

        output = discrete_state.get_mutable_vector().get_mutable_value()
        output[:] = tau + tau_ff
        self.t += self.control_period  # debug
        self.prev_tau = tau
        output[:] = np.clip(output, -self.effort_limit, self.effort_limit)
        print(f"v: {np.linalg.norm(v):.3f}")

    def CalcJointTorques(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        y[:] = state


class LowPassFilter:
    def __init__(self, dimension: int, h: float, w_cutoff: float):
        if w_cutoff == np.inf:
            self.a = 1.0
        else:
            self.a = h * w_cutoff / (1 + h * w_cutoff)

        self.n = dimension
        self.x = None

    def update(self, u: np.array):
        assert u.size == self.n

        if self.x is None:
            self.x = u
        else:
            self.x = (1 - self.a) * self.x + self.a * u

    def reset_state(self):
        self.x = None

    def has_valid_state(self):
        return self.x is not None

    def get_current_state(self):
        assert self.x is not None
        return self.x.copy()
