import gym
import numpy as np

import warnings
from gym.utils import seeding
from pydrake.all import Simulator, SimulatorStatus, RandomGenerator
from pydrake.math import RollPitchYaw, RigidTransform

try:
    import open3d as o3d
except:
    pass

from pydrake.common import RandomGenerator
from env.env_util import *
import IPython
from core.utils import *
from colored import fg
import matplotlib.pyplot as plt
import time


class BaseDrakeRobotEnv(gym.Env):
    """
    Base Drake Environment Class for Manipulation
    """

    def __init__(self, task_config):
        self.reset_from_scene_file = False

    def cleanup_meshcat(self):
        """
        Cleans up the meshcat visualization by hiding collision geometry and deleting visualizer objects and tools.
        """
        # cleanup meshcat
        from env.env_util import meshcat

        if meshcat is not None:
            if not self.task_config.vis_collision_geometry:
                meshcat.SetProperty("collision", "visible", False)
            for index in range(self.plant.num_model_instances()):
                model_name = self.plant.GetModelInstanceName(ModelInstanceIndex(index))
                if "body_link" in model_name:  # cleanup objects and tools
                    meshcat.Delete("visualizer/" + model_name)
            time.sleep(0.1)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.drake_random = RandomGenerator(seed)
        return [seed]

    def advance_time(self, context):
        self.time = context.get_time()
        catch = False
        s = time.time()
        repeat_num = self.task_config.sim.substep_num

        try:
            for _ in range(repeat_num):  # make the simulation smooth and visulizable
                self.time = context.get_time()
                status = self.simulator.AdvanceTo(self.time + self.sim_dt)
                self._get_robot_freq_data()  # 0.02s slower per step

        except RuntimeError as e:
            warnings.warn("Calling Done after catching RuntimeError:")
            warnings.warn(e.args[0])
            catch = True
            self.info["sim_failure"] = True  # emergent reset
            color_print("simulation failed, reset!", "light_slate_blue")

        sim_done = catch or status.reason() == SimulatorStatus.ReturnReason.kReachedTerminationCondition
        return sim_done

    def get_ee_pos(self):
        """get end effector poses"""
        plant_context = self.system.GetSubsystemContext(self.plant, self.context)
        return self.plant.GetFrameByName("panda_hand").CalcPoseInWorld(plant_context)

    def _get_reward(self, context, action):
        return 0

    def _get_progress(self, context):
        # determine which stage is at for the expert
        return None

    def control_finger(self, state=None):
        """extra function to control the finger"""
        if state == "open":
            self.finger_state = [0, 0]
        if state == "close":
            self.finger_state = [0.04, 0.04]

    def get_joint_angles(self):
        """get the joint angle of the robot"""

        # q0 = self.joint_measurement_port.Eval(self.context)
        q0 = self.plant.GetPositions(self.plant_context, self.plant.GetModelInstanceByName("panda"))
        self.joint_positions = np.array(q0).copy()
        return self.joint_positions

    def get_actual_joint_infos(self):
        joint_pos = self.plant.GetPositions(self.plant_context, self.plant.GetModelInstanceByName("panda"))  #
        joint_vel = self.plant.GetVelocities(self.plant_context, self.plant.GetModelInstanceByName("panda"))
        return joint_pos, joint_vel

    def get_obs(self, render=True):
        """get the environment observation"""
        return self._get_obs(self.context, render=render)

    def check_collision_name(self, body_a_name, body_b_name):
        """only return collisions if two body parts collide or anything collide with anything but tool and object"""
        self_collision = "panda_link" in body_a_name and "panda_link" in body_b_name
        external_collision = ("model" not in body_a_name and "panda" not in body_a_name and "panda" in body_b_name) or (
            "model" not in body_b_name and "panda" not in body_b_name and "panda" in body_a_name
        )
        return self_collision or external_collision  # or tool_collision

    def export_context(self):
        context = self.simulator.get_context()
        return context.Clone()

    def export_all_context(self):
        context = self.simulator.get_context()
        self.controller_context = self.controller_plant_system.GetSubsystemContext(self.controller_plant, self.context)

        return context.Clone(), self.controller_context.Clone()

    def load_all_context(self, context, controller_context):
        self.context.SetTimeStateAndParametersFrom(context)
        self.controller_context.SetTimeStateAndParametersFrom(controller_context)

        self.simulator.Initialize()

    def load_context(self, context):
        self.context.SetTimeStateAndParametersFrom(context)
        self.simulator.Initialize()

    def seed(self, seed=None):
        """Implements gym.Env.seed using Drake's RandomGenerator."""
        if seed:
            self.generator = RandomGenerator(seed)
        else:
            seed = self.generator()

        return [seed]

    def set_plant_positions(self, joints, joint_vel=None):
        # sync all plants that have the robot states
        self.plant_context = self.system.GetSubsystemContext(self.plant, self.context)
        self.plant.SetPositions(self.plant_context, self.plant.GetModelInstanceByName("panda"), joints)  #
        if joint_vel is None:
            self.plant.SetVelocities(self.plant_context, self.plant.GetModelInstanceByName("panda"), [0] * 9)
        else:
            self.plant.SetVelocities(self.plant_context, self.plant.GetModelInstanceByName("panda"), joint_vel)

        ee_pose = self.set_diffik_state(self.context, reinit=True, set_joints=joints)
        controller_context = self.controller_plant.CreateDefaultContext()

        self.controller_plant.SetPositions(
            controller_context,
            self.controller_plant.GetModelInstanceByName("panda"),
            joints,
        )
        self.controller_plant.SetVelocities(
            controller_context,
            self.controller_plant.GetModelInstanceByName("panda"),
            [0] * 9,
        )
        self.joint_positions = joints
        self.diffik_port.FixValue(self.context, RigidTransform(ee_pose))
        if self.use_joint_controller():
            self.joint_action_port.FixValue(self.context, self.joint_positions)
        self.ee_pose = np.array(ee_pose.GetAsMatrix4()).copy().flatten()

    def reset_object(self, context, curr_object_id=-1):
        """sample a random object to drop on the table"""
        self.plant_context = self.system.GetSubsystemContext(self.plant, self.context)
        x_range = self.task_config.env.init_object_x_range
        y_range = self.task_config.env.init_object_y_range

        # RollPitchYaw(0, 0, np.random.uniform(-np.pi, np.pi)
        initial_rot = RollPitchYaw(0, 0, 0)  # np.random.uniform(-np.pi, np.pi)

        tf = RigidTransform(
            initial_rot,
            [
                np.random.uniform(x_range[0], x_range[1]),
                np.random.uniform(y_range[0], y_range[1]),
                self.table_init_height,
            ],
        )

        self.set_object_pose(tf)
        if self.task_config.randomize_mass:
            random_mass_value = np.random.uniform(0.1, 3)
            self.plant.GetBodyByName(self.object_name).SetMass(self.context, mass=1)

    def set_object_pose(self, tf):
        """setting object pose
        :param tf: Rigidtransform tf for the pose
        """
        all_object_bodies = self.plant.GetBodyIndices(self.object_asset)

        for idx, obj_id in enumerate(all_object_bodies):
            body = self.plant.get_body(obj_id)
            self.plant.SetFreeBodyPose(self.plant_context, body, tf)
            self.plant.SetFreeBodySpatialVelocity(body, SpatialVelocity(np.zeros((6, 1))), self.plant_context)

    def _get_object_pose(self, context=None):
        if context is None:
            context = self.context
        plant_context = self.system.GetSubsystemContext(self.plant, self.context)
        self.object_pose = self.plant.GetFrameByName(self.object_name).CalcPoseInWorld(plant_context).GetAsMatrix4()
        return self.object_pose

    def _get_tool_pose(self, context=None):
        if context is None:
            context = self.context
        plant_context = self.system.GetSubsystemContext(self.plant, self.context)
        self.tool_pose = self.plant.GetFrameByName(self.tool_name).CalcPoseInWorld(plant_context).GetAsMatrix4()
        return self.tool_pose

    def solve_ik(self, target_pose, seed, rot_tol=0.01, filter_manipulability=False, table_col=True):
        s = time.time()
        diagram_context = self.controller_plant_system.CreateDefaultContext()

        res = solve_ik(
            self.controller_plant,
            self.controller_plant.GetFrameByName("panda_hand"),
            RigidTransform(target_pose),
            seed.reshape(-1, 1),
            rot_tol=rot_tol,
            add_table_col=table_col,
            timeout=True,
            # add_gripper_faceup=False,
            filter_manipulability=filter_manipulability,
        )
        return res

    ### function for panda with tools
    def reset_robot(self, context, ik_consider_collision=False):
        """sample a random half sphere view and solve for inverse kinematics"""
        self.plant_context = self.system.GetSubsystemContext(self.plant, self.context)
        self.joint_positions = np.array(self.task_config.env.init_joints)
        self.plant.SetPositions(
            self.plant_context,
            self.plant.GetModelInstanceByName("panda"),
            self.joint_positions,
        )  #
        self.plant.SetVelocities(self.plant_context, self.plant.GetModelInstanceByName("panda"), [0] * 9)

        if not self.task_config.env.randomize_init_eff:
            return self.get_ee_pos()

        target_obj_pose = self.plant.GetFrameByName(self.object_name).CalcPoseInWorld(self.plant_context)
        target = target_obj_pose.translation()[:3]
        target = np.array(target)
        target[-1] = 0
        ik = None
        loop_num = 50
        initial_near = self.task_config.env.init_endeffector_range[0]
        initial_far = self.task_config.env.init_endeffector_range[1]
        inner_loop_num = 5
        res = None

        for i in range(loop_num):
            # the object is not on the table in the beginning
            target_pose = randomize_sphere_lookat(target, near=initial_near, far=initial_far, x_near=0.2, x_far=0.5)

            if target_pose[2, 3] < 0.3:
                print("end effector init too low. skip: ", target_pose[2, 3])
                continue

            idx = 0 if target_pose[1, 3] > target[1] else 1
            for j in range(inner_loop_num):

                seed = np.array(anchor_seeds[np.random.randint(len(anchor_seeds))])
                res = self.solve_ik(target_pose, seed.reshape(-1, 1), filter_manipulability=True)
                if res is not None:
                    res = res.get_x_val()[:9]
                    res[-2:] = 0.0
                    self.set_plant_positions(res[:9])
                    self.initial_ee_pose = self.get_ee_pos().GetAsMatrix4()
                    return self.initial_ee_pose

        color_print("robot end effector does not get to randomize", "light_slate_blue")
        res = self.task_config.env.init_joints
        self.set_plant_positions(res[:9])
        self.initial_ee_pose = self.get_ee_pos().GetAsMatrix4()
        return self.initial_ee_pose

    def setup(self, tool_obj_info, tool_info):
        self.object_asset, self.object_name, self.tool_obj_info, _ = tool_obj_info
        (
            self.tool_asset,
            self.tool_name,
            self.tool_info,
            _,
            self.tool_rel_pose,
        ) = tool_info
