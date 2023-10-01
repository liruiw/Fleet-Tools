import numpy as np

from pydrake.math import RollPitchYaw, RigidTransform
from env.env_util import *
from .frankadrakeenv import FrankaDrakeEnv
import IPython
from core.utils import *


class FrankaDrakeWrenchEnv(FrankaDrakeEnv):
    def __init__(self, task_config):
        self.tool_class_name = task_config.tool_class_name
        super().__init__(task_config)

    def reset(self, done=False):
        self.init_context()
        if self.use_pose_controller() or self.task_config.use_controller_portswitch:
            self.robot_controller.set_joint_stiffness("compliant")
            self.robot_controller.set_pose_stiffness([400, 40, 100, 20])

        # smooth out contact force scaling a bit
        self.history_wrench_force = deque([0] * 10, maxlen=10)
        for _ in range(self.task_config.sim.warmstart_iter):
            sim_done = self.advance_time(self.context)  #
            if sim_done:  # terminate because of simulation error
                return [], 0, True, self.info

        self.action = np.zeros(self.action_num)
        self._get_obs(self.context)
        obs = self.select_obs()
        reward = self._get_reward(self.context, self.action)

        from env.env_object_related import screw_joint_

        screw_joint_.set_translation(self.plant_context, 0.00)
        screw_joint_.set_angular_velocity(self.plant_context, 0)
        screw_joint_.set_translational_velocity(self.plant_context, 0)

        return obs, reward, done, self.info

    def render_two_views(self):
        pose = OVERHEAD_CAM_EXTR.copy()
        pose[:3, 3] = [0.5, 0.0, 0.25]
        base_cam_img = self._render("panda_link0_cam")
        wrist_cam_img = self.render_any_view(pose)  # not much need to vis
        return base_cam_img, wrist_cam_img

    def _get_object_pose(self, context=None):
        if context is None:
            context = self.context
        plant_context = self.system.GetSubsystemContext(self.plant, self.context)
        # the nut is the actual object pose
        self.object_pose = (
            self.plant.GetFrameByName(self.object_name.replace("_body", "_nut_body"))
            .CalcPoseInWorld(plant_context)
            .GetAsMatrix4()
        )
        return self.object_pose

    def step(self, action):

        self.time_step += 1
        cur_pose = self.get_ee_pos().GetAsMatrix4()
        sim_done = self.advance_time(self.context)
        if sim_done:  # terminate because of simulation error
            return [], 0, True, self.info
        self._get_obs(self.context)

        obs = self.select_obs()
        reward = self._get_reward(self.context, action)
        done = sim_done or (reward > 0) or self.need_termination

        return obs, reward, done, self.info

    def _get_obs(self, context, render=True):
        super()._get_obs(context, render=render)
        from env.env_object_related import screw_joint_, extra_force_system
        from env.env_util import meshcat

        contact_results = self.contact_results
        extra_force_system.wrench = np.zeros(6)
        self.nut_contacts = []  # might be used later

        if self.task_config.sim.compliant_hydroelastic_model:
            if contact_results.num_hydroelastic_contacts() > 0:
                # scale the force
                for i in range(contact_results.num_hydroelastic_contacts()):
                    contact_info_i = contact_results.hydroelastic_contact_info(i)
                    contact_surface = contact_info_i.contact_surface()
                    body_ia_index = contact_surface.id_M()
                    body_ib_index = contact_surface.id_N()  #
                    body_a_name = self.scene_inspector.GetName(body_ia_index)
                    body_b_name = self.scene_inspector.GetName(body_ib_index)

                    if "nut_body_link" in body_a_name or "nut_body_link" in body_b_name:
                        contact_spatial_force = contact_info_i.F_Ac_W()  # on body B expressed in world, contact_force
                        contact_force = contact_spatial_force.translational()
                        contact_torque = contact_spatial_force.rotational()

                        self.nut_contacts.append(contact_info_i)
                        force_norm = min(np.linalg.norm(contact_force[:2]), 0.1)
                        self.history_wrench_force.append(force_norm)
                        force_norm = np.mean(self.history_wrench_force)
                        torque = np.array([0, 0, force_norm])

        else:
            if contact_results.num_point_pair_contacts() > 1:
                # scale the force
                for i in range(contact_results.num_point_pair_contacts()):
                    contact_info_i = contact_results.point_pair_contact_info(i)
                    body_ia_index = contact_info_i.bodyA_index()
                    body_ib_index = contact_info_i.bodyB_index()  #
                    if (
                        "nut_body_link" in self.plant.get_body(body_ib_index).name()
                        or "nut_body_link" in self.plant.get_body(body_ia_index).name()
                    ):
                        contact_force = contact_info_i.contact_force()  # on body B expressed in world, contact_force
                        self.nut_contacts.append(contact_info_i)
                        force_norm = min(np.linalg.norm(contact_force[:2]), 0.1)
                        self.history_wrench_force.append(force_norm)
                        force_norm = np.mean(self.history_wrench_force)
                        torque = np.array([0, 0, force_norm])

    def set_object_pose(self, tf):
        return

    def _get_reward(self, context, action):
        """compute reward based on how much the screw is turned"""
        from env.env_object_related import screw_joint_, extra_force_system

        rotation = screw_joint_.get_rotation(self.plant_context)
        translation = screw_joint_.get_translation(self.plant_context)
        rew = float(rotation < -0.4)
        return rew  #  rew

    def _get_progress(self, context):
        pass
