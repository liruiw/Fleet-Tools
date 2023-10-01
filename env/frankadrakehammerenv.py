import gym
import numpy as np
from env.env_util import *
from .frankadrakeenv import FrankaDrakeEnv
import IPython
from core.utils import *


class FrankaDrakeHammerEnv(FrankaDrakeEnv):
    def __init__(self, task_config):
        self.tool_class_name = task_config.tool_class_name
        super().__init__(task_config)

    def reset(self, done=False):
        self.init_context()
        from env.env_object_related import extra_force_system

        # extra force to make sure the pin does not go down due to gravity
        g = -self.plant.gravity_field().gravity_vector()[2]
        extra_force_system.wrench = np.concatenate(
            (
                [
                    0,
                    0,
                    self.plant.GetBodyByName(self.object_name).get_mass(self.context) * g - 1e-10,
                ],
                [0, 0, 0],
            )
        )

        for _ in range(self.task_config.sim.warmstart_iter):
            sim_done = self.advance_time(self.context)  #
            if sim_done:  # terminate because of simulation error
                return [], 0, True, self.info

        self.action = np.zeros(self.action_num)
        self._get_obs(self.context)
        obs = self.select_obs()
        reward = self._get_reward(self.context, self.action)
        return obs, reward, done, self.info

    def step(self, action):
        self.time_step += 1
        sim_done = self.advance_time(self.context)

        if sim_done:  # terminate because of simulation error
            return [], 0, True, self.info
        self._get_obs(self.context)

        obs = self.select_obs()
        reward = self._get_reward(self.context, action)
        done = sim_done or (reward > 0) or self.need_termination

        return obs, reward, done, self.info

    def check_collision_name(self, body_a_name, body_b_name):
        collision = super().check_collision_name(body_a_name, body_b_name)
        # anything hitting the table or the box would mean failure.
        table_collision = "link" == body_a_name or "link" == body_b_name
        return collision or table_collision

    def _get_obs(self, context, render=True):
        from env.env_object_related import extra_force_system

        super()._get_obs(context, render=render)
        contact_results = self.contact_measurement_port.Eval(self.context)
        g = -self.plant.gravity_field().gravity_vector()[2]
        z_force = self.plant.GetBodyByName(self.object_name).get_mass(self.context) * g - 1e-10
        if contact_results.num_point_pair_contacts() == 1:  # manually set the downward force
            extra_force_system.wrench = np.concatenate(([0, 0, z_force], [0, 0, 0]))
        else:
            extra_force_system.wrench = np.concatenate(([0, 0, z_force], [0, 0, 0]))

    def _get_reward(self, context, action):
        """hammer environment reward is based on how much the pin is pressed"""
        from env.env_object_related import bolt_joint

        hammer_state = bolt_joint.get_translation(self.plant_context)
        rew = float(hammer_state > -0.015 and self.time > 2)
        return rew

    def set_object_pose(self, tf):
        return

    def _get_progress(self, context):
        pass
