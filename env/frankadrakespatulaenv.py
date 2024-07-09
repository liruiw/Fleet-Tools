import numpy as np
from env.env_util import *
from .frankadrakeenv import FrankaDrakeEnv
from core.utils import *


class FrankaDrakeSpatulaEnv(FrankaDrakeEnv):
    def __init__(self, task_config):
        self.tool_class_name = task_config.tool_class_name
        super().__init__(task_config)

    def reset(self, done=False):
        self.init_context()
        if self.use_pose_controller() or self.task_config.use_controller_portswitch:
            self.robot_controller.set_joint_stiffness("normal")

        for _ in range(self.task_config.sim.warmstart_iter):
            sim_done = self.advance_time(self.context)  #
            if sim_done:  # terminate because of simulation error
                return [], 0, True, self.info

        self.action = np.zeros(self.action_num)
        self._get_obs(self.context)
        obs = self.select_obs()
        reward = self._get_reward(self.context, self.action)
        self.reset_env_time = self.time
        return obs, reward, done, self.info

    def transform_keypoints_env(self):
        self.curr_tool_keypoint_head = self.tool_pose[:3, :3].dot(self.tool_keypoint_head) + self.tool_pose[:3, 3]
        self.curr_tool_keypoint_tail = self.tool_pose[:3, :3].dot(self.tool_keypoint_tail) + self.tool_pose[:3, 3]
        self.curr_tool_keypoint_side = self.tool_pose[:3, :3].dot(self.tool_keypoint_side) + self.tool_pose[:3, 3]

        # transform the object keypoint to be the bottom point of the object
        self.curr_tool_obj_keypoints = self.object_pose[:3, 3]
        self.curr_tool_obj_keypoints[2] = self.table_height + 0.005  #

    def step(self, action):
        self.time_step += 1
        sim_done = self.advance_time(self.context)

        if sim_done:  # terminate because of simulation error
            return [], 0, True, self.info
        self._get_obs(self.context)

        obs = self.select_obs()
        reward = self._get_reward(self.context, action)
        done = sim_done or (reward > 0) or self.need_termination
        self.terminate_mechanism()
        return obs, reward, done, self.info

    def _get_reward(self, context, action):
        """spatula environment reward is based on if the object is lifted with the spatula"""
        obj_pose = self._get_object_pose(context)
        obj_rot = mat2euler(obj_pose[:3, :3])
        reward = float(obj_pose[2, 3] > self.table_height + 0.05)
        return reward

    def _get_progress(self, context):
        pass
