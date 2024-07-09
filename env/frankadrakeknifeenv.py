import numpy as np
from pydrake.math import RollPitchYaw, RigidTransform
from env.env_util import *
from .frankadrakeenv import FrankaDrakeEnv
from core.utils import *


class FrankaDrakeKnifeEnv(FrankaDrakeEnv):
    def __init__(self, task_config):
        self.tool_class_name = task_config.tool_class_name
        super().__init__(task_config)
        self.init_dist = 0

    def get_combined_mask(self, cam):
        """the policy needs to see both objects in this task"""
        render_port = self.system.GetOutputPort(cam + "_label_image")
        maskdata = render_port.Eval(self.context).data.copy()

        # simulate whatever is above the table and not the robot
        object_bodies = self.plant.GetBodyIndices(self.object_asset)
        object_bodies += [self.plant.GetBodyByName(self.object_name + "_2").index()]
        tool_bodies = self.plant.GetBodyIndices(self.tool_asset)
        obj_masked_ids = [int(x) for x in object_bodies]
        tool_masked_ids = [int(x) for x in tool_bodies]
        obj_mask = sum([maskdata == x for x in obj_masked_ids]) > 0
        tool_mask = sum([maskdata == x for x in tool_masked_ids]) > 0
        filtered_maskdata = obj_mask | tool_mask

        # as auxiliary info
        full_mask = np.zeros(obj_mask.shape)
        full_mask[obj_mask] = 1.0
        full_mask[tool_mask] = 2.0
        self.full_mask_data.append(full_mask)

        return filtered_maskdata

    def reset(self, done=False):
        self.init_context()
        # reset the other object's pose based on the first object
        body = self.plant.GetBodyByName(self.object_name + "_2")
        tf = self.plant.GetFrameByName(self.object_name).CalcPoseInWorld(self.plant_context)

        # sample a random delta pose
        x_delta = np.random.choice([-1, 1]) * np.random.uniform(0.04, 0.07)
        y_delta = np.random.choice([-1, 1]) * np.random.uniform(0.04, 0.07)
        delta_translation = np.array([x_delta, y_delta, 0.1]) + tf.translation()
        tf = RigidTransform(RollPitchYaw(np.random.uniform(-np.pi, np.pi), 0, 0), delta_translation)
        self.plant.SetFreeBodyPose(self.plant_context, body, tf)  # overlap and then split

        for _ in range(self.task_config.sim.warmstart_iter):
            sim_done = self.advance_time(self.context)  #
            if sim_done:  # terminate because of simulation error
                return [], 0, True, {}

        self.action = np.zeros(self.action_num)
        self.obj_pose1 = self.plant.GetFrameByName(self.object_name).CalcPoseInWorld(self.plant_context).GetAsMatrix4()
        self.obj_pose2 = (
            self.plant.GetFrameByName(self.object_name + "_2").CalcPoseInWorld(self.plant_context).GetAsMatrix4()
        )
        self._get_obs(self.context)
        self.init_dist = np.linalg.norm(self.obj_pose1[:3, 3] - self.obj_pose2[:3, 3])
        obs = self.select_obs()
        reward = self._get_reward(self.context, self.action)
        return obs, reward, done, self.info

    def step(self, action):
        self.time_step += 1
        sim_done = self.advance_time(self.context)
        if sim_done:  # terminate because of simulation error
            return [], 0, True, self.info

        self.obj_pose1 = self.plant.GetFrameByName(self.object_name).CalcPoseInWorld(self.plant_context).GetAsMatrix4()
        self.obj_pose2 = (
            self.plant.GetFrameByName(self.object_name + "_2").CalcPoseInWorld(self.plant_context).GetAsMatrix4()
        )
        self._get_obs(self.context)

        obs = self.select_obs()
        reward = self._get_reward(self.context, action)
        done = sim_done or (reward > 0) or self.need_termination

        return obs, reward, done, self.info

    def _get_reward(self, context, action):
        """knife (planar push) environment reward is based on how much two objects are pushed from one another"""
        translation_dist = np.linalg.norm(self.obj_pose1[:3, 3] - self.obj_pose2[:3, 3])
        reward = float((translation_dist - self.init_dist > 0.05) and (self.time > 5.0))

        return reward

    def update_keypoint_information(self):
        """get two keypoint positions and the direction between these two points"""
        super().update_keypoint_information()
        self.curr_tool_obj_keypoints = (self.obj_pose1[:3, 3] + self.obj_pose2[:3, 3]) / 2

    def _get_progress(self, context):
        pass
