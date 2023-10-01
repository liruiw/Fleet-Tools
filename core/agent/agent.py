# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os

import numpy as np
from core.utils.utils import *
import IPython
import time
from core.agent.replay_memory import ReplayMemory
import os.path as osp
import ray
import gym


# from core.utils.torch_utils import *


class Agent(object):
    """
    A general agent class
    """

    def __init__(self, config, logdir, agent_id=-1, args=None):
        self.env_name = config.env_name
        self.logdir = logdir
        self.device = "cuda"
        self.agent_id = agent_id
        self.image_input = config.task.env.use_image
        self.keypoint_input = config.task.env.use_keypoint

        self.log_freq = config.log_freq
        self.batch_size = config.train.learn.minibatches
        self.latent_size = config.train.policy.latent_size

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=config.train.numObservations)
        self.action_space = PandaTaskSpace6D(*config.task.env.action_scale)  # action_space(config.task.env.numActions)
        self.config = config
        self.train_iter = 0
        self.train_time = 0
        self.memory = ReplayMemory(config.train.replaybuffer_size, config.train.seed, config)

        if not config.run_expert:  # no network is needed
            self.init_policies()

    def get_self(self):
        return self

    def get_exp_stats(self):
        """get experiment details"""
        return self.memory.episode_position

    def select_action(self, state):
        """policy generates action. not used."""
        extra = {}
        self.set_mode(True)
        state = self.select_state_test(state)
        actions = self.policy.sample(state)[0][0].detach().cpu().numpy()
        return actions, extra

    def train(self, step):
        """training loop for the policy. not used."""
        start_time = time.time()
        self.set_mode(test=False)

        for _ in range(self.config.train.train_frequency):
            self.train_iter += 1
            iter_start_time = time.time()
            if not hasattr(self, "memory_dataloader_iter"):
                data = self.memory.sample(batch_size=self.batch_size)
            else:
                data = next(self.memory_dataloader_iter)
            self.data_time = time.time() - iter_start_time

        self.train_time += time.time() - start_time
        self.get_lr()
        info = {k: float(getattr(self, k)) for k in self.info if hasattr(self, k)}
        return info

    def train_step(self, memory, batch_size):
        pass

    def step(self):
        """step the optimizer and scheduler"""
        self.log_stat()

    def log_stat(self):
        """log grad and param statistics for tensorboard"""
        self.train_batch_size = len(self.state_batch)

    def set_memory_position(self, start_episode_position):
        self.memory.episode_position = start_episode_position

    def get_memory_position(self):
        return self.memory.size

    # functions for ray remote calls
    def append_traj_data(self, traj_infos, scene_info, save_demo=False):
        """add a full trajectory to the replay buffer and then save it to disk"""
        episode_idx = self.memory.episode_position + 1
        success = True
        status = False

        try:
            start_time = time.time()
            for t in traj_infos:
                if t[0] is not None:
                    obs, reward, done, info = t[0]
                    info["reward"] = reward
                    info["obs"] = obs
                    info["done"] = done
                success = self.memory.push(**info)
                if not success:
                    print("push failed")
                    status = False
                    break

            if reward > 0 and save_demo and success:
                # save an episode in the buffer to local disk
                print(f"saving traj data: {episode_idx}")
                demo_path = "assets/demonstrations/{}".format(
                    self.config.task.tool_class_name + "for" + self.config.task.task_name + self.config.save_demo_suffix
                )
                start_time = time.time()
                status = self.memory.save_episode(demo_path, scene_info, episode_idx)
                curr_pos = self.memory.position
                self.memory.episode_end_map[curr_pos - len(traj_infos) : curr_pos] = curr_pos - 1
                print(f"save episode time: {time.time() - start_time:.3f} to {demo_path}")

        except Exception as e:
            print(e)
            print("saving memory failed:", episode_idx)
        return status


@ray.remote(num_cpus=1, num_gpus=0.1)
class AgentWrapper(Agent):
    pass
