import datetime
import os
import os.path as osp
import numpy as np
import cv2

from typing import Dict
import time
from env import *
from dotmap import DotMap
import ray
from tqdm import trange
from core.agent.agent import Agent, AgentWrapper
from collections import deque

from core.utils import *
from visdom import Visdom
from core.expert.expert import AnalyticExpert
from core.expert.human_expert import HumanExpert
import copy
import gc


def parse_task(cfg):
    register_env(cfg.task.name)
    env = make_env(cfg.task.name, task_config=cfg.task)
    return env


class ActorWrapper:
    """
    Wrapper class for agent training and logging
    Fleet Agent is the model for the fleet
    """

    def __init__(self, config, fleet_agent_id, robot_id):
        self.config = copy.deepcopy(config)
        self.robot_id = robot_id

        self.experiment_setup()
        if config.train.local_agent:
            print("build local agent")
            self.agent = eval(config.agent)(self.config, None, robot_id, self.config)

            if len(self.config.pretrained) > 0:
                # loading pretrained
                self.agent.load(self.config.pretrained, "fleet", self.config.load_pretrained_mode)
            self.agent.memory.episode_position = self.config.start_episode_position

        self.config.task.random_seed = self.config.start_episode_position + robot_id

        # setting the memory episode
        self.local_train_steps = 0
        self.fleet_agent_id = fleet_agent_id

        self.active = True
        self.parallel = self.config.parallel
        _, _, sim_fail, _ = self.env.reset()
        while sim_fail:
            _, _, sim_fail, _ = self.env.reset()

        self.local_episode = 0
        self.success_episode = 0
        self.plan_success_episode = 0
        self.avg_rewards = deque([], maxlen=1000)

        if self.config.visdom and len(self.config.task.env.numObservations) > 1:
            self.vis = Visdom(port=8097)
            _, w, h = self.config.task.env.numObservations
            self.win_id_wrist = self.vis.image(np.zeros([3, w, h]))
            self.win_id_overhead = self.vis.image(np.zeros([3, w, h]))
        self.curr_traj_for_replaybuffer = []

    def experiment_setup(self):
        self.env = parse_task(self.config)

    def reset_sim(self):
        """rebuild the manipulation station"""
        if hasattr(self, "env"):
            self.env.cleanup_meshcat()

        # c++ backend is leaking
        del self.env
        gc.collect()
        self.experiment_setup()

    def add_trajectory_transition(self, scene_info):
        """add the full trajectory into the memory"""
        save_demo = True
        # print("save data")
        if self.parallel:
            # should be blocking
            success = ray.get(
                self.fleet_agent_id.append_traj_data.remote(
                    self.curr_traj_for_replaybuffer,
                    scene_info,
                    save_demo and (self.local_episode < self.config.demo_number),
                )
            )
        else:
            success = self.fleet_agent_id.append_traj_data(
                self.curr_traj_for_replaybuffer,
                scene_info,
                save_demo and (self.local_episode < self.config.demo_number),
            )

        self.curr_traj_for_replaybuffer = []
        return success

    def reset(self):
        """reset the environment"""
        return self.env.reset()

    def visdom_visualize(self, ret):
        """publish rendered images to visdom"""
        if not len(self.config.task.env.numObservations) > 1:
            return
        head_cam_img = ret[0][1]
        wrist_cam_img = ret[0][0]  #
        self.vis.image(wrist_cam_img.transpose([2, 0, 1])[:3], win=self.win_id_wrist)
        self.vis.image(head_cam_img.transpose([2, 0, 1])[:3], win=self.win_id_overhead)

    def get_expert(self):
        """get the current expert type"""
        return eval(self.config.expert)(self.env, self.config, self.config, self.robot_id)

    def get_flags(self, time):
        """get different booleans for the current step"""
        apply_dart = (
            (time > 1)
            and (time < self.config.task.task_completion_time - 3)
            and (np.random.uniform() < self.config.task.inject_noise_to_expert)
        )
        return apply_dart

    def reset_task_episode(self, info):
        self.curr_traj_for_replaybuffer = []
        return {
            "tool_class_name": info["tool_class_name"],
            "tool_name": info["tool_name"],
            "object_name": info["tool_object_name"],
            "tool_pose": [info["tool_pose"].tolist()],
            "object_pose": [info["object_pose"].tolist()],
            "robot_joints": [info["joint_pos"].tolist()],
        }

    def rollout(self, explore=False, dagger=False, test=False):
        """policy rollout and save data"""
        self.reset_sim()  # reset the entire simulation

        episode_info = {}
        start_time = time.time()

        # more checks on this
        max_loop_iter = 10
        for idx in range(max_loop_iter):
            ret = self.env.reset()
            if not ret[-2]:
                break

        if ret[-2]:
            return episode_info, False

        if self.config.record_video:
            video_writer = cv2.VideoWriter(
                f"assets/demonstrations/{self.config.task.tool_class_name}for{self.config.task.task_name}/episode_{self.success_episode}/video_{self.local_episode}_{self.config.task.tool_fix_idx}.mp4",
                cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                5,
                (640, 480),
            )

        self.expert = self.get_expert()
        onpolicy_agent = self.agent if not self.config.run_expert else self.expert
        context = self.env.export_context()
        info = ret[-1]
        scene_info = self.reset_task_episode(info)

        self.config.task.scene_info = scene_info

        rews = 0
        planned_success = 0
        self.local_episode += 1

        for step in range(self.config.task.env.episodeLength):
            if self.config.task.dart_demonstration and self.get_flags(self.env.time):
                t = np.random.uniform(-0.02, 0.02, size=(3,))
                r = np.random.uniform(-0.2, 0.2, size=(3,))
                perturb_action = np.hstack([t, r])
                perturb_action = self.env.process_action(perturb_action)
                self.env.step(perturb_action)

                # self.env.random_perturb()
                self.expert.reset_expert()
                scene_info = self.reset_task_episode(info)

            action, extra = onpolicy_agent.select_action(ret)
            planned_success = self.config.run_expert and (self.config.teleop or not self.expert.check_plan_empty())
            if self.config.run_expert and not planned_success:  # only run planned succeeded episode
                return episode_info

            if "reset" in extra and extra["reset"]:
                print("reset env")
                return episode_info, False

            action = self.env.process_action(action)
            ret[-1]["action"] = action  # to save
            self.curr_traj_for_replaybuffer.append(copy.deepcopy([ret]))
            ret = self.env.step(action)

            # append trajectory to the replay buffer
            obs, rew, done, info = ret
            if "collision" in info or "sim_failure" in info:
                print(f"env collision: {'collision' in info} or sim terminate request: {'sim_failure' in info}")
                return episode_info, False

            if self.config.record_video:
                vis_img = self.env.img_obs[0]
                vis_img = (vis_img[..., :3] * 255).astype(np.uint8)
                video_writer.write(vis_img)

            rews += rew
            if "tool_pose" in info:  # avoid the exception case
                scene_info["tool_pose"].append(info["tool_pose"])
                scene_info["object_pose"].append(info["object_pose"])
                scene_info["robot_joints"].append(info["joint_pos"])

                if self.config.visdom:
                    self.visdom_visualize(ret)

            if done:
                break

        total_time = time.time() - start_time
        self.plan_success_episode += float(planned_success)
        self.curr_traj_for_replaybuffer.append(copy.deepcopy([ret]))
        start_time = time.time()
        save_success = self.add_trajectory_transition(scene_info)
        print(f"add / save trajectory to buffer time: {time.time() - start_time}")

        self.log_info(episode_info, scene_info, total_time, step, rews)
        return episode_info, save_success

    def log_info(self, episode_info, scene_info, total_time, step, rews):
        episode_info["episode_length"] = step
        episode_info["local_episode_reward"] = rews

        episode_info["fps"] = total_time / episode_info["episode_length"] if episode_info["episode_length"] > 0 else 0
        if self.parallel:
            episode_info["buffer_size"] = ray.get(self.fleet_agent_id.get_memory_position.remote())
        else:
            episode_info["buffer_size"] = self.fleet_agent_id.get_memory_position()

        # episode_info['global_buffer_size'] = fleet_agent.memory.size
        episode_info["local_episode"] = self.local_episode
        episode_info["object_name"] = self.local_episode
        episode_info["tool_class_name"] = self.local_episode
        episode_info["tool_name"] = self.local_episode

        # measure expert's success rates
        self.avg_rewards.append(float(rews > 0))
        self.success_episode += float(rews > 0)
        print(
            f" episode time: {total_time:.3f}"
            f" traj length: {step}"
            f" success rates: {self.success_episode}/{int(self.plan_success_episode)}"
            f" {self.success_episode/int(self.plan_success_episode):.3f}"
            f" plan success rates: {int(self.plan_success_episode)}/{self.local_episode}"
            f" {(self.plan_success_episode)/self.local_episode:.3f}"
        )

    def get_demos(self):
        return self.env.get_offline_data(self.exp_cfg.agent_cfg.num_task_transitions)

    def set_stat(self, stat):
        (self.local_episode, self.success_episode, self.plan_success_episode) = stat

    def return_stat(self):
        return (
            int(self.local_episode),
            int(self.success_episode),
            int(self.plan_success_episode),
        )


class RobotFleet:
    def __init__(self, exp_cfg):
        """Fleet of robots for running experiments"""
        self.config = exp_cfg
        self.exp_cfg = DotMap(exp_cfg)

        # Logging setup
        if len(self.config.save_dir) == 0:
            self.logdir = os.path.join(
                self.config.logdir,
                "{}_{}_{}".format(
                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    self.config.env_name + self.config.config_suffix,
                    self.config.logdir_suffix,
                ),
            )
        else:
            self.logdir = exp_cfg.save_dir

        print("LOGDIR:", self.logdir)

        self.log_freq = self.config.log_freq
        self.parallel = self.config.parallel
        print(f"number of robots in the fleet: {self.config.num_envs}")

        # Experiment setup
        if self.parallel:
            self.fleet_agent = eval(self.exp_cfg.agent + "Wrapper").remote(
                self.exp_cfg, self.logdir, -1, self.config
            )  #
            self.robots = [
                eval(self.exp_cfg.actor_wrapper).remote(self.exp_cfg, self.fleet_agent, robot_idx)
                for robot_idx in range(self.config.num_envs)
            ]
            self.fleet_agent.set_memory_position.remote(self.config.start_episode_position)

        else:
            # if not self.exp_cfg.cpu_only:
            self.fleet_agent = eval(self.exp_cfg.agent)(self.exp_cfg, self.logdir, -1, self.config)
            self.fleet_agent.set_memory_position(self.config.start_episode_position)

            self.robots = ActorWrapper(self.exp_cfg, self.fleet_agent, 0)
        self.fleet_steps = 0
        self.episode_steps = [0 for _ in range(self.config.num_envs)]
        self.num_actions = self.exp_cfg.task.env.numActions
        self.start_time = time.time()


@ray.remote(num_cpus=1, num_gpus=0.03)
class ActorWrapperGPU05(ActorWrapper):
    """
    A ray wrapper over an agent
    """

    pass
