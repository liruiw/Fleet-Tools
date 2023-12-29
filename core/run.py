import IPython
import warnings

warnings.simplefilter("ignore")

import os.path as osp
from core.experiment import *
from core.expert import *
from core.utils.utils import *

from core.agent.agent import Agent, AgentWrapper
from pydrake.all import RandomGenerator
from collections import deque

import IPython
import numpy as np
import GPUtil
import psutil
import hydra
import os
from omegaconf import OmegaConf

os.environ["WANDB_SILENT"] = "true"
MEMORY_THRE = 75


def choose_setup(args):
    actor_wrapper = "ActorWrapperGPU05"
    max_memory = psutil.virtual_memory().available * 0.5
    print("worker: {} memory: {}".format(args.num_envs, int(max_memory)))
    return actor_wrapper, max_memory


def monitor_usage():
    """monitor whether gpu or cpu memory has reached a state that needs reset"""
    gpu_usage, memory_usage = get_usage()
    GPUs = GPUtil.getGPUs()
    gpu_limit = np.max([GPU.memoryTotal for GPU in GPUs])
    gpu_max = float(gpu_usage) / gpu_limit > 0.98
    memory_max = memory_usage >= MEMORY_THRE
    print(f"==================== Memory: {memory_usage} GPU: {gpu_usage} =====================")

    return memory_max or gpu_max


def reinit(robot):
    """reiniting ray for memory leak"""
    print("============================ Reinit ==========================")
    robot.config.start_episode_position = (
        ray.get(robot.fleet_agent.get_exp_stats.remote())
        if robot.config.parallel
        else robot.fleet_agent.get_exp_stats()
    )
    os.system("nvidia-smi")
    print(f"===================== Ray Reinit Episode: {robot.config.start_episode_position} =================")
    if robot.config.parallel:
        curr_stat = ray.get([r.return_stat.remote() for r in robot.robots])
    else:
        curr_stat = robot.robots.return_stat()

    time.sleep(4)
    ray.shutdown()
    time.sleep(4)
    from env.env_util import g

    # easy way to generate stochasty
    g = RandomGenerator(np.random.randint(1e10) + robot.config.start_episode_position)
    np.random.seed(g())
    robot = RobotFleet(robot.config)  # reinitialize
    print(None, "==============================================================")
    if robot.config.parallel:
        ray.get([r.set_stat.remote(curr_stat[idx]) for idx, r in enumerate(robot.robots)])
    else:
        robot.robots.set_stat(curr_stat)
    return robot


def run_episode(robot):
    """run the training experiment episode by episode"""
    fleet_steps = 0
    saved_episodes = 0

    if robot.config.eval:
        robot.fleet_agent.load(robot.config.eval)

    if robot.config.record_video and fleet_steps <= 3:
        from env.env_util import visualizer, meshcat

        if visualizer is not None:
            animation = visualizer.StartRecording(set_transforms_while_recording=False)

    total_rewards, total_lengths, scene_descriptions = [], [], []
    print(robot.config.num_episode, robot.config.max_episodes)
    while fleet_steps < robot.config.num_episode:
        # load weights
        if robot.config.parallel:
            res = ray.get([r.rollout.remote(1) for r in robot.robots])
            info = [r[0] for r in res]
            success = [r[1] for r in res]
            info = {k: np.mean([info_i[k] for info_i in info if k in info_i]) for k in info[0].keys()}
        else:
            info, success = robot.robots.rollout(1)

        if robot.config.training:
            train_info = (
                ray.get(robot.fleet_agent.train.remote(fleet_steps))
                if robot.config.parallel
                else robot.fleet_agent.train(fleet_steps)
            )
            info = merge_two_dicts(info, train_info)

        info["total_time"] = time.time() - robot.start_time
        info["log_dir"] = robot.logdir
        info["global_episode"] = fleet_steps
        saved_episodes += np.sum(success)
        success = 0

        if robot.config.record_video and fleet_steps <= 5 and visualizer is not None:
            # and fleet_steps % 10 == 0 might only work with one runner
            visualizer.StopRecording()
            visualizer.PublishRecording()
            html = meshcat.StaticHtml()

            task_dir = f"assets/demonstrations/{robot.config.task.tool_class_name}for{robot.config.task.task_name}/episode_{fleet_steps}"
            mkdir_if_missing(task_dir)

            # it should be the last one
            with open(f"{task_dir}/recording.html", "w") as f:
                print("write to video path", task_dir)
                f.write(html)
            visualizer.DeleteRecording()  # delete
            animation = visualizer.StartRecording(set_transforms_while_recording=False)

            # restart recording after certain number of episodes
        if "episode_length" in info:
            total_lengths.append(info["episode_length"])
            total_rewards.append(info["local_episode_reward"])
            scene_descriptions.append((info["tool_class_name"], info["tool_name"], info["object_name"]))

        # reiniting
        if fleet_steps % robot.config.reinit_num == 0 or monitor_usage():
            robot = reinit(robot)

        fleet_steps += 1
        print("curr eps:", saved_episodes, fleet_steps)
        if saved_episodes > robot.config.max_episodes:
            break

    print("total number of saved trajectories:", robot.config.max_episodes)
    print(f"log rollout results: {len(total_rewards)} {np.sum(total_rewards)} {np.mean(total_lengths)}")


@hydra.main(config_path="../experiments/config", config_name="config", version_base="1.2")
def main(cfg):

    print(OmegaConf.to_yaml(cfg, resolve=True))
    np.random.seed(cfg.seed)

    cfg.parallel = cfg.num_envs > 1
    actor_wrapper, max_memory = choose_setup(cfg)
    cfg.actor_wrapper = actor_wrapper
    cfg.max_memory = max_memory
    cfg.task.random_seed = cfg.seed

    # Create experiment and run it
    if cfg.parallel:
        ray.init(
            _temp_dir=f"{os.environ['HOME']}/tmp",
            num_cpus=cfg.num_envs + 1,
            object_store_memory=int(cfg["max_memory"]),
        )
    robot = RobotFleet(cfg)
    save_args_hydra(robot.logdir, cfg)

    start_time = time.time()
    run_episode(robot)
    print("rollout total time: {:.3f}".format(time.time() - start_time))


if __name__ == "__main__":
    main()
