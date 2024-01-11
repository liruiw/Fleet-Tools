"""
Built on on SAC implementation from
https://github.com/pranz24/pytorch-soft-actor-critic
"""
import random
import numpy as np
from operator import itemgetter
import IPython
import ray
from core.utils.utils import *
import os
import cv2
import json
from env.env_util import color_print


class ReplayMemory:
    """
    Replay buffer class
    """

    def __init__(self, capacity, seed, config):
        random.seed(seed)
        self.capacity = capacity

        self.task_config = config.task
        self.train_config = config.train

        # self.buffer = []
        for idx, key in enumerate(key_list):
            if key == "obs" and not self.task_config.env.use_image:
                self.obs = np.zeros(self.capacity)
            else:
                setattr(
                    self,
                    key,
                    np.zeros((self.capacity, *key_shape_list[idx]), dtype=np.float32),
                )
        self.position = 0
        self.size = 0
        self.episode_position = 0
        self.buffer_idx = 200
        self.pos_idx = np.zeros(self.capacity)
        self.seed = seed
        self.episode_positions = np.zeros(self.capacity)
        self.episode_rewards = np.zeros(self.capacity)
        self.episode_end_map = np.zeros(self.capacity)

    def __len__(self):
        return self.upper_idx()

    def reset(self):
        """reset the replay buffer"""
        for idx, key in enumerate(key_list):
            setattr(
                self,
                key,
                np.zeros((self.capacity, *key_shape_list[idx]), dtype=np.float32),
            )
        self.position = 0
        self.size = 0
        self.episode_position = 0
        self.pos_idx = np.zeros(self.capacity)
        self.episode_positions = np.zeros(self.capacity)
        self.episode_rewards = np.zeros(self.capacity)

    def push(self, **kwargs):
        """push a single data item into the replay buffer"""
        for idx, key in enumerate(key_list):
            if key == "obs":
                if type(kwargs[key]) == list and len(kwargs[key]) > 0:
                    kwargs[key] = np.concatenate(kwargs[key], axis=-1)
                if not self.task_config.env.use_image:  # dummy
                    kwargs[key] = 0.0

            try:
                getattr(self, key)[self.position] = kwargs[key]
            except Exception as e:  # would be overwrite
                print(e)
                print(
                    "push to memory failure:",
                    self.position,
                    key,
                    getattr(self, key).shape,
                    kwargs[key].shape,
                )
                return False

        timestep = self.timestep[self.position]
        reward = self.reward[self.position]
        self.pos_idx[self.position] = float(reward > 0.5)
        # print("timestep:", timestep)

        if timestep == 0:
            # print("increments index.")
            self.episode_position += 1

        self.episode_positions[self.position] = self.episode_position
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return True

    def save_scene_descriptions(self, path, info_dict):
        """save object and tool related poses into a scene descriptions"""
        arg_json = os.path.join(path, "scene_descriptions.json")
        info_dict["tool_pose"] = np.array(info_dict["tool_pose"]).tolist()
        info_dict["object_pose"] = np.array(info_dict["object_pose"]).tolist()
        info_dict["robot_joints"] = np.array(info_dict["robot_joints"]).tolist()

        with open(arg_json, "w") as f:
            json.dump(info_dict, f, indent=4, sort_keys=True)

    def load_offline(self, saved_path):
        """load offline data. Can add train/test split here."""
        for idx, key in enumerate(key_list):
            delattr(self, key)  # reset attributes

        episode_dirs = sorted(os.listdir(saved_path), key=lambda s: int(s.split("_")[1]))
        print("number of saved trajectories:", len(episode_dirs))

        traj_datas = {}
        for path in episode_dirs:
            episode_path = os.path.join(saved_path, path)
            if "traj_data.npz" in os.listdir(episode_path):
                traj_data = np.load(os.path.join(episode_path, "traj_data.npz"))
                try:
                    traj_datas = {
                        k: np.concatenate((traj_datas[k], v), axis=0) if k in traj_datas else v
                        for k, v in traj_data.items()
                    }
                except Exception as e:  # would be overwrite
                    print(e)
                    print("load trajectory folder: {} failed".format(episode_path))

        for k in key_list:
            if k in traj_datas:
                setattr(self, k, traj_datas[k])

        # compute episode index
        self.position = len(self.state) - 1
        self.size = self.position + 1
        self.pos_idx = np.zeros(len(self.state))
        self.episode_positions = traj_datas["episode_positions"]
        self.episode_position = traj_datas["episode_positions"][-1]
        self.episode_rewards = np.zeros(len(self.state))
        self.episode_dirs = episode_dirs

        print(
            "total number of data:",
            self.position,
            key_list,
            traj_datas.keys(),
            [len(traj_datas[k]) for k in key_list if k in traj_datas],
        )

    def save_episode(self, saved_path, scene_info, episode_idx):
        """saving an episode to the disk"""
        prev_episode_idx = np.where(self.episode_positions[: self.position] == episode_idx)[0]

        # skip the first episode due to boundary issues
        # print(prev_episode_idx)
        if len(prev_episode_idx) == 0 or prev_episode_idx[0] == 0 or prev_episode_idx[-1] == self.capacity - 1:
            # check the last episode step and first episode step
            print("skip empty traj")
            return False

        saved_path = os.path.join(saved_path, "episode_{}".format(episode_idx))
        mkdir_if_missing(saved_path)

        # IPython.embed()
        color_print(
            (f"saving episode {prev_episode_idx} | path: {saved_path}"),
            "light_slate_blue",
        )
        self.save_scene_descriptions(saved_path, scene_info)

        # separate out image and the rest
        try:
            images = self.obs[prev_episode_idx]
            data = {k: getattr(self, k)[prev_episode_idx] for k in key_list if k != "obs"}
            data["episode_positions"] = self.episode_positions
            # print("obs shape:", data["obs"].shape)
            # data["overhead_image"] = data["obs"][...,:5]
            data["obs"] = [f"{saved_path}/{i}" for i in range(len(prev_episode_idx))]

            np.savez(os.path.join(saved_path, "traj_data.npz"), **data)

            for i, img in enumerate(images):
                # import IPython; IPython.embed()
                cv2.imwrite(
                    f"{saved_path}/{i}_wrist_color.png",
                    data["wrist_image"][i][..., [2, 1, 0]].astype(np.uint8),
                )
                # cv2.imwrite(
                #     f"{saved_path}/{i}_wrist_depth.png",
                #     (wrist_img[..., [3]] * 1000).astype(np.uint16),
                # )
                # cv2.imwrite(f"{saved_path}/{i}_wrist_mask.png", wrist_img[..., [4]])
                cv2.imwrite(
                    f"{saved_path}/{i}_overhead_color.png",
                    data["overhead_image"][i][..., [2, 1, 0]].astype(np.uint8),
                )
                # cv2.imwrite(
                #     f"{saved_path}/{i}_overhead_depth.png",
                #     (overhead_img[..., [3]] * 1000).astype(np.uint16),
                # )
                # cv2.imwrite(f"{saved_path}/{i}_overhead_mask.png", overhead_img[..., [4]])
            return True

        except Exception as e:  # would be overwrite
            print(e)
            print("writing image failed")
            return False

    def save(self, save_dir="."):
        """Saves the current buffer to memory."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        s = time.time()
        save_dict = {}
        save_attrs = []
        for name in save_attrs:
            save_dict[name] = getattr(self, name)

        np.savez(os.path.join(save_dir, self.save_data_name), **save_dict)

    def __len__(self):
        return len(self.state)


@ray.remote(num_cpus=1)
class ReplayMemoryWrapper(ReplayMemory):
    pass
