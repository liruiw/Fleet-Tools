import math
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import os
import IPython
import numpy as np
import cv2
import copy
import json


def process_image(img):
    """process RGB image"""
    color_img = img[..., :3]
    img = cv2.resize(img, IMG_SIZE)
    return img


def load_image(path):
    wrist_img_color = cv2.imread(f"{path}_wrist_color.png", cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]] / 255.0
    wrist_img_depth = cv2.imread(f"{path}_wrist_depth.png", cv2.IMREAD_UNCHANGED)[..., None] / 1000.0
    wrist_img_mask = cv2.imread(f"{path}_wrist_mask.png", cv2.IMREAD_UNCHANGED)[..., None]

    overhead_img_color = cv2.imread(f"{path}_overhead_color.png", cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]] / 255.0
    overhead_img_depth = cv2.imread(f"{path}_overhead_depth.png", cv2.IMREAD_UNCHANGED)[..., None] / 1000.0
    overhead_img_mask = cv2.imread(f"{path}_overhead_mask.png", cv2.IMREAD_UNCHANGED)[..., None]

    wrist_img_obs = np.concatenate((wrist_img_color, wrist_img_depth, wrist_img_mask), axis=-1)
    overhead_img_obs = np.concatenate(
        (overhead_img_color, overhead_img_depth, overhead_img_mask),
        axis=-1,
    )
    overhead_img_obs = process_image(overhead_img_obs)
    wrist_img_obs = process_image(wrist_img_obs)

    new_data_obs = [overhead_img_obs, wrist_img_obs]
    return np.array(new_data_obs)


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


parser = argparse.ArgumentParser()
parser.add_argument("--saved_path", "-s", type=str, default="assets/demonstrations")
parser.add_argument("--env_name", "-e", type=str, default="wrench")
parser.add_argument("--tool_idx", "-t", type=int, default=1)
parser.add_argument("--max_num", type=int, default=100000)
parser.add_argument("--output_path", "-o", type=str, default="assets/demonstrations/processed")
parser.add_argument("--processed_output_path", type=str, default="data/demo_drake/")

args = parser.parse_args()

IMG_SIZE = (128, 128)
saved_path = os.path.join(args.saved_path, args.env_name + f"tool_{args.tool_idx}")
output_path = os.path.join(args.output_path, args.env_name + f"tool_{args.tool_idx}")
mkdir_if_missing(saved_path)
mkdir_if_missing(output_path)

TEST_EPS_NUM = 200

# read the offline folder and just save state / action
episode_dirs = sorted(os.listdir(saved_path), key=lambda s: int(s.split("_")[1]))  # [:-TEST_EPS_NUM]
# note! the last ten is used for testing
MAX_NUM = args.max_num
USE_IMAGE = False
images = []
traj_datas = {}
traj_datas["tool_name"] = []
traj_datas["object_name"] = []
traj_datas["task_name"] = []

for path in episode_dirs:
    curr_num = len(images)
    episode_path = os.path.join(saved_path, path)
    print(path)

    if "traj_data.npz" in os.listdir(episode_path):
        traj_data = np.load(os.path.join(episode_path, "traj_data.npz"))
        traj_data = dict(traj_data)
        scene_descriptions = json.load(open(os.path.join(episode_path, "scene_descriptions.json")))
        # traj_data["global_scene_index"] =
        traj_data["tool_name"] = [scene_descriptions["tool_name"]] * len(traj_data["action"])
        traj_data["object_name"] = [scene_descriptions["object_name"]] * len(traj_data["action"])
        traj_data["task_name"] = [saved_path[saved_path.find("for") + len("for") : saved_path.rfind("tool")]] * len(
            traj_data["action"]
        )
        # print(traj_data['action'].shape)
        try:
            traj_datas = {
                k: np.concatenate((traj_datas[k], v), axis=0) if k in traj_datas else v for k, v in traj_data.items()
            }
        except:
            print("load trajectory folder: {} failed".format(episode_path))
            traj_datas = {k: v[:curr_num] for k, v in traj_data.items()}
            images = images[:curr_num]

    if len(traj_datas["action"]) > MAX_NUM:
        break

if USE_IMAGE:
    images = np.array(images)
    wrist_view_only = False
    if wrist_view_only:
        images = images[:, 1]  # , ..., :3]
    else:
        images = np.concatenate((images[:, 0], images[:, 1]), axis=-1)

# read images
mkdir_if_missing(f"{ output_path}/")
print(f"save to {output_path}")
print("number of saved trajectories:", len(episode_dirs), traj_datas["action"].shape, path)

key_list = [
    "state",
    "action",
    "reward",
    "next_state",
    "done",
    "obs",
    "timestep",
    "joint_pos",
    "hand_camera_pose",
    "base_camera_pose",
    "ee_pose",
    "cam_intr",
    "curr_keypoint",
    "goal_keypoint",
    "goal_pose",
    "keypoint_wrench",
    "external_torque",
    "endeffector_wrench",
    "history_endeffector_wrench",
    "point_cloud",
    "tool_rel_pose",
    "img_feature",
    "overhead_image",
    "wrist_image",
]

# for idx, combined_pc in enumerate(traj_datas["point_cloud"][::100]):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(combined_pc[:3].T)
#     pc_colors = np.zeros_like(pcd.points)
#     pc_colors[combined_pc.T[:, -1] == 1] = (1,0,0)
#     pc_colors[combined_pc.T[:, -1] == 0] = (0,1,0)
#     pcd.colors = o3d.utility.Vector3dVector(pc_colors)

#     o3d.visualization.draw_geometries([pcd]) #

data_dict = {
    "tool_name": traj_datas["tool_name"],
    "task_name": traj_datas["task_name"],
    "object_name": traj_datas["object_name"],
}
traj_datas["tool_name"] = traj_datas["tool_name"]
traj_datas["task_name"] = traj_datas["task_name"]
traj_datas["object_name"] = traj_datas["object_name"]

# remove one copy
# for key in key_list:
#    data_dict[key] = traj_datas[key]

np.savez(f"{ output_path}/demo_state.npz", **traj_datas)
print("saving offline dataset on:", f"{output_path}/demo_state.npz")
output_path2 = f"{args.processed_output_path}/FrankaDrake{args.env_name.capitalize()}Env-Tool{args.tool_idx}"
mkdir_if_missing(f"{output_path2}/")

# save in two places
np.savez(f"{output_path2}/demo_state.npz", **traj_datas)
print("saving offline dataset on:", f"{ output_path2}/demo_state.npz")
