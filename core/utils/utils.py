# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
try:
    import open3d as o3d
except:
    pass
import numpy as np
import os
import IPython
import time

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tabulate

import shutil

from .geometry import *
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *
import gym
import json

from .teleop_utils import *
from colored import fg
import GPUtil
import psutil

np.set_printoptions(precision=3)
import warnings
from pydrake.common.deprecation import DrakeDeprecationWarning

warnings.simplefilter("always", DrakeDeprecationWarning)

# anchor seeds for franka panda IK solutions
anchor_seeds = np.array(
    [
        [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0, 0],
        [2.5, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [2.8, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [2, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [2.5, 0.83, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [0.049, 1.22, -1.87, -0.67, 2.12, 0.99, -0.85, 0, 0],
        [-2.28, -0.43, 2.47, -1.35, 0.62, 2.28, -0.27, 0, 0],
        [-2.02, -1.29, 2.20, -0.83, 0.22, 1.18, 0.74, 0, 0],
        [-2.2, 0.03, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [-2.5, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56, 0, 0],
        [-2, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56, 0, 0],
        [-2.66, -0.55, 2.06, -1.77, 0.96, 1.77, -1.35, 0, 0],
        [1.51, -1.48, -1.12, -1.55, -1.57, 1.15, 0.24, 0, 0],
        [-2.61, -0.98, 2.26, -0.85, 0.61, 1.64, 0.23, 0, 0],
    ]
)

# predefined lists for replay buffer, environment, and policy
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

key_shape_list = [
    (27,),
    (6,),
    (1,),
    (27,),
    (1,),
    (480, 640, 10),
    (1,),
    (9,),
    (4, 4),
    (4, 4),
    (16,),
    (3, 3),
    (4, 3),
    (3, 3),
    (4, 4),
    (3, 6),
    (9,),
    (6,),
    (6 * 32,),
    (4, 1024),
    (4, 4),
    (384,),
    (512, 512, 3),
    (512, 512, 3),
]


def color_print(str, color="white"):
    print(str)  #


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def mkdir_if_missing(dst_dir, cleanup=False):
    if os.path.exists(dst_dir) and cleanup:
        shutil.rmtree(dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def print_and_write(file_handle, text):
    print(text)
    if file_handle is not None:
        file_handle.write(text + "\n")
    return text


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def get_usage():
    GPUs = GPUtil.getGPUs()
    memory_usage = psutil.virtual_memory().percent
    try:
        gpu_usage = max([GPU.memoryUsed for GPU in GPUs])
    except:
        gpu_usage = 1e4  # why?
    return gpu_usage, memory_usage


def visualize_image_observation(state):
    if type(state) is not np.ndarray:
        state = state.detach().cpu().numpy()

    for idx in range(0, len(state), 2):
        fig = plt.figure(figsize=(25.6, 9.6))
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow((state[idx][:3].transpose(1, 2, 0) * 255).astype(np.uint8))

        if state[idx].shape[0] == 4:
            ax = fig.add_subplot(1, 3, 2)
            # IPython.embed()
            plt.imshow(state[idx][-1])

        if state[idx].shape[0] >= 5:
            ax = fig.add_subplot(1, 3, 2)
            plt.imshow(state[idx][-2])

            ax = fig.add_subplot(1, 3, 3)
            mask = (state[idx][-1]).astype(np.uint8)
            plt.imshow(mask)
        plt.show()
        # break


def randomize_sphere_lookat(target, near, far, x_near, x_far, theta_low=0, theta_high=np.pi / 2):
    """
    randomly initialize pose to look at some center.
    useful for initializing wrist or overhead cameras
    """

    res = None
    # the object is not on the table in the beginning
    theta = np.random.uniform(low=theta_low, high=theta_high)
    phi = np.random.uniform(low=np.pi / 2, high=3 * np.pi / 2)
    # top sphere
    r = np.random.uniform(low=near, high=far)
    # sphere radius
    pos = np.array(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    )

    # noise and clip
    trans = pos + target + np.random.uniform(-0.03, 0.03, 3)
    trans[2] = np.clip(trans[2], 0.3, 0.7)
    trans[1] = np.clip(trans[1], -0.4, 0.4)
    trans[0] = np.clip(trans[0], x_near, x_far)

    rand_vec = [0, 0, -1]
    pos = trans - target

    # open gl coordinate transform
    R = inv_lookat(pos, 2 * pos, rand_vec).dot(rotZ(np.pi / 2)[:3, :3])

    # from end effector to the world
    target_pose = np.eye(4)
    target_pose[:3, 3] = trans
    target_pose[:3, :3] = R

    eff_up_vec = R.dot(np.array([1, 0, 0]))
    eff_forward_vec = R.dot(np.array([0, 0, 1]))

    return target_pose


import numpy as np

def projection_to_intrinsics(mat, width=224, height=224):
    """
    Convert a projection matrix to intrinsic matrix.

    Args:
        mat (array-like): The projection matrix.
        width (int, optional): The width of the image. Defaults to 224.
        height (int, optional): The height of the image. Defaults to 224.

    Returns:
        array-like: The intrinsic matrix.

    """
    intrinsic_matrix = np.eye(3)
    mat = np.array(mat).reshape([4, 4]).T
    fv = width / 2 * mat[0, 0]
    fu = height / 2 * mat[1, 1]
    u0 = width / 2
    v0 = height / 2

    intrinsic_matrix[0, 0] = fu
    intrinsic_matrix[1, 1] = fv
    intrinsic_matrix[0, 2] = u0
    intrinsic_matrix[1, 2] = v0
    return intrinsic_matrix


def backproject_camera_target(im_depth, K, target_mask=None):
    """
    Backprojects the camera target from the depth image.

    Args:
        im_depth (numpy.ndarray): The depth image.
        K (numpy.ndarray): The camera intrinsic matrix.
        target_mask (numpy.ndarray, optional): The mask indicating the target region. Defaults to None.

    Returns:
        numpy.ndarray: The backprojected camera target.
    """
    Kinv = np.linalg.inv(K)

    width = im_depth.shape[1]
    height = im_depth.shape[0]
    depth = im_depth.astype(np.float32, copy=True).flatten()

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

    # backprojection
    R = Kinv.dot(x2d.transpose())  #
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    # X[1] *= -1 # not sure if this is needed

    if target_mask is not None:
        target_mask = (depth != 0) * (target_mask.flatten() == 0)
        return X[:, target_mask]
    return X


def inv_lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
    """Generate LookAt matrix."""
    up = np.random.uniform(size=(3,))
    eye = np.float32(eye)  #  trans - target
    forward = normalize(target - eye)  #  trans - target
    side = normalize(np.cross(forward, up))
    up = np.cross(side, forward)  # redo

    R = np.stack([side, up, -forward], axis=-1)
    return R


def safe_div(a, b):
    return a / (b + 1e-10)


def save_args_json(path, args):
    mkdir_if_missing(path)
    arg_json = os.path.join(path, "config.json")
    with open(arg_json, "w") as f:
        # args = vars(args)
        json.dump(args, f, indent=4, sort_keys=True)


def make_video_writer(name, window_width, window_height):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(name, fourcc, 10.0, (window_width, window_height))


def write_video(
    traj,
    scene_file,
    expert_traj=None,
    IMG_SIZE=(112, 112),
    output_dir="output_misc/",
    target_name="",
    extra_text=None,
):
    """
    Writes a video file based on the given trajectory and scene file.

    Args:
        traj (list): List of images representing the trajectory.
        scene_file (str): Path to the scene file.
        expert_traj (list, optional): List of expert trajectory images. Defaults to None.
        IMG_SIZE (tuple, optional): Size of the output video frames. Defaults to (112, 112).
        output_dir (str, optional): Directory to save the output video. Defaults to "output_misc/".
        target_name (str, optional): Name of the target. Defaults to "".
        extra_text (str, optional): Additional text to add to the video frames. Defaults to None.
    """
    ratio = 1 if expert_traj is None else 2
    video_writer = make_video_writer(
        os.path.join(
            output_dir + "_{}_rollout.avi".format(scene_file),
        ),
        int(ratio * IMG_SIZE[1]),
        int(IMG_SIZE[0]),
    )

    for i in range(len(traj)):
        img = traj[i][..., :3] * 255
        if expert_traj is not None:
            idx = min(len(expert_traj) - 1, i)
            img = np.concatenate((img, expert_traj[idx][..., [2, 1, 0]]), axis=1)

        img = img.astype(np.uint8)
        if extra_text is not None:
            img = add_extra_text(img, extra_text)
        video_writer.write(img)


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def PandaTaskSpace6D(trans_scale=0.04, rotation_scale=np.pi / 8):
    """end effector action space for the franka environment"""
    high = np.array(
        [
            trans_scale,
            trans_scale,
            trans_scale,
            rotation_scale,
            rotation_scale,
            rotation_scale,
        ]
    )  # np.pi/10
    low = np.array(
        [
            -trans_scale,
            -trans_scale,
            -trans_scale,
            -rotation_scale,
            -rotation_scale,
            -rotation_scale,
        ]
    )  # -np.pi/3
    shape = [6]
    # self.bounds = np.vstack([self.low, self.high])
    return gym.spaces.Box(low=low, high=high, shape=shape)


def get_interp_time(curr_time, finish_time, ratio):
    """get interpolated time between curr and finish"""
    return (finish_time - curr_time) * ratio + curr_time


def save_args_hydra(path, cfg):
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    save_args_json(path, cfg_dict)


def tabulate_print_state(state_dict):
    """
    Print ego car state dict
    """
    state_dict = sorted(state_dict.items())
    headers = [kv[0] for kv in state_dict]
    data = [[kv[1] for kv in state_dict]]
    print(tabulate.tabulate(data, headers, tablefmt="psql", floatfmt=".2f"))


def proj_point_img(
    img,
    K,
    offset_pose,
    points,
    color=(255, 0, 0),
    neg_y=False,
):
    xyz_points = offset_pose[:3, :3].dot(points) + offset_pose[:3, [3]]
    if neg_y:
        xyz_points[:2] *= -1
    p_xyz = K.dot(xyz_points)
    p_xyz = p_xyz[:, p_xyz[2] > 0.03]
    x, y = (p_xyz[0] / p_xyz[2]).astype(np.int), (p_xyz[1] / p_xyz[2]).astype(np.int)
    valid_idx_mask = (x > 0) * (x < img.shape[1] - 1) * (y > 0) * (y < img.shape[0] - 1)
    img[y[valid_idx_mask], x[valid_idx_mask], :3] = [0, 1, 1]
    return img
