# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import os
import sys
import random

import scipy.io as sio
import IPython
import time

import cv2

import matplotlib.pyplot as plt
import tabulate
import yaml


import copy
import math

# transform stuff
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *

# from core.utils.torch_utils import *


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def inv_lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
    """Generate LookAt matrix."""
    eye = np.float32(eye)
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    up = np.cross(side, forward)
    R = np.stack([side, up, -forward], axis=-1)
    return R


def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ


def rotY(roty):
    RotY = np.array(
        [
            [np.cos(roty), 0, np.sin(roty), 0],
            [0, 1, 0, 0],
            [-np.sin(roty), 0, np.cos(roty), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotY


def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX


def ros_quat(tf_quat):  # wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


def tf_quat(ros_quat):  # xyzw -> wxyz
    quat = np.zeros(4)
    quat[0] = ros_quat[-1]
    quat[1:] = ros_quat[:-1]
    return quat


def se3_inverse(RT):
    RT = RT.reshape(4, 4)
    R = RT[:3, :3]
    T = RT[:3, 3].reshape((3, 1))
    RT_new = np.eye(4, dtype=np.float32)
    RT_new[:3, :3] = R.transpose()
    RT_new[:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new


def se3_inverse_tensor(RT):
    R = RT[:, :3, :3]
    T = RT[:, :3, 3].view((-1, 3, 1))
    RT_new = torch.eye(4).repeat((len(R), 1, 1))
    RT_new[:, :3, :3] = R.permute(0, 2, 1)
    RT_new[:, :3, 3] = -1 * torch.matmul(R.permute(0, 2, 1), T).view(-1, 3)
    return RT_new


def unpack_action_batch(actions):
    pose_deltas = []
    for action in actions:
        pose_delta = np.eye(4)
        pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
        pose_delta[:3, 3] = action[:3]
        pose_deltas.append(pose_delta)
    return np.stack(pose_deltas, axis=0)


def unpack_action(action, rot_type="euler"):
    if rot_type == "euler":
        rot_func = euler2mat_
    elif rot_type == "axangle":
        rot_func = axangle2mat_

    pose_delta = np.eye(4)
    pose_delta[:3, :3] = rot_func(action[3:])
    pose_delta[:3, 3] = action[:3]

    return pose_delta


def invert_action_batch(actions):
    pose_deltas = []
    for action in actions:
        pose_delta = np.eye(4)
        pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
        pose_delta[:3, 3] = action[:3]
        pose_deltas.append(pack_pose(se3_inverse(pose_delta)))
    return np.stack(pose_deltas, axis=0)


def invert_action(action):
    pose_delta = np.eye(4)
    pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
    pose_delta[:3, 3] = action[:3]
    return pack_pose(se3_inverse(pose_delta))


def invert_action_pose(action):
    pose_delta = np.eye(4)
    pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
    pose_delta[:3, 3] = action[:3]
    return se3_inverse(pose_delta)


def unpack_poses_batch(pose):
    pose_mat = torch.eye(4, device=pose.device).repeat(len(pose), 1, 1)
    pose_mat[:, :3, 3] = pose[:, :3]
    pose_mat[:, :3, :3] = quat2mat_batch(pose[:, 3:7])
    return pose_mat


def relative_pose_between_bodies(pose1, pose2):
    """
    get the relative pose from pose2 to pose1
    """
    pose_mat1 = unpack_poses_batch(pose1)
    pose_mat2 = unpack_poses_batch(pose2)
    return se3_inverse_tensor(pose_mat1) @ pose_mat2


def unpack_pose_pytorch3d(pose):
    pose_mat = torch.eye(4, device=pose.device).repeat(len(pose), 1, 1)
    pose_mat[:, :3, 3] = pose[:, :3]
    pose_mat[:, :3, :3] = quat2mat_batch(pose[:, 3:7])
    return pose_mat


def se3_inverse_batch(RT):
    R = RT[:, :3, :3]
    T = RT[:, :3, 3].reshape((-1, 3, 1))
    RT_new = np.tile(np.eye(4, dtype=np.float32), (len(R), 1, 1))
    RT_new[:, :3, :3] = R.transpose(0, 2, 1)
    RT_new[:, :3, 3] = -1 * np.matmul(R.transpose(0, 2, 1), T).reshape(-1, 3)
    return RT_new


def unpack_action_pytorch3d(pose):
    pose_mat = torch.eye(4, device=pose.device).repeat(len(pose), 1, 1)
    pose_mat[:, :3, 3] = pose[:, :3]
    pose_mat[:, :3, :3] = to_rotation_matrix(pose[:, 3:])
    return pose_mat


def mat2euler_torch(rot_mat, seq="xyz"):
    """
    convert rotation matrix to euler angle
    :param rot_mat: rotation matrix rx*ry*rz [B, 3, 3]
    :param seq: seq is xyz(rotate along z first) or zyx
    :return: three angles, x, y, z
    """
    r11 = rot_mat[:, 0, 0]
    r12 = rot_mat[:, 0, 1]
    r13 = rot_mat[:, 0, 2]
    r21 = rot_mat[:, 1, 0]
    r22 = rot_mat[:, 1, 1]
    r23 = rot_mat[:, 1, 2]
    r31 = rot_mat[:, 2, 0]
    r32 = rot_mat[:, 2, 1]
    r33 = rot_mat[:, 2, 2]
    if seq == "xyz":
        z = torch.atan2(-r12, r11)
        y = torch.asin(r13)
        x = torch.atan2(-r23, r33)
    else:
        y = torch.asin(-r31)
        x = torch.atan2(r32, r33)
        z = torch.atan2(r21, r11)
    return torch.stack((z, y, x), dim=1)


def pack_action_pytorch3d(pose_mat):
    pose = torch.zeros(len(pose_mat), 6).to(pose_mat.device)
    pose[:, :3] = pose_mat[:, :3, 3]
    pose[:, 3:6] = mat2euler_torch(pose_mat[:, :3, :3])
    return pose


def euler2mat_(euler):
    return euler2mat(euler[0], euler[1], euler[2])


def unpack_pose(pose, rot_first=False, rot_type="quat"):
    rot_index = 4 if rot_type == "quat" else 3

    if rot_type == "quat":
        rot_func = quat2mat
    elif rot_type == "euler":
        rot_func = euler2mat_
    elif rot_type == "axangle":
        rot_func = axangle2mat_

    unpacked = np.eye(4)
    if rot_first:
        unpacked[:3, :3] = rot_func(pose[:rot_index])
        unpacked[:3, 3] = pose[rot_index:]
    else:
        unpacked[:3, :3] = rot_func(pose[rot_index:])
        unpacked[:3, 3] = pose[:rot_index]
    return unpacked


def _hat(x):
    """
    hat operator, compute skew matrix for cross product
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def rodrigue_formula(p, q):
    """
    allocentric to egocentric
    """
    r = _hat(np.cross(p, q))
    R_a2e = np.eye(3) + r + r.dot(r) / (1 + np.dot(p, q))
    return R_a2e


def mat2axangle_(mat):
    axangle = mat2axangle(mat[:3, :3])
    return axangle[0] * axangle[1]


def axangle2mat_(axangle):
    angle = np.linalg.norm(axangle)
    axis = axangle / angle
    return axangle2mat(axis, angle)


def action_rot_repr_from_euler_to_axangle(action):
    return pack_pose(unpack_action(action), rot_type="axangle")


def action_rot_repr_from_axangle_to_euler(action):
    return pack_pose(unpack_action(action, rot_type="axangle"))


def pack_pose(pose, rot_type="euler"):  # for action
    rot_index = 4 if rot_type == "quat" else 3
    if rot_type == "quat":
        rot_func = mat2quat
    elif rot_type == "euler":
        rot_func = mat2euler
    elif rot_type == "axangle":
        rot_func = mat2axangle_
    packed = np.zeros(3 + rot_index)
    packed[3:] = rot_func(pose[:3, :3])
    packed[:3] = pose[:3, 3]
    return packed


def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ


def rotY(roty):
    RotY = np.array(
        [
            [np.cos(roty), 0, np.sin(roty), 0],
            [0, 1, 0, 0],
            [-np.sin(roty), 0, np.cos(roty), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotY


def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def compose_orn(curr_quat, action_rpy):
    q = quat_from_euler_xyz(action_rpy[..., 0], action_rpy[..., 1], action_rpy[..., 2])
    return quat_mul(curr_quat, q)


def quat2mat_batch(norm_quat):
    """Convert quaternion coefficients to rotation matrix."""
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = norm_quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).reshape(B, 3, 3)
    return rotMat


def get_frame_spatial_velocity(frame_A, frame_B, dt):
    """ """
    rel_pose = se3_inverse(frame_A) @ frame_B
    translation_diff = rel_pose[:3, 3]
    rot_diff = mat2axangle_(rel_pose[:3, :3])
    v = np.concatenate((translation_diff, rot_diff)) / dt
    # print("velocity:", v)
    # zero velocity
    return np.zeros_like(v)
