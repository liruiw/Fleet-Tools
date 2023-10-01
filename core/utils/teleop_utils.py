# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


import numpy as np
import time

import cv2
import matplotlib.pyplot as plt
import tabulate
import yaml
from collections import deque
import shutil

from .geometry import *
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *

from colored import fg

joystick_lowpass_filter = deque([np.zeros(6)], maxlen=30)
vr_has_init = False
rec_action_pose = None
vr_frame_origin = None
oculus_reader = None


def get_keyboard_input(action, robot_id):
    finger_state = ""
    cv2.imshow(
        "robot_{}_teleop_wsadqe_translation_ikjluo_rotation".format(robot_id), np.zeros((64, 64, 3), dtype=np.uint8)
    )
    k = cv2.waitKey(1)
    action = np.zeros_like(action)
    trans_scale = 0.01
    rot_scale = 0.05

    if k == ord("w"):
        action[0] = trans_scale
    elif k == ord("s"):
        action[0] = -trans_scale
    elif k == ord("a"):
        action[1] = trans_scale
    elif k == ord("d"):
        action[1] = -trans_scale
    elif k == ord("q"):
        action[2] = trans_scale
    elif k == ord("e"):
        action[2] = -trans_scale

    elif k == ord("i"):
        action[3] = rot_scale
    elif k == ord("k"):
        action[3] = -rot_scale
    elif k == ord("j"):
        action[4] = rot_scale
    elif k == ord("l"):
        action[4] = -rot_scale
    elif k == ord("u"):
        action[5] = rot_scale
    elif k == ord("o"):
        action[5] = -rot_scale

    elif k == ord("n"):
        finger_state = "open"
    elif k == ord("m"):
        finger_state = "close"
    return action, finger_state


def get_mouse_input(action, robot_id):
    # Space mouse
    global joystick_lowpass_filter
    event = spacenav.poll()
    finger_state = ""
    action = np.zeros_like(action)
    trans_scale = 0.02
    rot_scale = 0.05
    normalize_scale = 512
    deadband = 0.1

    if event is spacenav.MotionEvent:
        if event.x != -2147483648:
            action[0] = event.z / normalize_scale
            action[1] = -event.x / normalize_scale
            action[2] = event.y / normalize_scale
            action[3] = event.rz / normalize_scale
            action[4] = -event.rx / normalize_scale
            action[5] = event.ry / normalize_scale

            if (np.abs(action[:3]) > deadband).sum() == 0:
                action[:3] = 0
            if (np.abs(action[3:]) > deadband).sum() == 0:
                action[3:] = 0

            action[:3] *= trans_scale
            action[3:] *= rot_scale

            spacenav_prev_action[:] = action
            # print("action:", action)

        if hasattr(event, "pressed"):
            if event.pressed:
                finger_state = "close"
    print("action: ", joystick_lowpass_filter)
    joystick_lowpass_filter.append(action)
    action = np.mean(list(joystick_lowpass_filter), axis=0)
    return action, finger_state


class LowPassFilter:
    def __init__(self, dimension: int, h: float, w_cutoff: float):
        if w_cutoff == np.inf:
            self.a = 1.0
        else:
            self.a = h * w_cutoff / (1 + h * w_cutoff)

        self.n = dimension
        self.x = None

    def update(self, u: np.array):
        assert u.size == self.n

        if self.x is None:
            self.x = u
        else:
            self.x = (1 - self.a) * self.x + self.a * u

    def reset_state(self):
        self.x = None

    def has_valid_state(self):
        return self.x is not None

    def get_current_state(self):
        assert self.x is not None
        return self.x.copy()


class PoseFilter:
    def __init__(self, control_period=1000, translation_ratio=1, rotation_ratio=1):
        self.control_period = 1.0 / control_period
        self.w_cutoff_r = 1 * np.pi * control_period  # tune this smaller to get more smoothed behavior
        self.w_cutoff_t = 2 * np.pi * control_period * 2  # tune this smaller to get more smoothed behavior
        self.translation_ratio = translation_ratio
        self.rotation_ratio = rotation_ratio

        n = 3
        self.rotation_position_estimator = LowPassFilter(n, self.control_period, self.w_cutoff_r)
        self.rotation_velocity_estimator = LowPassFilter(n, self.control_period, self.w_cutoff_r)
        self.rotation_acceleration_estimator = LowPassFilter(n, self.control_period, self.w_cutoff_r)

        self.translation_position_estimator = LowPassFilter(n, self.control_period, self.w_cutoff_t)
        self.translation_velocity_estimator = LowPassFilter(n, self.control_period, self.w_cutoff_t)
        self.translation_acceleration_estimator = LowPassFilter(n, self.control_period, self.w_cutoff_t)

        self.translation_prev = np.zeros(n)
        self.translation_v_prev = np.zeros(n)
        self.translation_a_prev = np.zeros(n)

        self.rotation_prev = np.eye(n)
        self.rotation_vec_prev = np.zeros(n)
        self.rotation_v_prev = np.zeros(n)
        self.rotation_a_prev = np.zeros(n)
        self.curr_step = 0

    def update(self, pose):
        # self.saturate_pose(pose)
        self.update_translation(pose[:3, 3])
        self.update_rotation(pose[:3, :3])
        self.curr_step += 1
        # if self.curr_step % 100 == 0:
        #     self.rotation_position_estimator.reset_state()
        #     self.rotation_velocity_estimator.reset_state()
        #     self.rotation_acceleration_estimator.reset_state()

        #     self.translation_position_estimator.reset_state()
        #     self.translation_velocity_estimator.reset_state()
        #     self.translation_acceleration_estimator.reset_state()

    def update_rotation(self, rotation_mat):
        # filter based on the delta otherwise it has the wrapped around issues.
        if hasattr(self, "rotation_prev"):
            axis, angle = mat2axangle(self.rotation_prev.T @ rotation_mat)
            rotation = angle * axis * self.rotation_ratio

            # print("update rotation:", axis * angle)
            self.rotation_position_estimator.update(rotation)

            # the control period is hard-coded for now.
            v_diff = (rotation - self.rotation_vec_prev) / self.control_period
            a_diff = (v_diff - self.rotation_v_prev) / self.control_period

            self.rotation_velocity_estimator.update(v_diff)
            self.rotation_acceleration_estimator.update(a_diff)
            self.rotation_v_prev = v_diff
            self.rotation_a_prev = a_diff

        else:
            # first time
            rotation = np.zeros(3)
            self.rotation_position_estimator.update(rotation)
        self.rotation_prev = rotation_mat
        self.rotation_vec_prev = rotation

    def update_translation(self, translation):
        self.translation_position_estimator.update(translation * self.translation_ratio)
        if hasattr(self, "translation_prev"):
            v_diff = (translation - self.translation_prev) / self.control_period
            a_diff = (v_diff - self.translation_v_prev) / self.control_period

            self.translation_velocity_estimator.update(v_diff)
            self.translation_acceleration_estimator.update(a_diff)
            self.translation_v_prev = v_diff
            self.translation_a_prev = a_diff

        self.translation_prev = translation

    def get_estimates(self, saturate=False):
        p_est = self.translation_position_estimator.get_current_state()
        v_est = self.translation_velocity_estimator.get_current_state()
        a_est = self.translation_acceleration_estimator.get_current_state()

        if not hasattr(self, "rotation_est_prev"):
            # first step
            self.rotation_est_prev = self.rotation_prev

        # integration
        rp_est = self.rotation_position_estimator.get_current_state()
        angle = np.linalg.norm(rp_est)
        axis = rp_est / angle
        rot_mat = self.rotation_est_prev @ axangle2mat(axis, angle, is_normalized=True)
        self.rotation_est_prev = rot_mat

        if saturate:
            p_est, rot_mat = self.saturate_pose(p_est, rot_mat)

        rv_est = self.rotation_velocity_estimator.get_current_state()
        ra_est = self.rotation_acceleration_estimator.get_current_state()
        return p_est, v_est, a_est, rot_mat, rv_est, ra_est

    def saturate_pose(self, p_est, rot_mat):
        """
        cap the maximum rotations and translation offsets from the starting pose
        """
        translation_limit = 0.5
        rotation_limit = np.pi / 4
        p_est = np.clip(p_est, -translation_limit, translation_limit)
        axis, angle = mat2axangle(rot_mat)
        angle_scaled = np.clip(angle, -rotation_limit, rotation_limit)
        rot_mat = axangle2mat(axis, angle_scaled)

        return p_est, rot_mat


class VR_Interface:
    def __init__(self, freq=1000):
        self.vr_has_init = False
        self.rec_action_pose = None
        self.vr_frame_origin = None
        self.oculus_reader = None
        self.scale_translation_ratio = 0.8 if SCALE else 1.0
        self.scale_rotation_ratio = 0.3 if SCALE else 1.0

        self.pose_filter = PoseFilter(freq, self.scale_translation_ratio, self.scale_rotation_ratio)
        self.get_vr_init()

    def set_X_WG_Init(self, X_WGinit):
        self.X_WGinit = X_WGinit

    def get_vr_init(self):
        if not self.oculus_reader:
            try:
                self.oculus_reader = OculusReader()
            except:
                print("please install oculus reader first")
                pass

        print("VR Init, click rightTrig")
        while True:
            time.sleep(0.0001)
            transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
            if "rightTrig" not in buttons:
                continue

            # print("righttrig:", buttons["rightTrig"][0])
            if buttons["rightTrig"][0] > 0.8 or buttons["rightGrip"][0] > 0.5:  # trig
                self.vr_frame_origin = self.get_X_WU(transformations)
                self.vr_has_init = True
                break

        print("finish VR init")

    def get_X_WU(self, transformations):
        X_UhatU = RigidTransform(rotX(-np.pi / 2) @ rotZ(np.pi / 2))  # (rotY(-np.pi / 2) @ rotZ(np.pi/2))
        X_WU_hat = RigidTransform(transformations["r"])
        return X_WU_hat  # @ X_UhatU

    def get_curr_filtered_pose(self, limit=True):
        p_est, v_est, a_est, rot_mat, rv_est, ra_est = self.pose_filter.get_estimates()
        dX_GinitG_W_filtered = np.eye(4)

        dX_GinitG_W_filtered[:3, 3] = p_est
        dX_GinitG_W_filtered[:3, :3] = rot_mat if not NOROTATION else np.eye(3)

        velocity_estimates = np.concatenate((v_est, rv_est))
        acceleration_estimates = np.concatenate((a_est, ra_est))

        return dX_GinitG_W_filtered, velocity_estimates, acceleration_estimates

    def pose_postprocess(self, dX_UinitU_What):
        # 1. world frame velocity. push forward: x, push up: z, push left: y
        X_What_W = RigidTransform(rotZ(np.pi / 2) @ rotX(np.pi / 2))
        # 2. end effector frame. forward z. up x. and right y.
        X_U_Ustar = RigidTransform(rotZ(np.pi / 2))
        dX_UinitU_W = X_What_W @ dX_UinitU_What @ X_U_Ustar

        X_U_G = RigidTransform(rotZ(np.pi / 2)) @ RigidTransform(rotZ(np.pi / 2) @ rotX(np.pi / 2) @ rotZ(np.pi / 2))
        dX_GinitG_W = X_What_W @ dX_UinitU_What @ X_U_G
        self.pose_filter.update(dX_GinitG_W.GetAsMatrix4())
        dX_GinitG_W_filtered, V_GinitG_W_filtered, A_GinitG_W_filtered = self.get_curr_filtered_pose()

        if DEBUG:
            curr = pack_action(dX_GinitG_W.GetAsMatrix4())
            filtered = pack_action(dX_GinitG_W_filtered)
            res = curr  # np.concatenate((curr, filtered))
            update_line(tracking_online_logger, [curr[3], filtered[3]], rospy.get_time() - initial_time)
            # update_line(tracking_online_logger_filtered, pack_action(dX_GinitG_W_filtered), rospy.get_time() - initial_time)

        if FILTER:
            dX_GinitG_W = RigidTransform(dX_GinitG_W_filtered)

        X_WGdesired = apply_premultiply_pose_delta(RigidTransform(self.X_WGinit), dX_GinitG_W)
        return dX_UinitU_W.GetAsMatrix4(), X_WGdesired.GetAsMatrix4()

    def get_vr_input_inner(self):
        """Transform oculus motion to actual end effector motion"""
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()

        if "r" not in transformations:
            return None

        X_WUinit = self.vr_frame_origin
        X_WU = self.get_X_WU(transformations)
        dX_UinitU_W = compute_premultiply_pose_delta(X_WUinit, X_WU)
        dX_UinitU_W, X_WGdesired = self.pose_postprocess(dX_UinitU_W)

        exit_loop = "rightGrip" in buttons and buttons["rightGrip"][0] > 0.5
        reset = buttons["B"]
        finger_state = "close" if buttons["A"] else "open"
        teleop_hold = buttons["rightTrig"][0] < 0.2

        if (
            teleop_hold and not reset and not exit_loop
        ):  # clutching. move the vr to a more comfortable place to restart.
            print("================== clutching. ======================")
            self.set_X_WG_Init(X_WGdesired)
            self.get_vr_init()

        return (
            dX_UinitU_W,
            finger_state,  # Right Grip
            reset,  # B
            teleop_hold,  # Right Trig
            X_WGdesired,  # A
            exit_loop,
        )

    def get_vr_input():
        global vr_interface
        abs_pose = X_WGdesired_default
        (
            action_pose,
            finger_state,  # A
            reset,  # B
            teleop_hold,  # not pressing Right Trig
            X_WGdesired,
            exit_loop,  # Right Grip
        ) = vr_interface.get_vr_input_inner()
        return (X_WGdesired, teleop_hold, reset)


X_WGdesired_default = np.array(
    [
        [0.924, -0.006, 0.382, 0.480],
        [0.043, -0.992, -0.118, -0.003],
        [0.379, 0.126, -0.917, 0.53238],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
FILTER = True  # use filtered pose
DEBUG = False  # draw visualization would be slower
SCALE = True  # scale translation or not
init_joints = (0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785)
VR_MOVE = True  # use VR to control end effector
NOROTATION = False  # set rotation offset to be identity

try:
    from .oculus_reader import OculusReader

    vr_interface = VR_Interface()
    vr_interface.set_X_WG_Init(abs_pose)
except:
    pass
