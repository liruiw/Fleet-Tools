from core.utils import *
import json
import core.expert.kpam.SE3_utils as SE3_utils
from env.env_util import *

# The specification of optimization problem
import core.expert.kpam.term_spec as term_spec
import core.expert.kpam.mp_terms as mp_terms
from core.expert.kpam.optimization_problem import OptimizationProblemkPAM, solve_kpam
from core.expert.kpam.optimization_spec import OptimizationProblemSpecification

from colored import fg
import IPython


def dense_sample_traj_times(sample_times, task_completion_time, start_idx=1):
    """densify the waypoints for IK trajectory"""
    ik_times = list(sample_times)
    for i in range(start_idx, len(sample_times)):
        ratio = 1 if sample_times[i] < task_completion_time else 2
        N = int((sample_times[i] - sample_times[i - 1]) * ratio)  #
        for j in range(N):
            ik_times.insert(i, get_interp_time(sample_times[i - 1], sample_times[i], j / N))
    ik_times = sorted(np.unique(ik_times))
    return ik_times


def compose_circular_key_frames(
    center_pose,
    start_time=0,
    r=0.02,
    discrete_frame_num=5,
    total_degrees=np.pi * 2,
    total_time=3,
):
    """generating the circular motion for whisk for instance"""
    poses = []
    times = [start_time + (i + 1) / discrete_frame_num * total_time for i in range(discrete_frame_num)]
    for i in range(discrete_frame_num):
        pose = center_pose.copy()
        pose[:2, 3] += (rotZ(i / discrete_frame_num * total_degrees)[:3, :3] @ np.array([r, 0, 0]))[:2]
        poses.append(pose)
    return times, poses


def compose_rotate_y_key_frames(
    center_pose,
    start_time=0,
    r=0.02,
    discrete_frame_num=5,
    total_degrees=np.pi * 2,
    total_time=3,
):
    """generating the circular motion for whisk for instance"""
    poses = []
    times = [start_time + (i + 1) / discrete_frame_num * total_time for i in range(discrete_frame_num)]
    for i in range(discrete_frame_num):
        pose = center_pose.copy()
        pose[:2, 3] += (rotZ(i / discrete_frame_num * total_degrees)[:3, :3] @ np.array([r, 0, 0]))[:2]
        poses.append(pose)
    return times, poses


def compose_rotating_key_frames(
    ef_pose,
    center_point,
    start_time=0,
    discrete_frame_num=5,
    total_degrees=np.pi * 2,
    total_time=3,
):
    """generating the circular motion with facing center for wrench for instance"""
    poses = []
    times = [start_time + (i + 1) / discrete_frame_num * total_time for i in range(discrete_frame_num)]
    rel_pose = ef_pose.copy()
    rel_pose[:3, 3] -= center_point

    for i in range(discrete_frame_num):  # clockwise
        pose = rotZ(-(i + 1) / discrete_frame_num * total_degrees) @ rel_pose.copy()
        pose[:3, 3] += center_point
        poses.append(pose)
    return times, poses


class TaskExpert:
    # keypoint-based method for tool use
    def __init__(self, envs, args, cfg, robot_id=-1):
        self.actuation_time = 3.0

    def generate_spatula_poses(self, reach_ratio=0.3, reach_dist=0.2):
        """hard-coded post-activation trajectory:
        (1) reach to the pre-actuation
        (2) actuate it and lift it forward to flip the object
        """
        if self.env.task_config.expert_augment_goals:
            self.augment_goal_pose_and_select()

        # 1 cm lower
        self.task_goal_tool_pose[2, 3] -= 0.002
        tool_forward_dir = (self.task_goal_tool_pose[:3, :3] @ np.array([0, 0, 1]))[:2]
        tool_forward_dir = safe_div(tool_forward_dir, np.linalg.norm(tool_forward_dir))
        # self.task_goal_tool_pose[:2, 3] += tool_forward_dir * 0.05 # avoid reaching the middle point to maintain stick contact
        self.task_goal_tool_pose[:2, 3] += tool_forward_dir * 0.01
        # avoid reaching the middle point to maintain stick contact

        # define task poses
        self.task_goal_hand_pose = self.task_goal_tool_pose @ se3_inverse(self.env.tool_rel_pose)
        hand_forward_dir = tool_forward_dir  # (self.tool_forward_dir[:3, :3] @ np.array([0, 0, 1]))[:2]
        hand_forward_dir = safe_div(hand_forward_dir, np.linalg.norm(hand_forward_dir))
        trans_standoff1 = hand_forward_dir * -reach_dist  # / 2

        self.standoff_pose = self.task_goal_hand_pose.copy()
        self.standoff_pose[:2, 3] += trans_standoff1
        self.standoff_pose3 = self.standoff_pose.copy()
        self.standoff_pose3[2, 3] += 0.15

        # also add a bit of rotation
        self.lift_pose = self.task_goal_hand_pose.copy()
        self.lift_pose[:3, :3] = self.lift_pose[:3, :3] @ rotY(np.pi / 6)[:3, :3]
        self.lift_pose[2, 3] += 0.1  # 0.15
        # self.lift_pose[:2, 3] += hand_forward_dir * 0.12  #  * 3 # no need to flip
        self.lift_pose[:2, 3] -= hand_forward_dir * 0.02  # 13 # lift

        curr_time = self.env.time
        finish_time = self.cfg.task.task_completion_time
        standoff1_time = get_interp_time(curr_time, finish_time, 1 - reach_ratio)
        standoff3_time = get_interp_time(curr_time, finish_time, 1 - 2 * reach_ratio)
        self.actuate_times = finish_time + self.actuation_time
        self.sample_times = [
            curr_time,
            standoff1_time,
            finish_time,
            self.actuate_times,
        ]

        # standoff3_time, self.standoff_pose3,
        self.traj_keyframes = [
            self.env.ee_pose.reshape(4, 4),
            self.standoff_pose,
            self.task_goal_hand_pose,
            self.lift_pose,
        ]

    def generate_scoop_poses(self, reach_ratio=0.3, reach_dist=0.2):
        """hard-coded post-activation trajectory:
        (1) reach the center of the container
        (2) rotate the spoon while moving forward to scoop the ball
        """
        if self.env.task_config.expert_augment_goals:
            self.augment_goal_pose_and_select()

        tool_forward_dir = (self.task_goal_tool_pose[:3, :3] @ np.array([1, 0, 0]))[:2]
        tool_forward_dir = tool_forward_dir[:2] / np.linalg.norm(tool_forward_dir[:2])
        self.task_goal_tool_pose[:2, 3] += tool_forward_dir * 0.015
        self.task_goal_tool_pose[2, 3] += 0.01

        # define task poses
        self.task_goal_hand_pose = self.task_goal_tool_pose @ se3_inverse(self.env.tool_rel_pose)
        self.standoff_pose = self.task_goal_hand_pose.copy()  # in tool coordinate
        self.standoff_pose[2, 3] += reach_dist

        # scoop motion requires some efforts
        hand_forward_dir = (self.task_goal_hand_pose[:3, :3] @ np.array([0, -1, 0]))[:2]
        hand_forward_dir = hand_forward_dir[:2] / np.linalg.norm(hand_forward_dir[:2])
        self.lift_pose = self.task_goal_hand_pose.copy()  # in tool coordinate
        self.lift_pose[:3, :3] = self.lift_pose[:3, :3] @ rotX(np.pi / 3)[:3, :3]
        self.lift_pose[2, 3] += 0.07  # in the hand coordinate, so it rotates the tool a lot
        self.lift_pose[:2, 3] -= hand_forward_dir * 0.14

        # how to measure balls
        curr_time = self.env.time
        finish_time = self.cfg.task.task_completion_time
        standoff_time = get_interp_time(curr_time, finish_time, 1 - reach_ratio)
        self.actuate_times, self.actuate_poses = [finish_time + self.actuation_time], [self.lift_pose]
        self.sample_times = [curr_time, standoff_time, finish_time] + self.actuate_times
        self.traj_keyframes = [
            self.env.ee_pose.reshape(4, 4),
            self.standoff_pose,
            self.task_goal_hand_pose,
        ] + self.actuate_poses

    def generate_hammer_poses(self, reach_ratio=0.3, reach_dist=0.1):
        """hard-coded post-activation trajectory:
        (1) reach above the screw
        (2) hammer it by moving downward
        """
        if self.env.task_config.expert_augment_goals:
            self.augment_goal_pose_and_select()

        # define task poses
        self.task_goal_hand_pose = self.task_goal_tool_pose @ se3_inverse(self.env.tool_rel_pose)
        self.standoff_pose = self.task_goal_hand_pose.copy()  # in tool coordinate
        self.standoff_pose[2, 3] += reach_dist / 2

        curr_time = self.env.time
        finish_time = self.cfg.task.task_completion_time
        standoff_time = get_interp_time(curr_time, finish_time, 1 - reach_ratio)

        # maybe add a rotation in the tool space for both standoff and the task goal pose
        self.hammer_pose = self.task_goal_hand_pose.copy()
        self.hammer_pose[2, 3] -= 0.05

        self.task_goal_hand_pose_rotated = self.task_goal_hand_pose.copy()
        self.lift_pose = self.task_goal_hand_pose.copy()
        self.lift_pose[2, 3] += reach_dist

        self.actuate_times = [
            finish_time + self.actuation_time * 0.6,
            finish_time + self.actuation_time,
        ]
        self.actuate_poses = [self.hammer_pose, self.lift_pose]
        self.sample_times = [curr_time, standoff_time, finish_time] + self.actuate_times
        self.traj_keyframes = [
            self.env.ee_pose.reshape(4, 4),
            self.standoff_pose,
            self.task_goal_hand_pose,
        ] + self.actuate_poses

    def generate_whisk_poses(self, reach_ratio=0.3, reach_dist=0.2):
        """hard-coded post-activation trajectory:
        (1) reach the container with whisk
        (2) whisk around the container
        """
        if self.env.task_config.expert_augment_goals:
            self.augment_goal_pose_and_select()

        # define task poses
        self.task_goal_hand_pose = self.task_goal_tool_pose @ se3_inverse(self.env.tool_rel_pose)
        self.standoff_pose = self.task_goal_hand_pose.copy()  # in tool coordinate
        self.standoff_pose[2, 3] += reach_dist

        curr_time = self.env.time
        finish_time = self.cfg.task.task_completion_time
        standoff_time = get_interp_time(curr_time, finish_time, 1 - reach_ratio)

        self.actuate_times, self.actuate_poses = compose_circular_key_frames(self.task_goal_hand_pose, finish_time)
        self.sample_times = [curr_time, standoff_time, finish_time] + self.actuate_times
        self.traj_keyframes = [
            self.env.ee_pose.reshape(4, 4),
            self.standoff_pose,
            self.task_goal_hand_pose,
        ] + self.actuate_poses

    def generate_knife_poses(self, reach_ratio=0.3, reach_dist=0.2):
        """hard-coded post-activation trajectory:
        (1) reach the middle of the two objects
        (2) push toward one of them with the knife side to separate
        """

        self.task_goal_tool_pose[2, 3] -= 0.005
        self.task_goal_hand_pose = self.task_goal_tool_pose @ se3_inverse(self.env.tool_rel_pose)
        self.standoff_pose = self.task_goal_hand_pose.copy()  # in tool coordinate
        self.standoff_pose[2, 3] += reach_dist / 2

        curr_time = self.env.time
        finish_time = self.cfg.task.task_completion_time
        standoff_time = get_interp_time(curr_time, finish_time, 1 - reach_ratio)

        # need two points on both objects. push from the current tool tip point towards one of the object
        tool_forward_dir1 = (self.env.obj_pose1[:3, 3] - self.curr_solution_tool_keypoint_head)[:2]
        tool_forward_dir1 = tool_forward_dir1[:2] / np.linalg.norm(tool_forward_dir1[:2])

        tool_forward_dir2 = (self.env.obj_pose2[:3, 3] - self.curr_solution_tool_keypoint_head)[:2]
        tool_forward_dir2 = tool_forward_dir2[:2] / np.linalg.norm(tool_forward_dir2[:2])

        # choose based on which one's x direction is more negative
        tool_forward_dir = tool_forward_dir2 if tool_forward_dir2[0] < tool_forward_dir1[0] else tool_forward_dir1
        self.lift_pose = self.task_goal_hand_pose.copy()
        self.lift_pose[:2, 3] -= tool_forward_dir * 0.2

        self.actuate_times = [finish_time + self.actuation_time]
        self.actuate_poses = [self.lift_pose]
        self.sample_times = [curr_time, standoff_time, finish_time] + self.actuate_times
        self.traj_keyframes = [
            self.env.ee_pose.reshape(4, 4),
            self.standoff_pose,
            self.task_goal_hand_pose,
        ] + self.actuate_poses

    def generate_mug_poses(self, reach_ratio=0.3, reach_dist=0.2):
        """hard-coded post-activation trajectory:
        (1) reach right to the screw
        (2) rotate the wrench around the screw
        """
        # has to be the tool keypoint axis
        tool_forward_dir = (self.curr_solution_tool_keypoint_head - self.curr_solution_tool_keypoint_tail)[:2]

        # tool_forward_dir = (self.task_goal_tool_pose[:3, :3] @ np.array([0, 0, 1]))[:2]
        tool_forward_dir = safe_div(tool_forward_dir, np.linalg.norm(tool_forward_dir))

        self.task_goal_tool_pose[:2, 3] += tool_forward_dir * 0.04
        self.task_goal_hand_pose = self.task_goal_tool_pose @ se3_inverse(self.env.tool_rel_pose)
        self.task_goal_hand_pose[2, 3] += 0.04
        self.standoff_pose = self.task_goal_hand_pose.copy()  # in tool coordinate
        self.standoff_pose[2, 3] += 0.03  # a bit higher

        curr_time = self.env.time
        finish_time = self.cfg.task.task_completion_time
        standoff_time = get_interp_time(curr_time, finish_time, 1 - reach_ratio)  # self.task_goal_tool_pose[:3, 3]
        # then rotate around Z axis local

        self.actuate_times = [self.actuation_time + finish_time]
        rotated_pose = self.task_goal_hand_pose.copy()
        rotated_pose[:3, :3] = rotated_pose[:3, :3] @ rotY(-np.pi / 6)[:3, :3]
        self.actuate_poses = [rotated_pose]  #

        self.sample_times = [curr_time, standoff_time, finish_time] + self.actuate_times
        self.traj_keyframes = [
            self.env.ee_pose.reshape(4, 4),
            self.standoff_pose,
            self.task_goal_hand_pose,
        ] + self.actuate_poses

    def generate_wrench_poses(self, reach_ratio=0.3, reach_dist=0.2):
        """hard-coded post-activation trajectory:
        (1) reach right to the screw
        (2) rotate the wrench around the screw
        """

        # has to be the tool keypoint axis
        tool_forward_dir = (self.curr_solution_tool_keypoint_head - self.curr_solution_tool_keypoint_tail)[:2]
        tool_forward_dir = safe_div(tool_forward_dir, np.linalg.norm(tool_forward_dir))
        self.task_goal_hand_pose = self.task_goal_tool_pose @ se3_inverse(self.env.tool_rel_pose)
        self.standoff_pose = self.task_goal_hand_pose.copy()  # in tool coordinate
        hand_forward_dir = (self.task_goal_hand_pose[:3, :3] @ np.array([0, 0, 1]))[:2]
        hand_forward_dir = safe_div(hand_forward_dir, np.linalg.norm(hand_forward_dir))
        self.standoff_pose[:2, 3] -= tool_forward_dir * 0.08  #

        curr_time = self.env.time
        finish_time = self.cfg.task.task_completion_time
        standoff_time = get_interp_time(curr_time, finish_time, 1 - reach_ratio)  # self.task_goal_tool_pose[:3, 3]
        self.actuate_times, self.actuate_poses = compose_rotating_key_frames(
            self.task_goal_hand_pose,
            self.curr_solution_tool_keypoint_head,
            total_degrees=np.pi / 2,
            total_time=self.actuation_time * 2,
            start_time=finish_time,
        )
        self.lift_pose = self.actuate_poses[-1]
        self.sample_times = [curr_time, standoff_time, finish_time] + self.actuate_times
        self.traj_keyframes = [
            self.env.ee_pose.reshape(4, 4),
            self.standoff_pose,
            self.task_goal_hand_pose,
        ] + self.actuate_poses
