import gym
import numpy as np

import warnings
from gym.utils import seeding
from pydrake.all import Simulator, SimulatorStatus
from pydrake.math import RollPitchYaw, RigidTransform

try:
    import open3d as o3d
except:
    pass

from pydrake.common import RandomGenerator
from env.env_util import *
import IPython
from core.utils import *
from colored import fg
import matplotlib.pyplot as plt
from env.basedrakeenv import BaseDrakeRobotEnv
import torch


class FrankaDrakeEnv(BaseDrakeRobotEnv):
    def __init__(self, task_config):
        self.reset_from_scene_file = False
        self.init(task_config)

    def init(self, task_config, info={}):
        self.task_config = task_config
        self.sim_dt = self.task_config.sim.dt
        self.at_contact = False
        self.fix_pose_controller = False
        self.target_obj_idx = 0

        if hasattr(self, "tool_class_name"):
            (
                builder,
                logger,
                self.system,
                self.plant,
                self.robot_controller,
                self.controller_plant,
                self.controller_plant_system,
                self.differential_ik,
                self.scene_graph,
                self.scene_inspector,
                tool_obj_info,
                tool_info,
            ) = MakeManipulationStationTools(
                time_step=self.sim_dt,
                tool_class_name=self.tool_class_name,
                task_config=self.task_config,
                instance_info=info,
            )
        else:
            (
                builder,
                logger,
                self.system,
                self.plant,
                self.robot_controller,
                self.controller_plant,
                self.controller_plant_system,
                self.differential_ik,
                self.scene_graph,
                self.scene_inspector,
            ) = MakeManipulationStation(time_step=self.sim_dt, task_config=self.task_config)

        self.simulator = Simulator(self.system)
        self.env_dt = self.sim_dt * self.task_config.sim.substep_num
        self.diffik_port = self.system.GetInputPort("panda_hand_pose")
        self.diffik_output_port = self.system.GetOutputPort("differential_ik_output")
        if self.task_config.use_controller_portswitch:
            self.controller_switch_port = self.system.GetInputPort("controller_switch_port")

        if self.task_config.use_controller_portswitch:
            self.action_port = self.system.GetInputPort("desired_eff_pose")
            self.action_vel_port = self.system.GetInputPort("desired_eff_velocity")
            self.joint_action_port = self.system.GetInputPort("panda_joint_commanded")

        elif not self.task_config.pose_controller:
            self.joint_action_port = self.system.GetInputPort("panda_joint_commanded")

        else:
            self.action_port = self.system.GetInputPort("desired_eff_pose")
            self.action_vel_port = self.system.GetInputPort("desired_eff_velocity")

        self.torque_input_port = self.system.GetInputPort("feedforward_torque")
        self.joint_measurement_port = self.system.GetOutputPort("panda_position_measured")
        self.contact_measurement_port = self.system.GetOutputPort("contact_results")
        self.external_torque_port = self.system.GetOutputPort("panda_torque_external")
        self.object_query_port = self.system.GetOutputPort("query_object")

        # after gravity compensated
        self.joint_min_limit = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0, 0])
        self.joint_max_limit = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
        self.joint_null_space = (self.joint_min_limit + self.joint_max_limit) / 2

        # initialize info
        self.info = {k: np.zeros(1) for k in key_list}
        self.action_num = self.task_config.env.numActions
        self.observation_num = self.task_config.env.numObservations
        self.action_worldframe = self.task_config.env.action_worldframe
        self.generator = RandomGenerator()
        self.seed(0)

        self.history_wrench_obs = deque(
            [],
            maxlen=self.task_config.env.history_force_buffer_length,
        )
        self.table_height = 0.15
        self.table_init_height = 0.3
        self.finger_state = [0, 0]
        self.time_step = 0
        self.need_termination = False
        self.goal_pose = None
        self.time = 0

        if hasattr(self, "tool_class_name"):
            self.setup(tool_obj_info, tool_info)

    def use_pose_controller(self):
        contact_and_hybrid = self.task_config.use_controller_portswitch and self.at_contact
        return (self.task_config.pose_controller) or contact_and_hybrid

    def use_joint_controller(self):
        nocontact_and_hybrid = self.task_config.use_controller_portswitch and not self.at_contact
        return (not self.task_config.pose_controller or nocontact_and_hybrid) and hasattr(self, "joint_action_port")

    @property
    def action_space(self):
        """action space of the simulation environment"""
        return PandaTaskSpace6D(*self.task_config.env.action_scale)

    @property
    def observation_space(self):
        """state space of the simulation environment"""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=self.observation_num)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.drake_random = RandomGenerator(seed)
        return [seed]

    def advance_time(self, context):
        """
        simulate forward
        """
        self.time = context.get_time()
        catch = False
        s = time.time()
        repeat_num = self.task_config.sim.substep_num

        try:
            for _ in range(repeat_num):  # make the simulation smooth and visulizable
                self.time = context.get_time()
                status = self.simulator.AdvanceTo(self.time + self.sim_dt)
                self._get_robot_freq_data()  # 0.02s slower per step

        except RuntimeError as e:
            warnings.warn("Calling Done after catching RuntimeError:")
            warnings.warn(e.args[0])
            catch = True
            self.info["sim_failure"] = True  # emergent reset
            color_print("simulation failed, reset!", "light_slate_blue")

        sim_done = catch or status.reason() == SimulatorStatus.ReturnReason.kReachedTerminationCondition
        return sim_done

    def process_action(self, action):
        """process the action for the actuation ports
        if the action is a concatenation of joint and pose, then it would use joint unless
        the environment has been configured to use pose.
        """
        self.action = action
        use_expert_traj_joints = len(action) == 15

        if use_expert_traj_joints:
            # use the expert joint trajectories
            target_joint_positions = action[:9]
            action = action[9:]

            # compute delta action with forward kinematics
            target_pose = self.set_diffik_state(self.context, set_joints=target_joint_positions).GetAsMatrix4()
            rel_pose = se3_inverse(self.ee_pose.reshape(4, 4)).dot(target_pose)
            self.action = pack_pose(rel_pose)
            self.target_joint = target_joint_positions

        action_delta_pose = unpack_pose(action, rot_type="euler")
        cur_pose = self.get_ee_pos().GetAsMatrix4()

        self.target_pose = cur_pose.dot(action_delta_pose)
        self.commanded_pose = self.target_pose.copy()

        # use OSC if (1) configured to use pose controller (2) hybrid and has contact
        if self.use_pose_controller():  # and not use_expert_traj_joints
            self.action_port.FixValue(self.context, RigidTransform(self.target_pose))
            self.target_pose_vel = get_frame_spatial_velocity(cur_pose, self.target_pose, self.env_dt)
            self.action_vel_port.FixValue(self.context, SpatialVelocity(self.target_pose_vel))

        # (1) expert demonstration (2) hybrid and no-contact
        elif self.use_joint_controller() or use_expert_traj_joints:
            if not use_expert_traj_joints:
                res = self.solve_ik(self.target_pose, self.joint_positions)
                if res is not None:
                    next_joint = res.get_x_val()[:9]

                else:
                    # use differential IK
                    self.set_diffik_state(self.context, set_joints=self.joint_positions)
                    self.diffik_port.FixValue(self.context, RigidTransform(self.target_pose))
                    next_joint = self.diffik_output_port.Eval(self.context)
                target_joint_positions = next_joint

            target_joint_positions[-2:] = 0.0  # no need to control the finger
            if hasattr(self, "joint_action_port"):
                self.joint_action_port.FixValue(self.context, target_joint_positions)

        return action

    def forward_kinematics(self, joint):
        return self.set_diffik_state(self.context, set_joints=joint).GetAsMatrix4()

    def set_diffik_state(self, context, reinit=False, set_joints=None):
        """set the differential ik. Could also be used as forward kinematics function"""
        diff_ik_context = self.differential_ik.GetMyMutableContextFromRoot(context)
        if set_joints is None:
            set_joints = self.joint_measurement_port.Eval(context)  # use measured joint
        try:
            self.differential_ik.SetPositions(diff_ik_context, set_joints)
        except:
            print("set joint failed")
        return self.differential_ik.ForwardKinematics(diff_ik_context)

    def step(self, action):
        """environment step"""
        self.time_step += 1
        self.context = self.simulator.get_context()

        sim_done = self.advance_time(self.context)
        if sim_done:  # terminate because of simulation error
            return [], 0, True, self.info

        self._get_obs(self.context)
        obs = self.select_obs()
        reward = self._get_reward(self.context, action)
        done = sim_done

        return obs, reward, done, self.info

    def select_obs(self):
        """select different observation space"""
        if self.task_config.env.use_image:
            return self.img_obs
        elif self.task_config.env.use_keypoint:
            return self.info["curr_keypoint"].reshape(-1)
        else:
            return self.state_obs

    def init_context(self):
        """higher-level scene and robot reset function for the environment"""
        self.need_termination = False
        self.at_contact = False

        self.time_step = 0
        self.simulator = Simulator(self.system)
        self.context = self.simulator.get_context()
        self.plant_context = self.system.GetSubsystemContext(self.plant, self.context)
        self.get_joint_angles()

        if hasattr(self, "tool_name"):
            if not self.reset_from_scene_file:
                self.reset_object(self.context)
                self.reset_robot(self.context)
            else:
                # object pose
                if self.task_config.verbose:
                    print(
                        "reset from scene", self.scene_descriptions["tool_name"], self.scene_descriptions["object_name"]
                    )
                try:
                    if "object_pose" in self.scene_descriptions:
                        pose = np.array(self.scene_descriptions["object_pose"][0])
                        pose[2, 3] += 0.03
                        tf = RigidTransform(pose)
                        self.set_object_pose(tf)
                    else:
                        self.reset_object(self.context)

                    self.joint_positions = np.array(self.scene_descriptions["robot_joints"][0])
                    self.plant.SetPositions(
                        self.plant_context,
                        self.plant.GetModelInstanceByName("panda"),
                        self.joint_positions.reshape(-1, 1),
                    )  #
                    self.plant.SetVelocities(
                        self.plant_context,
                        self.plant.GetModelInstanceByName("panda"),
                        [0] * 9,
                    )
                    self.set_diffik_state(self.context)
                except:
                    pass

        self.initial_ee_pose = self.get_ee_pos().GetAsMatrix4()
        self.context.SetTime(0)

        if self.use_pose_controller():
            self.action_port.FixValue(self.context, RigidTransform(self.initial_ee_pose))
            self.target_pose_vel = SpatialVelocity(np.zeros(6))
            self.action_vel_port.FixValue(self.context, self.target_pose_vel)

        if self.use_joint_controller():
            self.joint_action_port.FixValue(self.context, self.joint_positions)

        if self.task_config.use_controller_portswitch:
            self.controller_switch_port.FixValue(self.context, InputPortIndex(2))

        try:
            if self.task_config.blender_render:
                self.simulator.set_publish_every_time_step(False)
            else:
                self.simulator.Initialize()
        except:
            print("init context exception")
            res = self.task_config.env.init_joints
            self.plant.SetPositions(self.plant_context, self.plant.GetModelInstanceByName("panda"), res[:9])  #
            self.plant.SetVelocities(self.plant_context, self.plant.GetModelInstanceByName("panda"), [0] * 9)
            self.set_diffik_state(self.context, reinit=True, set_joints=res[:9])
            self.joint_positions = res[:9]
            self.context.SetTime(0)
            self.simulator.Initialize()

        ee_pose = self.get_ee_pos()
        self.simulator.set_target_realtime_rate(self.task_config.sim.realtime_rate)  #

        self.get_joint_angles()
        self.target_pose = np.array(ee_pose.GetAsMatrix4())
        self.ee_pose = np.array(ee_pose.GetAsMatrix4()).copy().flatten()
        self.commanded_pose = np.array(ee_pose.GetAsMatrix4())  # prev target pose

        self.set_diffik_state(self.context)
        self.diffik_port.FixValue(self.context, RigidTransform(ee_pose))
        self.target_joint_positions = self.joint_positions

        # reset some states
        self.fused_tool_points = torch.zeros([4, 0])  #
        self.fused_object_points = torch.zeros([4, 0])
        self.history_wrench_obs = deque([], maxlen=self.task_config.env.history_force_buffer_length)

        self.goal_ee_pose = self.ee_pose  # used for demonstration
        self.finger_state = [0, 0]
        self.endeffector_wrench = np.zeros(6)

        self.time_step = 0
        self.controller_context = self.controller_plant.CreateDefaultContext()
        self.goal_ik = InverseKinematics(self.controller_plant, self.controller_context)
        self.at_contact = False
        self.empty_pointcloud_step = False

    def reset(self, done=False):
        """
        Resets the environment to its initial state. 
        """
        self.init_context()
        self.object_pose = np.eye(4)
        self.tool_pose = np.eye(4)
        self.tool_rel_pose = np.eye(4)

        for _ in range(self.task_config.sim.warmstart_iter):
            sim_done = self.advance_time(self.context)
            if sim_done:
                return [], 0, True, self.info

        self.action = np.zeros(self.action_num)
        self._get_obs(self.context)
        obs = self.select_obs()
        reward = self._get_reward(self.context, self.action)
        return obs, reward, done, self.info

    def compute_state_obs(self):
        """
        Computes the state observation for the environment.

        Returns:
            numpy.ndarray: The concatenated state observation array.
        """
        return np.concatenate(
            [
                pack_pose(self.ee_pose.reshape(4, 4)),
                self.joint_positions,
                pack_pose(self.object_pose),
                pack_pose(self.tool_pose),
            ]
        )

    def reset_from_scene(self, scene_file=None):
        """load a scene including object poses and meshes from a saved file"""
        if scene_file is not None:
            scene_descriptions = json.load(open(os.path.join(scene_file, "scene_descriptions.json")))
            if "object_name" not in scene_descriptions:
                print("filling in real world object details")
                scene_descriptions["object_name"] = scene_descriptions["tool_name"] + "_model0_body_link"
                scene_descriptions["tool_name"] = scene_descriptions["tool_name"] + "_model0_body_link"

            self.scene_descriptions = scene_descriptions
            if self.task_config.verbose:
                print("init from scene file:", scene_file)
        self.init(self.task_config, self.scene_descriptions)

    def get_reward(self, *args):
        return self._get_reward(*args)

    def _get_reward(self, context, action):
        return 0

    def _get_progress(self, context):
        # determine which stage is at for the expert
        return None

    def get_keypoint_information(self):
        """get the keypoint observation"""
        self.tool_keypoints = self.tool_info["keypoints"]
        self.tool_obj_keypoints = self.tool_obj_info["keypoints"]
        self.curr_tool_keypoints = self.tool_keypoints.copy()
        self.curr_tool_obj_keypoints = self.tool_obj_keypoints.copy()
        self.tool_keypoint_head = self.tool_keypoints[0]
        self.tool_keypoint_tail = self.tool_keypoints[1]
        self.tool_keypoint_side = self.tool_keypoints[2]

    def transform_keypoints_env(self):
        """
        Transforms the keypoints of the tool and object based on their respective poses.

        This method applies the transformation matrix of the tool and object poses to the keypoints
        of the tool and calculates the new positions of the keypoints.
        """
        self.curr_tool_keypoint_head = self.tool_pose[:3, :3].dot(self.tool_keypoint_head) + self.tool_pose[:3, 3]
        self.curr_tool_keypoint_tail = self.tool_pose[:3, :3].dot(self.tool_keypoint_tail) + self.tool_pose[:3, 3]
        self.curr_tool_keypoint_side = self.tool_pose[:3, :3].dot(self.tool_keypoint_side) + self.tool_pose[:3, 3]
        self.curr_tool_obj_keypoints = (
            self.object_pose[:3, :3].dot(self.tool_obj_keypoints[0]) + self.object_pose[:3, 3]
        )

    def update_keypoint_information(self):
        """get two keypoint positions and the direction between these two points"""
        self.transform_keypoints_env()
        if len(self.tool_obj_keypoints) > 1:
            self.curr_tool_obj_keypoint_rest = (
                self.object_pose[:3, :3].dot(self.tool_obj_keypoints[1:].T).T + self.object_pose[:3, 3]
            )

        # compute the keypoint wrench in the object keypoint coordinate
        self.tool_keypoint_head_in_hand = (
            self.tool_rel_pose[:3, :3].dot(self.tool_keypoint_head) + self.tool_rel_pose[:3, 3]
        )
        self.tool_keypoint_tail_in_hand = (
            self.tool_rel_pose[:3, :3].dot(self.tool_keypoint_tail) + self.tool_rel_pose[:3, 3]
        )
        self.tool_keypoint_side_in_hand = (
            self.tool_rel_pose[:3, :3].dot(self.tool_keypoint_side) + self.tool_rel_pose[:3, 3]
        )

        # get the jacobian that measures speed in the world frame, and expressed in the world frame
        self.head_keypoint_joint_jacobian = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            self.plant.GetBodyByName("panda_hand").body_frame(),
            self.tool_keypoint_head_in_hand,
            self.plant.world_frame(),
            self.plant.world_frame(),
        )[:, :7]
        self.tail_keypoint_joint_jacobian = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            self.plant.GetBodyByName("panda_hand").body_frame(),
            self.tool_keypoint_tail_in_hand,
            self.plant.world_frame(),
            self.plant.world_frame(),
        )[:, :7]
        self.side_keypoint_joint_jacobian = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            self.plant.GetBodyByName("panda_hand").body_frame(),
            self.tool_keypoint_side_in_hand,
            self.plant.world_frame(),
            self.plant.world_frame(),
        )[:, :7]

        # visualize the force and torque at the keypoint locations. Maybe the wrench is more useful in the local frames.
        # rotation first and then position
        torque_sensing = self.external_torque[:7]
        self.head_keypoint_wrench = np.linalg.pinv(self.head_keypoint_joint_jacobian.T).dot(torque_sensing)
        self.tail_keypoint_wrench = np.linalg.pinv(self.tail_keypoint_joint_jacobian.T).dot(torque_sensing)
        self.side_keypoint_wrench = np.linalg.pinv(self.side_keypoint_joint_jacobian.T).dot(torque_sensing)

        keypoint_loc = np.stack(
            (
                self.curr_tool_keypoint_head,
                self.curr_tool_keypoint_tail,
                self.curr_tool_keypoint_side,
                self.curr_tool_obj_keypoints,
            ),
            axis=0,
        )
        self.info["curr_keypoint"] = keypoint_loc
        self.info["keypoint_wrench"] = np.stack(
            (
                self.head_keypoint_wrench,
                self.tail_keypoint_wrench,
                self.side_keypoint_wrench,
            ),
            axis=0,
        )

        if self.task_config.use_meshcat:
            from env.env_util import meshcat

            self.visualize_task_keypoints(
                [
                    self.curr_tool_keypoint_head,
                    self.curr_tool_keypoint_tail,
                    self.curr_tool_keypoint_side,
                    self.curr_tool_obj_keypoints,
                ],
                "task_points",
            )
            # visualize force and torque
            AddWrenchVis(
                meshcat,
                "keypoint_head",
                self.curr_tool_keypoint_head,
                self.head_keypoint_wrench,
            )
            if not self.task_config.vis_collision_geometry:
                meshcat.SetProperty("collision", "visible", False)

    def _get_robot_freq_data(self):
        """get the higher-freq robot state"""
        s = time.time()
        # expressed in the world frame
        self.joint_jacobian = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            self.plant.GetBodyByName("panda_hand").body_frame(),
            [0, 0, 0],
            self.plant.world_frame(),
            self.plant.GetBodyByName("panda_hand").body_frame(),
        )[:, :7]

        try:
            self.external_torque = self.external_torque_port.Eval(self.context)
            torque_sensing = self.external_torque[:7]
            self.get_joint_angles()
        except Exception as e:
            print(e)
            print("history wrench failed")
            torque_sensing = np.zeros(7)

        try:
            if not hasattr(self, "endeffector_wrench"):
                self.endeffector_wrench = np.zeros(6)

            # current
            prev_endeffector_wrench = self.endeffector_wrench.copy()
            self.endeffector_wrench = np.linalg.pinv(self.joint_jacobian.T).dot(torque_sensing)
            self.endeffector_wrench_diff = self.endeffector_wrench - prev_endeffector_wrench
            self.endeffector_wrench_diff = self.endeffector_wrench_diff[:3]
            self.endeffector_wrench_diff_norm = np.linalg.norm(self.endeffector_wrench_diff)
            if self.endeffector_wrench_diff_norm > 1e-4:
                self.endeffector_wrench_diff = self.endeffector_wrench_diff / np.linalg.norm(
                    self.endeffector_wrench_diff
                )
            else:
                self.endeffector_wrench_diff = np.zeros(3)

            self.history_wrench_obs.append(self.endeffector_wrench)
        except Exception as e:
            print(e)
            print("history wrench failed")

    def compute_pointcloud_observation(self, obs, separate_mask, vis=False):
        """compute the torque at each point on the tool
        farthest point sampling to downsample to 1024 points point fusion of two views
        """
        s = time.time()
        pc_img_transforms = []

        # fuse all point cloud from two cameras in world coordinate, partial view.
        pose_names = ["base_camera_pose", "hand_camera_pose"]
        drop_camera = (
            hasattr(self.task_config, "randomize_drop_camera")
            and self.task_config.randomize_drop_camera
            and np.random.uniform() < 0.05
            and self.task_config.train
        )
        drop_cam_index = int(np.random.uniform() < 0.5)
        for idx, (obs_i, mask_i, cam_name) in enumerate(zip(obs, separate_mask, pose_names)):

            depth_image = obs_i[..., 3]
            mask_image = mask_i.flatten().reshape(1, -1)
            K = CAM_INTR
            extrinsics = self.info[cam_name].reshape(4, 4)

            pc_img = backproject_camera_target(depth_image, K)
            pc_img_world = transform_point(extrinsics, pc_img)

            pc_img_world = np.concatenate((pc_img_world, mask_image), axis=0)

            if drop_camera and drop_cam_index == idx:
                print("drop camera idx:", idx)
                continue

            pc_img_transforms.append(pc_img_world)

        # fuse pointclouds
        pc_img_transforms = np.concatenate(pc_img_transforms, axis=1)

        # world coordinate and use mask
        object_points = pc_img_transforms[:, pc_img_transforms[-1] == 1]
        tool_points = pc_img_transforms[:, pc_img_transforms[-1] == 2]
        tool_pc = torch.from_numpy(tool_points.T).cuda().float()[None]

        obj_pc = torch.from_numpy(object_points.T).cuda().float()[None]
        ee_pose = torch.from_numpy(self.ee_pose.reshape(4, 4)).cuda().float()
        ee_inv_pose = torch.from_numpy(se3_inverse(self.ee_pose.reshape(4, 4))).cuda().float()

        # tool points fused in local frame
        self.fused_tool_points = self.fused_tool_points.cuda()
        self.fused_object_points = self.fused_object_points.cuda()
        self.empty_pointcloud_step = False

        if tool_points.shape[1] != 0:
            tool_xyz = downsample_point(tool_pc, self.task_config.env.num_points // 2).cuda()
            # if any of the pointcloud is empty. Need to return or gives some zero paddings
            # if there is a contact, we need to reset the pointcloud
            if self.external_torque[:7].sum() > 0:
                self.fused_tool_points = torch.zeros([4, 0]).cuda()  #

            tool_xyz[-1] = 0  # mask 0 for tool
            tool_xyz[:3] = transform_point(ee_inv_pose, tool_xyz[:3])
            self.fused_tool_points = tool_xyz
            if hasattr(self.task_config, "use_fused_pointcloud") and self.task_config.use_fused_pointcloud:
                self.fused_tool_points = torch.cat((self.fused_tool_points, tool_xyz), axis=1)
            tool_points = tool_xyz

        else:  # fused points could be used as backup
            tool_points = self.fused_tool_points
            print("empty tool points at the current timestep")
            if self.time < 10:
                self.empty_pointcloud_step = True

        # object points fused in world frame
        if object_points.shape[1] != 0:
            obj_xyz = downsample_point(obj_pc, self.task_config.env.num_points // 2).cuda()

            if self.external_torque[:7].sum() > 0:  # reset points when there is contact
                self.fused_object_points = torch.zeros([4, 0]).cuda()

            obj_xyz[-1] = 1  # mask 1 for obj
            self.fused_object_points = obj_xyz
            if hasattr(self.task_config, "use_fused_pointcloud") and self.task_config.use_fused_pointcloud:
                self.fused_object_points = torch.cat((self.fused_object_points.cuda(), obj_xyz.cuda()), axis=1)
            object_points = obj_xyz

        else:  # fused points could be used as backup
            object_points = self.fused_object_points
            print("empty object points at the current timestep")
            if self.time < 10:
                self.empty_pointcloud_step = True

        if hasattr(self.task_config, "use_fused_pointcloud") and self.task_config.use_fused_pointcloud:
            self.fused_tool_points = downsample_point(
                self.fused_tool_points.T, self.task_config.env.fused_num_points // 2
            )
            self.fused_object_points = downsample_point(
                self.fused_object_points.T, self.task_config.env.fused_num_points // 2
            )

        curr_frame_tool_xyz = downsample_point(tool_points.T, self.task_config.env.num_points // 2).cuda()
        curr_frame_obj_xyz = downsample_point(object_points.T, self.task_config.env.num_points // 2).cuda()

        # object point transform from world frame to the ee frame
        curr_frame_obj_xyz[:3] = transform_point(ee_inv_pose, curr_frame_obj_xyz[:3])
        combined_pc = torch.cat((curr_frame_tool_xyz, curr_frame_obj_xyz), axis=-1)
        combined_pc = combined_pc.detach().cpu().numpy()

        # convert to hand coordinate frame
        self.info["point_cloud"] = combined_pc

        if vis and self.time_step % 5 == 0 and self.task_config.use_meshcat:
            from env.env_util import meshcat, visualizer

            pcd = o3d.geometry.PointCloud()
            point_xyzs = transform_point(self.ee_pose.reshape(4, 4), combined_pc[:3]).T
            mask = self.info["point_cloud"][-1]
            point_colors = np.zeros_like(point_xyzs)

            point_colors[mask == 1] = [1.0, 0, 0]
            point_colors[mask == 0] = [0, 1.0, 0]
            pcd.points = o3d.utility.Vector3dVector(point_xyzs)
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
            draw_open3d_point_cloud(meshcat, "observed_pc", pcd, point_size=0.001)

    def _get_obs(self, context, render=True):
        """compute the observation of the environment"""
        self.get_joint_angles()
        self.ee_pose = np.array(self.get_ee_pos().GetAsMatrix4()).copy().flatten()
        self.info["action"] = self.action  # the action that brings to this observation
        self.img_obs = []
        self.full_mask_data = []
        render = self.task_config.env.use_image or self.task_config.env.use_point_cloud_model

        # get camera poses
        if render:
            cams = self.task_config.env.cam.split("+")
            if len(cams) == 1:
                self.img_obs = self._render(cams[0])
                camera_pose_port = self.system.GetOutputPort(cams[0] + "_camera_pose")
                cam_pose = camera_pose_port.Eval(context).GetAsMatrix4()
                self.info[cams[0] + "_pose"] = cam_pose.copy()

            else:  # multiple cameras
                for cam in cams:
                    self.img_obs.append(self._render(cam))
                    camera_pose_port = self.system.GetOutputPort(cam + "_camera_pose")
                    cam_pose = camera_pose_port.Eval(context).GetAsMatrix4()
                    self.info[cam + "_pose"] = cam_pose.copy()

            self.info["base_camera_pose"] = self.info["panda_link0_cam_pose"]
            self.info["hand_camera_pose"] = self.info["panda_hand_cam_pose"]

            if self.task_config.env.use_point_cloud_model:
                self.compute_pointcloud_observation(self.img_obs, self.full_mask_data, vis=True)

                if len(self.info["point_cloud"].shape) == 1 or self.empty_pointcloud_step:
                    print("empty pointcloud.")
                    if hasattr(self.task_config, "data_collection"):
                        print("reset!")
                        return self.reset()  # will count as failures

        self.info["joint_pos"] = self.joint_positions
        self.info["ee_pose"] = self.ee_pose
        self.info["cam_intr"] = CAM_INTR
        self.info["timestep"] = self.time_step
        self.info["task_name"] = self.task_config.task_name
        self.info["success"] = self._get_reward(self.context, self.action) > 0
        self.info["overhead_image"] = cv2.resize(self.img_obs[0][..., :3], (512, 512))
        self.info["overhead_image"] = (self.info["overhead_image"] * 255).astype(np.uint8)

        self.info["wrist_image"] = cv2.resize(self.img_obs[1][..., :3], (512, 512))
        self.info["wrist_image"] = (self.info["wrist_image"] * 255).astype(np.uint8)

        if hasattr(self, "tool_class_name"):
            self.info["tool_class_name"] = self.tool_class_name

        if hasattr(self, "tool_name"):
            self.info["tool_name"] = self.tool_name
            self.info["tool_pose"] = self._get_tool_pose(context)

        if hasattr(self, "object_name"):
            self.info["tool_object_name"] = self.object_name
            self.info["object_pose"] = self._get_object_pose(context)

        self.state_obs = self.compute_state_obs()
        self.info["state"] = self.state_obs
        self.info["next_state"] = self.state_obs.copy()  # ...

        if self.task_config.blender_render:
            print("publish blender image")
            from env.env_util import blender_color_cam

            blender_context = blender_color_cam.GetMyContextFromRoot(self.context)
            blender_color_cam.Publish(blender_context)

        try:
            self.contact_results = self.contact_measurement_port.Eval(context)
        except:
            print("nan in contact results")
            pass

        # detect collisions and reset if needed
        for i in range(self.contact_results.num_point_pair_contacts()):
            contact_info_i = self.contact_results.point_pair_contact_info(i)
            body_ia_index = contact_info_i.bodyA_index()
            body_ib_index = contact_info_i.bodyB_index()

            body_a_name = self.plant.get_body(body_ia_index).name()
            body_b_name = self.plant.get_body(body_ib_index).name()
            if self.check_collision_name(body_a_name, body_b_name):
                color_print(
                    (f"{body_a_name} collide with {body_b_name} reset"),
                    "light_slate_blue",
                )
                self.need_termination = True
                self.info["collision"] = True  # emergent reset

            if self.task_config.env.debug_contacts:
                if "finger" not in body_a_name and "finger" not in body_b_name:
                    color_print(
                        "body A: {} body B: {} force: {}".format(
                            body_a_name,
                            body_b_name,
                            np.round(contact_info_i.contact_force(), 3),
                        ),
                        "light_slate_blue",
                    )

        # get jacobian
        self.joint_jacobian = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            self.plant.GetBodyByName("panda_hand").body_frame(),
            [0, 0, 0],
            self.plant.world_frame(),
            self.plant.world_frame(),
        )[:, :7]

        self._get_robot_freq_data()
        self.info["external_torque"] = self.external_torque  # for each joint
        # for the end effector, rotation first
        self.info["endeffector_wrench_diff"] = self.endeffector_wrench_diff
        self.info["endeffector_wrench"] = self.endeffector_wrench

        if len(self.history_wrench_obs) > 3:
            self.info["history_wrench_obs"] = np.concatenate(self.history_wrench_obs)
            self.at_contact = np.linalg.norm(self.info["history_wrench_obs"]) > 0.0
            if self.task_config.use_controller_portswitch and self.task_config.automatic_controller_switch:

                # testing purpose
                if self.at_contact:
                    self.controller_switch_port.FixValue(self.context, InputPortIndex(1))
                else:
                    self.controller_switch_port.FixValue(self.context, InputPortIndex(2))

        # compute keypoint information
        if hasattr(self, "tool_name"):
            if not hasattr(self, "tool_keypoints"):
                self.get_keypoint_information()
            self.update_keypoint_information()

    def render_two_views(self):
        # for recording video purpose
        base_cam_img = self._render("panda_link0_cam")
        wrist_cam_img = self._render("panda_hand_cam")
        return base_cam_img, wrist_cam_img

    def render_any_view(self, pose):
        body = self.plant.GetBodyByName("dummy_cam_body")
        self.plant.SetFreeBodyPose(self.plant_context, body, RigidTransform(pose))
        return self._render("dummy_cam_body_cam")

    def get_combined_mask(self, cam):
        render_port = self.system.GetOutputPort(cam + "_label_image")
        maskdata = render_port.Eval(self.context).data.copy()

        # simulate whatever is above the table and not the robot
        object_bodies = self.plant.GetBodyIndices(self.object_asset)
        tool_bodies = self.plant.GetBodyIndices(self.tool_asset)
        # masked_ids  = [int(x) for x in object_bodies] + [int(x) for x in tool_bodies]

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

    def _render(self, cam):
        """rendering the image of the scene based on the spec"""
        data = []
        if "RGB" in self.task_config.env.img_type:
            render_port = self.system.GetOutputPort(cam + "_rgb_image")
            rgbdata = render_port.Eval(self.context).data.copy()[..., :3] / 255.0
            data.append(rgbdata)

        if "D" in self.task_config.env.img_type:
            render_port = self.system.GetOutputPort(cam + "_depth_image")
            depthdata = render_port.Eval(self.context).data.copy()
            depthdata[depthdata == np.inf] = 3
            data.append(depthdata)

        if "M" in self.task_config.env.img_type:
            filtered_maskdata = self.get_combined_mask(cam)
            # filtered_maskdata = filtered_maskdata[...,None].astype(np.bool)
            data.append(filtered_maskdata)

        if "POINTS" in self.task_config.env.img_type:
            render_port = self.system.GetOutputPort(cam + "_point_cloud")
            pointdata = render_port.Eval(self.context).data.copy()
            data.append(pointdata)
        else:
            data = np.concatenate(data, axis=-1)

        if self.task_config.env.debug_obs_image and self.time < 3:
            print("visualize rendered image")
            visualize_rendered_image(data)

        return data

    def terminate_mechanism(self):
        pass

    def export_context(self):
        """export the current simulator context"""
        context = self.simulator.get_context()
        return context.Clone()

    def load_context(self, context):
        """load the context to the current simulation"""
        self.context.SetTimeStateAndParametersFrom(context)
        self.simulator.Initialize()

    def seed(self, seed=None):
        """Implements gym.Env.seed using Drake's RandomGenerator."""
        if seed:
            self.generator = RandomGenerator(seed)
        else:
            seed = self.generator()

        return [seed]

    def visualize_task_keypoints(self, point_sets, pointset_name=""):
        """visualize the keypoint locations"""
        if not self.task_config.env.vis_keypoints:
            return
        from env.env_util import meshcat, visualizer

        pcd = o3d.geometry.PointCloud()
        points = np.stack(point_sets, axis=0)
        point_colors = np.stack(([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]), axis=0)[
            : len(point_sets)
        ]
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd.colors = o3d.utility.Vector3dVector(point_colors[: len(points)])
        draw_open3d_point_cloud(meshcat, pointset_name, pcd, point_size=0.003, sphere_vis=True)
