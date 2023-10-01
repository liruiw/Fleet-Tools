from core.utils import *
import json
import core.expert.kpam.SE3_utils as SE3_utils
from env.env_util import *

# The specification of optimization problem
import core.expert.kpam.term_spec as term_spec
import core.expert.kpam.mp_terms as mp_terms
from core.expert.kpam.optimization_problem import OptimizationProblemkPAM, solve_kpam
from core.expert.kpam.optimization_spec import OptimizationProblemSpecification
from core.expert.kpam.mp_builder import OptimizationBuilderkPAM

from colored import fg
import IPython
from core.expert.base_expert import *


class AnalyticExpert(TaskExpert):
    # keypoint-based method for tool use
    def __init__(self, envs, args, cfg, robot_id=-1):
        super(AnalyticExpert, self).__init__(envs, args, cfg, robot_id=-1)
        self.env = envs
        self.args = args
        self.cfg = cfg
        self.robot_id = robot_id
        self.tool_name = self.env.task_config.tool_class_name
        self.task_name = self.env.task_config.task_name

        self.plan_time = 0
        self.expert_differentialIK = cfg.task.expert_differentialIK
        self.goal_joint = None
        self.reset_expert()
        self.setup()

    def setup(self):
        # load keypoints
        self.solved_ik_times = []
        self.joint_traj_waypoints = []

    def reset_expert(self):
        """reinitialize expert state"""
        self.joint_space_traj = None
        self.task_space_traj = None
        self.contact_task_space_traj = None
        self.plan_succeeded = False

        self.plan_time = 0.0
        self.solved_expert_traj = []

    def check_plan_empty(self):
        """check if already have a plan"""
        return self.joint_space_traj is None

    def check_replan(self):
        """check if needs to replan"""
        return self.env.time < self.standoff_time and (
            self.env.time - self.plan_time > self.env.task_config.replan_freq
        )

    def use_contact_expert(self):
        """compute the actuation trajectory by planning through contact"""
        if self.contact_task_space_traj is None:  #  and self.env.time > 1 self.cfg.task.task_completion_time:
            # self.contact_expert = ContactExpert(self.env, self.args, self.cfg)
            self.build_tool_object_env()
            self.contact_task_space_traj = self.trajopt_through_contact()

    def select_action(self, state, env_idx=0):
        """the main calling function"""
        """select action for the current state, base on the env time and the solved trajectory"""
        s = time.time()
        if self.check_plan_empty() or self.check_replan():
            self.standoff_time = get_interp_time(self.env.time, self.cfg.task.task_completion_time, 0.7)
            if self.env.task_config.joint_space_kpam:
                self.solve_kpam_joint()
            else:
                self.solve_kpam_pose()

            print("env time: {:.3f} plan generation time: {:.3f}".format(self.env.time, time.time() - s))
            self.solve_postactuation_traj()
            self.solve_joint_traj()

        if self.env.task_config.use_meshcat and hasattr(self, "curr_solution_tool_keypoint_head"):
            self.env.visualize_task_keypoints(
                [
                    self.curr_solution_tool_keypoint_head,
                    self.curr_solution_tool_keypoint_tail,
                ],
                "solution_points",
            )
        extra = {"reset": self.check_plan_empty() and not self.expert_differentialIK}
        return self.get_action(), extra

    def calculate_distance_from_expert_plan(self):
        """compute the distance from the current environment state from what the expert plan is"""
        target_pose = self.task_space_traj.value(self.env.time)
        curr_state = self.env.ee_pose.reshape(4, 4)
        # compute point distance from the current and the target states

    def create_opt_problem(self, optimization_spec):
        """create a keypoint optimization problem from the current keypoint state"""
        if len(self.env.task_config.overwrite_kpam_config_file) > 0:
            optimization_spec.load_from_yaml(
                "core/expert/kpam/config/{}.yaml".format(self.env.task_config.overwrite_kpam_config_file)
            )
        else:
            optimization_spec.load_from_yaml("core/expert/kpam/config/{}.yaml".format(self.task_name))
        print(f"load tool keypoint file from {self.task_name}.yaml")

        # match the table
        constraint_update_keypoint_target = {"tool_head": self.env.curr_tool_obj_keypoints}

        # minimize movement
        cost_update_keypoint_target = {
            "tool_head": self.env.curr_tool_keypoint_head,
            "tool_tail": self.env.curr_tool_keypoint_tail,
            "tool_side": self.env.curr_tool_keypoint_side,
        }

        for term in optimization_spec._cost_list:
            if hasattr(term, "keypoint_name") and term.keypoint_name in cost_update_keypoint_target.keys():
                term.target_position = cost_update_keypoint_target[term.keypoint_name]

        for term in optimization_spec._constraint_list:
            if hasattr(term, "keypoint_name") and term.keypoint_name in constraint_update_keypoint_target.keys():
                term.target_position = constraint_update_keypoint_target[term.keypoint_name]

        return optimization_spec

    # tool use related
    def solve_kpam_joint(self, generate_traj=True):
        """solve the formulated kpam problem and get goal joint"""
        s = time.time()
        self.env.get_obs(render=False)

        # solve for the goal end effector pose to match keypoints
        keypoint_loc = np.stack(
            (
                self.env.curr_tool_keypoint_head,
                self.env.curr_tool_keypoint_tail,
                self.env.curr_tool_keypoint_side,
            ),
            axis=0,
        )

        keypoint_loc_in_hand = np.stack(
            (
                self.env.tool_keypoint_head_in_hand,
                self.env.tool_keypoint_tail_in_hand,
                self.env.tool_keypoint_side_in_hand,
            ),
            axis=0,
        )

        optimization_spec = OptimizationProblemSpecification()
        optimization_spec = self.create_opt_problem(optimization_spec)

        constraint_dicts = [c.to_dict() for c in optimization_spec._constraint_list]
        cost_dicts = [c.to_dict() for c in optimization_spec._cost_list]

        # need to parse the kpam config file and create a kpam problem
        indexes = np.random.randint(len(anchor_seeds), size=(8,))
        random_seeds = [self.env.joint_positions.copy()] + [anchor_seeds[idx] for idx in indexes]
        solutions = []

        for seed in random_seeds:
            res = solve_ik_kpam(
                [constraint_dicts, cost_dicts],
                self.env.controller_plant,
                self.env.controller_plant.GetFrameByName("panda_hand"),
                keypoint_loc_in_hand,
                self.env.curr_tool_obj_keypoints,
                RigidTransform(self.env.ee_pose.reshape(4, 4)),
                seed.reshape(-1, 1),
                self.env.joint_positions.copy(),
                rot_tol=0.01,
                timeout=True,
                consider_collision=False,
            )

            if res is not None:
                solutions.append(res.get_x_val()[:9])

        diff_ik_context = self.env.differential_ik.GetMyMutableContextFromRoot(self.env.context)
        if len(solutions) == 0:
            print("empty solution in kpam")
            self.env.need_termination = True
            self.goal_joint = self.env.joint_positions[:9].copy()
        else:
            solutions = np.array(solutions)
            joint_positions = self.env.joint_positions[:9]
            dist_to_init_joints = np.linalg.norm(solutions - joint_positions.copy(), axis=-1)
            res = solutions[np.argmin(dist_to_init_joints)]
            self.goal_joint = res

            # not used anyways
            self.env.differential_ik.SetPositions(diff_ik_context, res)

        self.task_goal_hand_pose = self.env.differential_ik.ForwardKinematics(diff_ik_context)
        self.task_goal_hand_pose = np.array(self.task_goal_hand_pose.GetAsMatrix4())
        self.task_goal_tool_pose = self.task_goal_hand_pose @ self.env.tool_rel_pose

        # Transform the keypoint
        self.curr_solution_tool_keypoint_head = SE3_utils.transform_point(
            self.task_goal_hand_pose, keypoint_loc_in_hand[0, :]
        )
        self.curr_solution_tool_keypoint_tail = SE3_utils.transform_point(
            self.task_goal_hand_pose, keypoint_loc_in_hand[1, :]
        )
        self.curr_solution_tool_keypoint_side = SE3_utils.transform_point(
            self.task_goal_hand_pose, keypoint_loc_in_hand[2, :]
        )

        self.plan_time = self.env.time
        self.goal_keypoint = np.stack(
            (
                self.curr_solution_tool_keypoint_head,
                self.curr_solution_tool_keypoint_tail,
                self.curr_solution_tool_keypoint_side,
            ),
            axis=0,
        )

        self.env.info["goal_keypoint"] = self.goal_keypoint
        self.env.info["goal_pose"] = self.task_goal_hand_pose

    # tool use related
    def solve_kpam_pose(self, generate_traj=True):
        """solve the formulated kpam problem and get goal pose
        and then solve for the trajectory"""
        self.env.get_obs(render=False)

        # solve for the goal end effector pose to match keypoints
        keypoint_loc = np.stack(
            (
                self.env.curr_tool_keypoint_head,
                self.env.curr_tool_keypoint_tail,
                self.env.curr_tool_keypoint_side,
            ),
            axis=0,
        )

        # create a kpam problem
        optimization_spec = OptimizationProblemSpecification()
        optimization_spec = self.create_opt_problem(optimization_spec)
        builder = OptimizationBuilderkPAM(optimization_spec)
        problem = builder.build_optimization(keypoint_loc)
        solve_kpam(problem)
        self.task_goal_tool_pose = problem.T_action.dot(self.env.tool_pose)

        # Transform the keypoint
        self.curr_solution_tool_keypoint_head = SE3_utils.transform_point(problem.T_action, keypoint_loc[0, :])
        self.curr_solution_tool_keypoint_tail = SE3_utils.transform_point(problem.T_action, keypoint_loc[1, :])
        self.curr_solution_tool_keypoint_side = SE3_utils.transform_point(problem.T_action, keypoint_loc[2, :])

        self.plan_time = self.env.time
        self.goal_keypoint = np.stack(
            (
                self.curr_solution_tool_keypoint_head,
                self.curr_solution_tool_keypoint_tail,
                self.curr_solution_tool_keypoint_side,
            ),
            axis=0,
        )

        self.env.info["goal_keypoint"] = self.goal_keypoint

    def solve_goal_joint(self):
        # solve a joint to reach task_goal_hand_pose

        seeds = [self.env.joint_positions.copy()]
        indexes = np.random.randint(len(anchor_seeds), size=(8,))
        seeds = seeds + [anchor_seeds[idx] for idx in indexes]
        solved_iks = []

        for seed_ in seeds:
            rot_tol = 0.005
            res = self.env.solve_ik(self.task_goal_hand_pose, seed_, rot_tol)

            if res is not None:
                res = res.get_x_val()[:9]
                solved_iks.append(res)

        if len(solved_iks) == 0:
            print("expert solve goal IK failed")

        else:
            # solve multiple ones and pick the closest one
            diff = np.array(solved_iks) - self.env.joint_positions[None]
            dist = np.linalg.norm(diff, axis=-1)
            self.goal_joint = solved_iks[np.argmin(dist)]

    def set_plan(self, joint_plan):
        self.joint_traj_waypoints = joint_plan
        self.joint_space_traj = PiecewisePolynomial.CubicShapePreserving(
            self.dense_traj_times, np.array(self.joint_traj_waypoints).T
        )
        self.get_task_traj_from_joint_traj()

    def solve_traj_endpoints(self):
        """
        solve for the IKs for each individual waypoint as an initial guess, and then
        solve for the whole trajectory with smoothness cost
        """
        keyposes = self.traj_keyframes
        keytimes = self.sample_times

        self.joint_traj_waypoints = [self.env.joint_positions.copy()]
        self.joint_space_traj = PiecewisePolynomial.FirstOrderHold(
            [self.env.time, self.cfg.task.task_completion_time],
            np.array([self.env.joint_positions.copy(), self.goal_joint]).T,
        )

        self.dense_traj_times = dense_sample_traj_times(self.sample_times, self.cfg.task.task_completion_time)

        # traj_pose = self.env.set_diffik_state(self.env.context, set_joints=self.env.joint_positions)
        print("solve traj endpoint")

        # interpolated joint
        res = solve_ik_traj_with_standoff(
            self.env.controller_plant,
            [self.env.ee_pose.reshape(4, 4), self.task_goal_hand_pose],
            np.array([self.env.joint_positions.copy(), self.goal_joint]).T,
            q_traj=self.joint_space_traj,
            waypoint_times=self.dense_traj_times,
            keyposes=keyposes,
            keytimes=keytimes,
        )

        # solve the standoff and the remaining pose use the goal as seed.
        # stitch the trajectory

        if res is not None:
            #  use the joint trajectory to build task trajectory
            self.joint_traj_waypoints = res.get_x_val().reshape(-1, 9)
            self.joint_traj_waypoints[:, -2:] = 0.04
            self.joint_traj_waypoints = list(self.joint_traj_waypoints)
            self.joint_space_traj = PiecewisePolynomial.CubicShapePreserving(
                self.dense_traj_times, np.array(self.joint_traj_waypoints).T
            )
            self.get_task_traj_from_joint_traj()
        else:
            print("endpoint trajectory not solved! environment termination")
            self.env.need_termination = True
            self.get_task_traj_from_joint_traj()

    def get_task_traj_from_joint_traj(self):
        """forward kinematics the joint trajectory to get the task trajectory"""
        self.pose_traj = []
        ik_times = dense_sample_traj_times(self.sample_times, self.cfg.task.task_completion_time)
        self.dense_ik_times = ik_times
        for traj_time in ik_times:
            diff_ik_context = self.env.differential_ik.GetMyMutableContextFromRoot(self.env.context)
            set_joints = self.joint_space_traj.value(traj_time)
            self.env.differential_ik.SetPositions(diff_ik_context, set_joints)
            pose = self.env.differential_ik.ForwardKinematics(diff_ik_context)
            self.pose_traj.append(pose.GetAsMatrix4())

        self.task_space_traj = PiecewisePose.MakeLinear(ik_times, [RigidTransform(p) for p in self.pose_traj])
        self.env.set_diffik_state(self.env.context)

    def solve_postactuation_traj(self, reach_ratio=0.3, reach_dist=0.12):
        """
        generate the full task trajectory with a FirstOrderHold
        """
        getattr(self, f"generate_{self.task_name}_poses")(reach_ratio, reach_dist)

        if self.env.task_config.contact_planner:
            print("use trajopt to improve the actuation trajectory")
            self.use_contact_expert()

        if self.env.task_config.env.vis_traj_keyframe and self.env.task_config.use_meshcat:
            self.vis_keyframes()

        self.env.info["goal_pose"] = self.task_goal_hand_pose

    def solve_traj_pose_interp(self, ik_times):
        """
        solve for the IKs for each individual waypoint as an initial guess, and then
        solve for the whole trajectory with smoothness cost
        """
        self.task_space_traj = PiecewisePose.MakeLinear(
            self.sample_times, [RigidTransform(p) for p in self.traj_keyframes]
        )
        seed = self.env.joint_positions.copy()
        solved_poses = [self.env.ee_pose.reshape(4, 4)]
        self.joint_traj_waypoints = [self.env.joint_positions.copy()]

        for idx, t in enumerate(ik_times[1:]):
            target_pose = self.task_space_traj.value(t)

            # try different seeds for solving IK
            seeds = [seed, self.goal_joint]
            per_joint_solved_iks = []
            indexes = np.random.randint(len(anchor_seeds), size=(2,))
            seeds = seeds + [anchor_seeds[idx] for idx in indexes]

            for seed_ in seeds:
                rot_tol = 0.1 if t < self.standoff_time else 0.005
                res = self.env.solve_ik(target_pose, seed_.reshape(-1, 1), rot_tol)

                if res is not None:
                    res = res.get_x_val()[:9]
                    per_joint_solved_iks.append(res)

            if len(per_joint_solved_iks) == 0:
                color_print(("expert solve IK failed {:.3f}".format(t)), "sea_green_1b")
                if len(self.joint_traj_waypoints) < idx:
                    color_print("early terminate", "sea_green_1b")
                    break
            else:
                # solve multiple ones and pick the closest one
                diff = np.array(per_joint_solved_iks) - seed[None]
                dist = np.linalg.norm(diff, axis=-1)

                if np.min(dist) > 1 and not self.env.task_config.skip_pose_interpolate:
                    print(f"solve ik min dist: {t:.3f} {np.min(dist):.3f}")
                    continue

                seed = per_joint_solved_iks[np.argmin(dist)]
                res = seed  # use seed for the next joint
                self.solved_ik_times.append(t)
                self.joint_traj_waypoints.append(res[:9])
                solved_poses.append(target_pose)

        solved_poses = np.array(solved_poses)
        self.joint_traj_waypoints = np.array(self.joint_traj_waypoints)

        smoothness_curr = smoothness(self.joint_traj_waypoints.T)
        print("smoothness before refine {:.3f}".format(smoothness_curr))

        res = refine_ik_traj(
            self.env.controller_plant,
            solved_poses,
            np.array(self.joint_traj_waypoints).T,
        )

        if res is not None:
            refined_waypoints = res.get_x_val().reshape(-1, 9)
            smoothness_new = smoothness(refined_waypoints.T)
            print("smoothness after refine {:.3f}".format(smoothness_new))
            if smoothness_new <= smoothness_curr:
                self.joint_traj_waypoints = refined_waypoints

    def solve_joint_traj(self):
        """
        solve single ik of each pose in the pose trajectory
        and then solve a small smoothness trajopt
        """
        if self.env.task_config.pose_interp_traj:
            self.joint_traj_waypoints = [self.env.joint_positions.copy()]
            self.solved_ik_times = [self.env.time]
            print("pose interpolation keyframe sampled times:", self.sample_times)

            ik_times = dense_sample_traj_times(self.sample_times, self.standoff_time)
            exported_context = self.env.export_context()
            """ interpolate pose trajectory and then sole and refine """
            self.solve_traj_pose_interp(ik_times)
            self.env.load_context(exported_context)
            if self.check_plan_empty():  #  self.sample_times[-1] not in self.solved_ik_times
                if len(self.solved_ik_times) < len(ik_times) - 3:
                    color_print((("expert failed to solve for a trajectory. reset env")), "sea_green_1b")
                    self.reset_expert()
                    self.env.need_termination = True

            # each column is a sample straightline trajectory example FirstOrderHold
            self.joint_space_traj = PiecewisePolynomial.CubicShapePreserving(
                self.solved_ik_times, np.array(self.joint_traj_waypoints).T
            )
        else:
            # directly solve the joint end point problem
            self.solve_traj_endpoints()

    def feedback_loop(self, action):
        """
        do a simple feedback loop for each task
        return 6dof delta pose
        """
        return getattr(self, f"reactive_{self.tool_name}_motion")(action)

    def vis_keyframes(self):
        """visualize keyframes specified to execute the task trajectory"""
        from env.env_util import meshcat, visualizer

        for idx, p in enumerate(self.traj_keyframes):
            AddMeshcatTriad(
                meshcat,
                f"keyframe_{idx}",
                X_PT=RigidTransform(p),
                length=0.03,
                radius=0.001,
            )

        vis_tool_pose(meshcat, self.task_goal_tool_pose, "tool_goal_pose")

    def get_joint_action(self):
        """get the joint space action"""
        if self.check_plan_empty():
            print("no joint trajectory")
            return self.env.reset()

        # look ahead
        return self.joint_space_traj.value(self.env.time + self.env.env_dt).reshape(-1)

    def get_pose_action(self, curr_pose, traj_eff_pose):
        """get the task space action"""
        # feedforward position action
        action = pack_pose(se3_inverse(curr_pose).dot(traj_eff_pose))
        return action

    def get_standoff_instant(self):
        # return True if the robot is before the standoff
        return (self.env.time - self.standoff_time) >= 0 and (self.env.time - self.standoff_time) <= (
            self.env.env_dt - 1e-5
        )

    def get_action(self):
        """get the task-space action from the kpam expert joint trajectory"""
        if self.task_space_traj is None:
            print("no task trajectory")
            return self.env.reset()

        curr_pose = self.env.ee_pose.reshape(4, 4)
        traj_eff_pose = self.task_space_traj.value(self.env.time + self.env.env_dt)
        pose_action = self.get_pose_action(curr_pose, traj_eff_pose)
        joint_action = self.get_joint_action()

        if self.env.task_config.reactive_planner and self.env.at_contact:
            new_traj_eff_pose = curr_pose @ unpack_pose(pose_action, rot_type="euler")
            new_ik = self.env.solve_ik(new_traj_eff_pose, joint_action)
            if new_ik is not None:
                joint_action = new_ik.get_x_val()[:9]

        if self.expert_differentialIK or (
            self.env.task_config.contact_shooting_planner and self.env.time > self.standoff_time
        ):  # use workspace motion as supervison

            return pose_action
        return np.concatenate((joint_action, pose_action))
