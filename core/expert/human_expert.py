from core.utils import *
from env.env_util import *

from colored import fg
from core.expert.expert import *
from env.env_util import meshcat

###############################################################################################
# Human expert control
class HumanExpert(AnalyticExpert):
    def __init__(self, envs, args, cfg, robot_id=-1):
        super(HumanExpert, self).__init__(envs, args, cfg, robot_id=-1)
        self.env = envs
        self.args = args
        self.robot_id = robot_id
        self.cfg = cfg
        self.desired_pose = self.env.ee_pose.reshape(4, 4)
        self.desired_joint = self.env.joint_positions
        self.task_transform_pose = np.eye(4)
        self.teleop_cam_pose = None

    def select_controller(self, name="pose"):
        """
        Selects the controller based on the given name.
        """
        if name == "pose":
            self.env.task_config.pose_controller = True
            self.env.controller_switch_port.FixValue(self.env.context, InputPortIndex(1))
        else:
            self.env.task_config.pose_controller = False
            self.env.controller_switch_port.FixValue(self.env.context, InputPortIndex(2))

    def select_action(self, curr_pose, env_idx=0):
        """control the relative delta action for the end effector. simulation environment only."""
        extra = {}
        if self.args.teleop_type == "keyboard":
            action, finger_state = get_keyboard_input(self.env.action_space.sample(), self.robot_id)
            action_pose = unpack_action(action)
            reset = False

        elif self.args.teleop_type == "vr":
            (
                dX_UinitU_W,  # current delta
                finger_state,  # Right Grip
                reset,  # B
                teleop_hold,  # Right Trig
                X_WGdesired,  # A
                exit_loop,
            ) = get_vr_input()

            action_pose = se3_inverse(curr_pose) @ X_WGdesired  # delta pose as action labels
            action = pack_action(action_pose)

            # visualize the target_X_TB using triad
            meshcat.SetTransform("hand_desired_coord", RigidTransform(X_WGdesired))
            meshcat.SetTransform("hand_actual_coord", RigidTransform(curr_pose))

        return action, extra
