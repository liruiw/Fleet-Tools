import argparse

"""
Util to compile command line arguments for core script
"""


def get_parser():
    # Global Parameters
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@", description="Arguments")
    parser.add_argument("--env_name", default="FrankaDrakeSpatulaEnv", help="Choice of environment.")
    parser.add_argument("--logdir", default="runs", help="exterior log directory")
    parser.add_argument("--logdir_suffix", default="", help="log directory suffix")
    parser.add_argument("--cuda", action="store_true", help="run on CUDA (default: False)")

    parser.add_argument("--teleop", action="store_true", help="use teleop control")
    parser.add_argument(
        "--eval",
        default="",
        help="if doing eval, this is the logdir from which to load model weights",
    )
    parser.add_argument("--seed", type=int, default=123456, help="random seed (default: 123456)")

    # rollout related
    parser.add_argument("--agent", type=str, default="BC", help="Type of parallel agent;")
    parser.add_argument(
        "--save_demo_suffix",
        type=str,
        default="",
        help="The suffix for which demo to save",
    )
    parser.add_argument("--expert", type=str, default="AnalyticExpert", help="Type of parallel expert;")
    parser.add_argument("--run_expert", action="store_true", help="use expert for action")
    parser.add_argument(
        "--teleop_type",
        type=str,
        default="keyboard",
        help="Type of teleop method, mouse, keyboard, vr",
    )

    parser.add_argument("--num_envs", type=int, default=10, help="number of robots")
    parser.add_argument("--log_freq", type=int, default=100, help="log frequency")
    parser.add_argument("--render", action="store_true", help="whether or not to render")
    parser.add_argument("--meshcat", action="store_true", help="use meshcat")
    parser.add_argument("--visdom", action="store_true", help="use visdom")
    parser.add_argument("--num_steps", type=int, default=100000, help="maximum number of timesteps ")
    parser.add_argument("--num_episode", type=int, default=1000, help="maximum number of episodes")

    parser.add_argument("--pretrained", type=str, default="", help="pretrained model to load")
    parser.add_argument(
        "--load_pretrained_model",
        type=str,
        default="both",
        help="pretrained policy or representation to load",
    )
    parser.add_argument("--demo_number", type=int, default=1000, help="number of demonstrations needed")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers for dataset")

    # train eval related
    parser.add_argument("--train_task", type=str, default="", help="the train / eval task")
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=10000,
        help="maximum number of training timesteps (default: 10000)",
    )
    parser.add_argument("--training", action="store_true", help="training from offline data")
    parser.add_argument(
        "--test_on_train_scenes",
        action="store_true",
        help="load the training scenes for testing",
    )
    parser.add_argument(
        "--save_demonstrations",
        action="store_true",
        help="saving the training scenes for testing",
    )
    parser.add_argument("--load_demonstrations", action="store_true", help="load demonstration data")
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="record video fro either expert rollout or policy rollout",
    )
    parser.add_argument(
        "--config_suffix",
        type=str,
        default="",
        help="suffix for the training / env config",
    )
    parser.add_argument(
        "--demonstration_dir",
        type=str,
        default="",
        help="the directory where demonstrations are stored",
    )
    parser.add_argument("--fixed_encoder", action="store_true", help="fix the representation encoder")
    parser.add_argument(
        "--start_episode_position",
        type=int,
        default=0,
        help="loaded episode position for the replay buffer",
    )
    parser.add_argument("--save_dir", type=str, default="", help="the directory for the log experiment")
    parser.add_argument(
        "--eval_save_video",
        action="store_true",
        help="save the overhead view video",
        default=True,
    )
    parser.add_argument(
        "--reinit_num",
        type=int,
        default=40,
        help="number of rounds before reiniting ray",
    )

    return parser
