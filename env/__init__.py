from .frankadrakeenv import FrankaDrakeEnv
from .frankadrakeknifeenv import FrankaDrakeKnifeEnv
from .frankadrakewrenchenv import FrankaDrakeWrenchEnv
from .frankadrakespatulaenv import FrankaDrakeSpatulaEnv
from .frankadrakehammerenv import FrankaDrakeHammerEnv

from gym.envs.registration import register
import gym

ENV_ID = {
    "FrankaDrakeEnv": "FrankaDrakeEnv-v0",
    "FrankaDrakeKnifeEnv": "FrankaDrakeKnifeEnv-v0",
    "FrankaDrakeSpatulaEnv": "FrankaDrakeSpatulaEnv-v0",
    "FrankaDrakeWrenchEnv": "FrankaDrakeWrenchEnv-v0",
    "FrankaDrakeHammerEnv": "FrankaDrakeHammerEnv-v0",
}

ENV_CLASS = {
    "FrankaDrakeEnv": "FrankaDrakeEnv",
    "FrankaDrakeKnifeEnv": "FrankaDrakeKnifeEnv",
    "FrankaDrakeSpatulaEnv": "FrankaDrakeSpatulaEnv",
    "FrankaDrakeWrenchEnv": "FrankaDrakeWrenchEnv",
    "FrankaDrakeHammerEnv": "FrankaDrakeHammerEnv",
}


def make_env(env_name, **kwargs):
    env_id = ENV_ID[env_name]
    return gym.make(env_id, **kwargs)  #  eval(env_name)(**kwargs) #


def register_env(env_name):
    assert env_name in ENV_ID, "unknown environment"
    env_id = ENV_ID[env_name]
    env_class = ENV_CLASS[env_name]
    register(id=env_id, entry_point="env." + env_name.lower() + ":" + env_class)
