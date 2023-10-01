
TOOL_IDX=${1-'0'}

python -m core.run cuda=True render=True env_name=FrankaDrakeHammerEnv num_envs=1 run_expert=True task.use_meshcat=True \
	  save_demonstrations=True start_episode_position=0 num_workers=0 task=FrankaDrakeHammerEnvMergingWeights \
	  train=FrankaDrakeEnv task.tool_fix_idx=${TOOL_IDX} save_demo_suffix=tool_${TOOL_IDX} max_episodes=1000 \
	  record_video=True +task.data_collection=True

 