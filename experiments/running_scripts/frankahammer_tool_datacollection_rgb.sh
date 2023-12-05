

TOOL_IDX=${1-'0'}

python -m core.run cuda=True render=True env_name=FrankaDrakeHammerEnv num_envs=20 run_expert=True \
		  save_demonstrations=True start_episode_position=0 num_workers=0 task=FrankaDrakeHammerEnvMergingWeights \
		  train=FrankaDrakeEnv save_demo_suffix=tool_${TOOL_IDX} task.tool_fix_idx=$TOOL_IDX  max_episodes=1500   num_episode=1500 \
		  task.env.randomize_camera_extrinsics=True  record_video=False +task.data_collection=True +task.env.use_image=True
python -m misc.collapse_dataset  -e hammerforhammer --tool $TOOL_IDX



# collect per tool and different folder for instance 
#  task.tool_fix_idx