
TOOL_IDX=${1-'0'}
for EPS in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
do
python -m core.run cuda=True render=True env_name=FrankaDrakeHammerEnv num_envs=1 run_expert=True task.use_meshcat=True \
	  save_demonstrations=True start_episode_position=$EPS num_workers=0 task=FrankaDrakeHammerEnvMergingWeights \
	  train=FrankaDrakeEnv task.tool_fix_idx=${TOOL_IDX} save_demo_suffix=tool_${TOOL_IDX} max_episodes=1500 \
	  record_video=False +task.data_collection=True +task.use_image=True  num_episode=1500 


done
wait
python -m scripts.collapse_dataset  -e hammerforhammer --tool $TOOL_IDX 