
TOOL_IDX=${1-'0'}

for EPS in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 4100 4200 4300
do
python -m core.run cuda=True render=True env_name=FrankaDrakeSpatulaEnv num_envs=1 run_expert=True task.use_meshcat=True \
	  save_demonstrations=True start_episode_position=$EPS num_workers=0 task=FrankaDrakeSpatulaEnvMergingWeights \
	  train=FrankaDrakeEnv task.tool_fix_idx=${TOOL_IDX} save_demo_suffix=tool_${TOOL_IDX} max_episodes=1500 \
	  record_video=False +task.data_collection=True +task.use_image=True  num_episode=1500 
done
wait


python -m scripts.collapse_dataset  -e spatulaforspatula --tool $TOOL_IDX 

 