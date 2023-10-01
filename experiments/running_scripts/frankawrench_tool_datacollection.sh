
TOOL_IDX=${1-'0'}
unset DISPLAY

rm -rf assets/demonstrations/*wrench*
rm -rf assets/demonstrations/processed/*wrench*
rm -rf ../metaworld_weights/data/demo_drake/*wrench*
for TOOL_IDX in 0 1 2 3 4
do
echo "cleanup...."
python -m core.run cuda=True render=True env_name=FrankaDrakeWrenchEnv num_envs=20 run_expert=True \
		  save_demonstrations=True start_episode_position=0 num_workers=0 task=FrankaDrakeWrenchEnvMergingWeights \
		  train=FrankaDrakeEnv task.tool_fix_idx=${TOOL_IDX} save_demo_suffix=tool_${TOOL_IDX} max_episodes=1000 \
		  task.env.randomize_camera_extrinsics=True  record_video=True +task.data_collection=True 
		  
python -m misc.collapse_dataset  -e wrenchforwrench --tool $TOOL_IDX

done
wait 