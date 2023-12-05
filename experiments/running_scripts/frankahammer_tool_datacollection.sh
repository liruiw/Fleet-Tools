
unset DISPLAY

TOOL_IDX=${1-'0'}
rm -rf assets/demonstrations/*hammer*
rm -rf assets/demonstrations/processed/*hammer*
rm -rf ../metaworld_weights/data/demo_drake/*hammer*

for TOOL_IDX in  0 1 2 3 4 5
do
echo "cleanup...."
 
python -m core.run cuda=True render=True env_name=FrankaDrakeHammerEnv num_envs=20 run_expert=True \
		  save_demonstrations=True start_episode_position=0 num_workers=0 task=FrankaDrakeHammerEnvMergingWeights \
		  train=FrankaDrakeEnv save_demo_suffix=tool_${TOOL_IDX} task.tool_fix_idx=$TOOL_IDX  max_episodes=30000 \
		  task.env.randomize_camera_extrinsics=True  record_video=True +task.data_collection=True 
python -m scripts.collapse_dataset  -e hammerforhammer --tool $TOOL_IDX

done
wait


# collect per tool and different folder for instance 
#  task.tool_fix_idx