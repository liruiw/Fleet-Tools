# Fleet-Tools

[Arxiv]() | [Video]()

This repo is the simulation environment benchmark built with Gym API and Drake simulations for Franka Panda. The repository features tool-use tasks with scripted experts.

![](assets/doc/tooluse_fig.png)


## ⚙️ Installation
1. Create conda environment. ```pip install -r requirements```
2. Use as a package ```pip install -e . ```

## 🚶 Starting Commands
0. These commands will use the hand-scripted joint-space kpam planner to plan for demonstration trajectories 
1. Run ```bash experiments/running_scripts/frankahammer_tool_datacollection_test.sh```. open `local_host:6006`
2. Run `bash experiments/running_scripts/**_datacollection.sh` to generate the data. And export to the global python path such that the training repo `fleet_diffusion` can import it and run evaluation.

## 🚶 Teleop Demonstrations
0. Run ```python -m core.run run_expert=True teleop=True teleop_type=keyboard expert=HumanExpert task=FrankaDrakeSpatulaEnv task.use_meshcat=True```. open `local_host:6006` to use keyboard `wsad` to teleop.
1. If you want to try Oculus Quest >=2. Run:
`python -m core.run run_expert=True teleop=True teleop_type=vr expert=HumanExpert task=FrankaDrakeSpatulaEnv task.use_meshcat=True`
Hold the trigger for moving.

## 💾 File Structure
```angular2html
├── ...
├── Fleet-Tools
|   |── assets 			# mesh data for training
|   |── core 			# source code
|   |   |── agent 		# memory and replay buffer
|   |   |── expert  	# kpam expert to generate demonstrations 
|   |   └── ...
|   |── env 			# environment code
|   |── scripts 		# data preprocessing 
|   |── experiments     # data generation scripts and task configs

└── ...
```

![](assets/doc/realsetup.png)

## 🎛️ Note
- **Code Quality Level**: Tired grad student. 
- **Have Questions**: Open Github issues or send an email. 
- **Drake and Simulation Issues**: Ask in the [Drake](https://github.com/RobotLocomotion/drake) github repo or look for answers in the [API](https://drake.mit.edu/doxygen_cxx/index.html).

## License
MIT

## Acknowledgement
- [Kpam](https://github.com/weigao95/kplan-ros/tree/master/kplan) and [manipulation](https://github.com/RussTedrake/manipulation).
- Assets are a mix of [Mcmaster](https://www.mcmaster.com/) and [Objaverse](https://objaverse.allenai.org/) with CC-BY License. 