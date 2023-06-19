# SUMO-RL-MobiCharger

<!-- start intro -->

SUMO-RL-MobiCharger provides an OpenAI-gym-like environment for the implementation of RL-based mobile charger dispatching methods on the [SUMO](https://github.com/eclipse/sumo) simulator. The fetures of this environment are four-fold:

- A simple and customizable interface to work with Reinforcement Learning for Dispatching of Mobile Chargers on city-scale transportation network with SUMO
- Compatibility with OpenAI-gym and popular RL libraries such as [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
- Easy modification of state and reward functions for research focusing on vehicle routing or scheduling problems
- Support parallel training of multiple environments via the use of ```SubprocVecEnv``` in [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

| ![demo.gif](accessories/demo.gif) | 
|:--:| 
| *Blue vehicles are mobile chargers, yellow vehicles are electric vehicles, green highlight means charging between mobile chargers and EVs, and blue highlight means charging between mobile chargers and charging stations* |

<!-- end intro -->

## Install

<!-- start install -->

### Install SUMO >= 1.16.0:

Install SUMO as in their [doc](https://sumo.dlr.de/docs/Installing/Linux_Build.html).
Note that this environment uses Libsumo as default for simulation speedup, but sumo-gui does not work with Libsumo on Windows ([more details](https://sumo.dlr.de/docs/Libsumo.html#python)). If you need to go back to TraCI, uncomment ```import traci``` and modify the code in ```reset()``` of [SumoEnv](canalenv/envs/canalenv_gym.py).

### Install the Necessary Packages

Install the necessary packages listed in [requirements.txt](https://github.com/liyan2015/SUMO-RL-MobiCharger/blob/main/requirements.txt)

### Install SUMO-RL-MobiCharger

Clone the latest version and install it in gym
```bash
git clone https://github.com/liyan2015/SUMO-RL-MobiCharger.git
cd SUMO-RL-MobiCharger
pip install -e .
```

<!-- end install -->

## Training & Testing

<!-- start training -->

### Register SUMO-RL-MobiCharger in RL Baselines3 Zoo

The main class is [SumoEnv](canalenv/envs/canalenv_gym.py). To train with RL Baselines3 Zoo, you need to register the environment as in their [doc](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/custom_env.html) and add the following code to ```exp_manager.py```:

```python
# On most env, SubprocVecEnv does not help and is quite memory hungry
# therefore we use DummyVecEnv by default
if "SumoEnv" not in self.env_name.gym_id:
    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=self.seed,
        env_kwargs=self.env_kwargs,
        monitor_dir=log_dir,
        wrapper_class=self.env_wrapper,
        vec_env_cls=self.vec_env_class,
        vec_env_kwargs=self.vec_env_kwargs,
        monitor_kwargs=self.monitor_kwargs,
    )
else:
    def make_env(
        env_config={
            'gui_f':False, 
            'label':'evaluate'
        }, rank: int = 0, seed: int = 0
        ):
        def _init():
            env = gym.make('SumoEnv-v0', **env_config)
            env = Monitor(env, log_dir)
            env.seed(seed + rank)
            env.action_space.seed(seed + rank)
            return env
        set_random_seed(seed)
        return _init
    
    if eval_env:
        if self.verbose > 0:
            print("Creating evaluate environment.")
            
        env = SubprocVecEnv([make_env() for i in range(n_envs)])
    else:
        env = SubprocVecEnv([make_env(
            {
                'gui_f':False, 
                'label':'train'+str(i+1)
            }, rank=i*2) for i in range(n_envs)])
```

### Training

For training, use the following command line:
  
```bash
python train.py --algo ppo --env SumoEnv-v0 --num-threads 1 --progress --conf-file hyperparams/python/sumoenv_config.py --save-freq 500000 --log-folder /usr/data2/canaltrain_log/ --tensorboard-log /usr/data2/canaltrain_tensorboard/ --verbose 2 --eval-freq 2000000 --eval-episodes 10 --n-eval-envs 10 --vec-env subproc
```

### Resume Training

For resume training with different EV route files,  use the following command line or check the [doc](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/train.html#resume-training) of RL Baselines3 Zoo:

```bash
python train.py --algo ppo --env SumoEnv-v0 --num-threads 1 --progress --conf-file hyperparams/python/sumoenv_config.py --save-freq 500000 --log-folder /usr/data2/canaltrain_log/ --tensorboard-log /usr/data2/canaltrain_tensorboard/ --verbose 2 --eval-freq 2000000 --eval-episodes 10 --n-eval-envs 10 --vec-env subproc -i /usr/data2/canaltrain_log/ppo/SumoEnv-v0_16/rl_model_12999532_steps.zip
```

### Testing

Change the ```model_path``` and ```stats_path``` in ```canal_test.py``` and run:

```bash
python canal_test.py
```

<!-- end training -->

## MDP - Observation, Action and Reward

### Observation

<!-- start observation -->

The default observation for the agent is a vector:
```python
    obs = [SOC_state, charger_state, elig_act_state, dir_state, charge_station_state]
```
- ```SOC_state``` indicates the amount of SOC on the road network pending to be refilled by mobile chargers
- ```charger_state``` indicates current road segment, staying time, charging_others bit, charge_self bit, SOC, distance to target vehicle and neighbor_vehicle bit of each mobile charger
- ```elig_act_state``` indicates the eligible actions that each mobile charger can take at current road segment
- ```dir_state``` indicates the best action of each mobile charger given its current road segment
- ```charge_station_state``` indicates the remaining SOCs that the mobile chargers will have if they go to the charging stations for a recharge

<!-- end observation -->

### Action

<!-- start action -->

The action space is discrete. Each edge in SUMO network is partitioned into several road segments:
    
<p align="center">
<img src="accessories/actions.jpg" align="center" width="75%"/>
</p>

Thus, the possible actions of the agent at each road segment can be illustrated as:

<p align="center">
<img src="accessories/possible_actions.jpg" align="center" width="75%"/>
</p>

Througout the road network, a mobile charger can only take maximally 6 actions: stay (0), charge vehicles (1), go downstream road segments (2-5).

<!-- end action -->

### Reward

<!-- start reward -->

The default reward function is defined as:
    
- ```+ 2``` if a mobile charger charges an EV with ```step_charged_SOC```
- ```+ 3 * charger.step_charged_SOC + 0.5 * (1 - before_SOC)``` if a mobile charger charges itself with ```step_charged_SOC```
- ```+ 8e-2``` if a mobile charger takes the best action
- ```- 8e-2``` if a mobile charger takes an action different from the best one
- ```- 8e-1``` if a mobile charger takes an ineligible action given its current road segment
- ```- 300``` if a mobile charger exhausts its SOC
- ```+ 250``` if the agent succeeds in charging all the EVs and support the completion of their trips

<!-- end reward -->


## Citing

<!-- start citation -->

If you use this repository, please cite:
```bibtex
@article{yan2022mobicharger,
  title={MobiCharger: Optimal Scheduling for Cooperative EV-to-EV Dynamic Wireless Charging},
  author={Yan, Li and Shen, Haiying and Kang, Liuwang and Zhao, Juanjuan and Zhang, Zhe and Xu, Chengzhong},
  journal={IEEE Transactions on Mobile Computing},
  volume={Early Access}, 
  year={2022},
  publisher={IEEE}
}
```

<!-- end citation -->
