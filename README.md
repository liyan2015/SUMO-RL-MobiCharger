# SUMO-RL-MobiCharger
Reinforcement Learning environment for Dispatching of Mobile Chargers with SUMO. Compatible with Gym and popular RL libraries such as stable-baselines3.

<!-- start intro -->

SUMO-RL-MobiCharger provides an OpenAI-gym-like environment for the implementation of RL-based mobile charger dispatching on the [SUMO](https://github.com/eclipse/sumo) simulator. The fetures of this environment are three-fold:

- A simple and customizable interface to work with Reinforcement Learning for Dispatching of Mobile Chargers on city-scale transportation network with SUMO
- Compatibility with OpenAI-gym and popular RL libraries such as [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
- Easy modification of state and reward functions for research focusing on vehicle routing or scheduling problems

<img src="https://github.com/liyan2015/SUMO-RL-MobiCharger/blob/main/accessories/demo.gif?raw=true" width="400"/>

The main class is [SumoEnv](https://github.com/liyan2015/SUMO-RL-MobiCharger/blob/main/canalenv/envs/canalenv_gym.py). To train with RL Baselines3 Zoo, you need to register the environment as in their [doc](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/custom_env.html) and add the following code to ```exp_manager.py```:

```
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

For training, use the following command line:
  
```
python train.py --algo ppo --env SumoEnv-v0 --num-threads 1 --progress --conf-file hyperparams/python/sumoenv_config.py --save-freq 500000 --log-folder /usr/data2/canaltrain_log/ --tensorboard-log /usr/data2/canaltrain_tensorboard/ --verbose 2 --eval-freq 2000000 --eval-episodes 10 --n-eval-envs 10 --vec-env subproc
```


<!-- end intro -->