from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from canalenv.envs.canalenv_gym import SumoEnv
import random, time, sys, re, os, traci, sumolib, gym, canalenv
import numpy as np

def make_env(label, gui_f=True):
    def _init():
        env = SumoEnv(label=label, gui_f=gui_f)
        env = Monitor(env, None)
        return env
    return _init

if __name__ == "__main__":
    ## check environment correctness
    env_config = {'gui_f':False, 'label':'test'}
    test_py_env = gym.make('SumoEnv-v0', **env_config)
    check_env(test_py_env)
    print("state space:", test_py_env.observation_space)
    print("action space:", test_py_env.action_space)
    print("random action:", test_py_env.action_space.sample())
    test_py_env.close()
    test = False
    moretest = False
    parallel = True
    deterministic = False
    moretestLen = 100

    if not test:
        """ demo in SUMO GUI """
        os.chdir(os.path.dirname(__file__))
        work_dir = os.getcwd()
        modelPath = os.path.join(work_dir, "trained_agents/best_model.zip")
        statsPath = os.path.join(work_dir, "trained_agents/best_model.pkl")

        ## load policy
        model = PPO.load(modelPath)

        ## for evaluation
        evaluate_py_env = DummyVecEnv([make_env(
            gui_f=True, 
            label="test"
            )])
        evaluate_py_env = VecNormalize.load(statsPath, evaluate_py_env)
        #  do not update them at test time
        evaluate_py_env.training = False
        # reward normalization is not needed at test time
        evaluate_py_env.norm_reward = False
        print("load trained model")
    else:
        ## for random agent test
        env_config = {'gui_f':True, 'label':'test'}
        evaluate_py_env = gym.make('SumoEnv-v0', **env_config)

    obs = evaluate_py_env.reset()
    steps = 0

    tmpRewards = []
    startTime = time.perf_counter()
    testActions = [1]*1+[2]+[1]+[0]*1000
    actIndex = 0
    while True:
        if not test:
            action, _states = model.predict(obs,deterministic=deterministic)
            steps = evaluate_py_env.get_attr('step_count')
            simSteps = evaluate_py_env.get_attr('sim_step_count')
            obs, reward, done, info = evaluate_py_env.step(action)
        else:
            action = evaluate_py_env.action_space.sample() # [testActions[actIndex]] # [1] # [0,0] #
            # action = [1 for _ in range(evaluate_py_env.numCharger)]
            steps += 1
            obs, reward, done, info = evaluate_py_env.step(action)
            actIndex += 1
        
        if not test:
            tmpRewards.extend(reward)
        else:
            tmpRewards.append(reward)
        if done:
            if not test:
                print("reward=", np.sum(tmpRewards), "spent:", steps, "steps; eligible steps:", simSteps)
                print("All charged?", evaluate_py_env.get_attr('charge_complete'))
                print({k:v for k,v in info[0].items() if k not in ["terminal_observation"]})
            else:
                print("reward=", np.sum(tmpRewards), "spent:", steps)
                print("All charged?", evaluate_py_env.charge_complete)
                print({k:v for k,v in info.items() if k not in ["terminal_observation"]})
            break
            
    endTime = time.perf_counter()
    print("took", endTime-startTime)
        
    evaluate_py_env.close()

    ## more tests
    if not test and moretest:
        if not parallel:
            episode_rewards = []
            episode_lengths = []
            for _ in range(100):
                evaluate_py_env = DummyVecEnv([make_env(
                    gui_f=False, 
                    label="evaluate"
                    )])
                evaluate_py_env = VecNormalize.load(statsPath, evaluate_py_env)
                # do not update them at test time
                evaluate_py_env.training = False
                # reward normalization is not needed at test time
                evaluate_py_env.norm_reward = False
                obs = evaluate_py_env.reset()

                tmpRewards = []
                startTime = time.perf_counter()
                while True:
                    action, _states = model.predict(obs,deterministic=False)
                    steps = evaluate_py_env.get_attr('step_count')
                    simSteps = evaluate_py_env.get_attr('sim_step_count')
                    obs, reward, done, info = evaluate_py_env.step(action)
                    tmpRewards.extend(reward)
                    if done:
                        print("-----------------------------------------")
                        print("reward=", np.sum(tmpRewards), "spent:", steps, "steps; eligible steps:", simSteps)
                        episode_rewards.append(np.sum(tmpRewards))
                        episode_lengths.append(steps)
                        print("All charged?", evaluate_py_env.get_attr('charge_complete'), info[0]["charge_complete"])
                        print({k:v for k,v in info[0].items() if k not in ["terminal_observation"]})
                        break
                        
                endTime = time.perf_counter()
                print("took", endTime-startTime)
                
                evaluate_py_env.close()
                
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            print(f"Episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        else:
            #### parallel version
            startTime = time.perf_counter()
            n_envs = 72
            n_eval_episodes = n_envs*1
            evaluate_py_env = SubprocVecEnv([make_env(
                gui_f=False, 
                label="evaluate"
                ) for i in range(n_envs)])
            evaluate_py_env = VecNormalize.load(statsPath, evaluate_py_env)
            #  do not update them at test time
            evaluate_py_env.training = False
            # reward normalization is not needed at test time
            evaluate_py_env.norm_reward = False
            
            episode_rewards = []
            episode_lengths = []
            episode_counts = np.zeros(n_envs, dtype="int")
            # Divides episodes among different sub environments in the vector as evenly as possible
            episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

            current_rewards = np.zeros(n_envs)
            current_lengths = np.zeros(n_envs, dtype="int")
            observations = evaluate_py_env.reset()
            states = None
            episode_starts = np.ones((n_envs,), dtype=bool)
            while (episode_counts < episode_count_targets).any():
                actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
                observations, rewards, dones, infos = evaluate_py_env.step(actions)
                current_rewards += rewards
                current_lengths += 1
                for i in range(n_envs):
                    if episode_counts[i] < episode_count_targets[i]:
                        # unpack values so that the callback can access the local variables
                        reward = rewards[i]
                        done = dones[i]
                        info = infos[i]
                        episode_starts[i] = done

                        if dones[i]:
                            episode_rewards.append(current_rewards[i])
                            episode_lengths.append(current_lengths[i])
                            print("-----------------------------------------")
                            print("reward=", current_rewards[i], "spent:", info["steps"], "steps; eligible steps:", info["sim_step_count"])
                            print("All charged?", info["charge_complete"])
                            print({k:v for k,v in info.items() if k not in ["terminal_observation"]})
                            episode_counts[i] += 1
                            current_rewards[i] = 0
                            current_lengths[i] = 0

            evaluate_py_env.close()
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            print(f"Episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            endTime = time.perf_counter()
            print("took", endTime-startTime)
        
    elif test and moretest:
        env_config = {'gui_f':False, 'label':'test'}
        evaluate_py_env = gym.make('SumoEnv-v0', **env_config)
        allChargedCount = 0
        for _ in range(moretestLen):
            obs = evaluate_py_env.reset()
            steps = 0
            tmpRewards = []
            startTime = time.perf_counter()
            
            while True:
                action = evaluate_py_env.action_space.sample() # [2] # [0,0] #
                steps += 1
                obs, reward, done, info = evaluate_py_env.step(action)
                tmpRewards.append(reward)
                if done:
                    print("-----------------------------------------")
                    print("reward=", np.sum(tmpRewards), "spent:", steps)
                    print("All charged?", evaluate_py_env.charge_complete, evaluate_py_env.maxVehNum)
                    print({k:v for k,v in info.items() if k not in ["terminal_observation"]})
                    if evaluate_py_env.charge_complete:
                        allChargedCount += 1
                    break
                    
            endTime = time.perf_counter()
            print("took", endTime-startTime)
        
        evaluate_py_env.close()
        print("Charge success rate:", allChargedCount/moretestLen)