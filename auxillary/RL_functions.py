# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 08:32:23 2023

@author: jonas
"""

import os
import time
import numpy as np
# import matplotlib.pyplot as plt
# from config_env import lx, ux, ly, uy, BALL_RADIUS

from RL_Pool_env3 import PoolEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import PPO, TD3, A2C, SAC, DDPG
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback
# from custom_callbacks import EvalCallback_masked
from stable_baselines3.common.noise import NormalActionNoise

from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()


def load_RL_model(algo, env_kwargs, model_path=None):
    load_model = True
    # if model_path is None:
    #     load_model = False
    # model_path = "baseline_tests\\models\\PPO_2_balls1709820285\\best_model\\best_model"
    model_path = f'RL_models/{algo}.zip'
    if algo != "PPO_masked":
        env = make_vec_env(PoolEnv, seed=1, n_envs=1, env_kwargs=env_kwargs)
        if algo == "PPO":
            if load_model: model = PPO.load(model_path, env)
            else: model = PPO(policy='MlpPolicy', env=env)
        elif algo == "TD3":
            if load_model: model = TD3.load(model_path, env)
            else: model = TD3(policy='MlpPolicy', env=env)
        elif algo == "A2C":
            if load_model: model = A2C.load(model_path, env)
            else: model = A2C(policy='MlpPolicy', env=env)
        elif algo == "SAC":
            if load_model: model = SAC.load(model_path, env=env)
            else: model = SAC(policy='MlpPolicy', env=env)
        elif algo == "DDPG":
            if load_model: model = DDPG.load(model_path, env=env)
            else: model = DDPG(policy='MlpPolicy', env=env)
    else:
        env = PoolEnv(**env_kwargs)         # Initialize environment
        env = ActionMasker(env, mask_fn)    # Wrap to enable masking
        env = DummyVecEnv([lambda: env])

        if load_model: model = MaskablePPO.load(model_path, env=env)
        else: model = MaskablePPO(policy='MlpPolicy', env=env)

    return model, env


def run_model(model, env_test, max_steps, render=True, deterministic=True, use_masking=False):
    if use_masking and not is_masking_supported(env_test):
        raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

    if not isinstance(env_test, VecEnv):
        env_test = DummyVecEnv([lambda: env_test])

    steps = 0
    success_rate = []
    success_rate_one_ball = []
    rew = []
    first_step = True
    obs = env_test.reset()
    while steps < max_steps:
        if use_masking:
            action_masks = get_action_masks(env_test)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
        else:
            action, _states = model.predict(obs, deterministic=deterministic)

        obs, rewards, dones, info = env_test.step(action)
        if render:
            env_test.render()

        if first_step:
            balls_pocketed = [b.ballclass for b in info[0].get("ball_pocketed")]
            # print(balls_pocketed)
            success_rate_one_ball.append(True if 2 in balls_pocketed else False)
            first_step = False

        if dones[0]:
            if steps % int(max_steps / 20) == 0:
                print(f"{(steps/max_steps) * 100}%")
            steps += 1
            success_rate.append(info[0].get("is_success"))

            rew.append(rewards[0])
            first_step = True
            obs = env_test.reset()

    success_rate = np.mean(success_rate)
    success_rate_one_ball = np.mean(success_rate_one_ball)
    rew_mean = np.mean(rew)
    rew_std = np.std(rew)
    return (success_rate, success_rate_one_ball, rew_mean, rew_std)
# def run_model(model, env_test, max_steps, render=True):
#     steps = 0
#     success_rate = 0
#     rew = 0
#     # max_steps = 50000
#     while steps < max_steps:
#         steps += 1
#         obs = env_test.reset()
#         action, _states = model.predict(obs, deterministic=True)
#         # action, _states = model2.predict(obs)
#         obs, rewards, dones, info = env_test.step(action)
#         # if render:
#         #     env_test.render()
#         success_rate += info[0].get("is_success")
#         rew += rewards[0]

#     success_rate /= max_steps
#     rew /= max_steps
#     return (success_rate, rew)
        
def lr_schedule_linear(initial_lr):
    # initial_lr is the initial learning rate we would normally give to the model
    
    def func(progress_remaining):
        # progress_remaining will decrease from 1 to 0
        return progress_remaining * initial_lr
    
    return func

def load_trained_model(algo, model_path, env):
    m_algo = model_path.split('_')[0]
    assert algo == m_algo, f"Algorithm {algo} and model_path algorithm {m_algo} should be the same"
    
    if algo == "PPO":
        model = PPO.load(model_path, env=env)
    elif algo == "TD3":
        model = TD3.load(model_path, env=env)
        

# def create_states(states, number_balls, classes):
#     xs = np.random.rand(number_balls, 1, states)
#     ys = np.random.rand(number_balls, 1, states)
#     cs = np.array([ [[i]*states] for i in classes ])
#     # test = np.array([ [[i]*states] for i in [2]*(nb-1) + [3] ])
#     ls = np.concatenate((xs, ys, cs), axis = 1)
#     return ls

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_states(n_states, n_balls, classes):
    
    # These are taken from config when zoom = 1
    lx, ly = 47, 47
    ux = 688
    uy = 363
    ball_radius = 7
    x_diff = ux - lx
    y_diff = uy - ly
    # x_diff = ux - lx
    # y_diff = uy - ly
    # s = np.sqrt(x_diff**2 + y_diff**2)
    # r_unit = 2*BALL_RADIUS * 1/s
    
    close_counter = 0
    assert n_balls == len(classes), f"Amount of balls ({n_balls}) should be equal to the amount of classes ({len(classes)})"
    
    ls = []
    for i in range(n_states):
        if i % int(n_states/100) == 0:
            print(f"Current progress: {np.round((i*100)/n_states, 2)}%")
        
        positions = [np.random.randint([lx, ly], [ux, uy])]
        for k in range(n_balls - 1):
            
            overlap = True
            while overlap:
                too_close = False
                new_pos = np.random.randint([lx, ly], [ux, uy])
                
                for pos in positions:
                    dist = np.sqrt((pos[0] - new_pos[0])**2 + (pos[1] - new_pos[1])**2)
                    
                    if dist <= 2.3 * ball_radius:
                        close_counter += 1
                        too_close = True
                        break
                
                if not too_close:
                    positions.append(new_pos)
                    overlap = False
          
        transformed_positions = [np.clip(np.array([(x-lx)/x_diff, (y-ly)/y_diff]), 0.01, 0.99) for x,y in positions]
        ls.append(transformed_positions)
    
    ls = np.array(ls)
    print(ls.shape)
    cs = np.array([ [[i] for i in classes] for _ in range(n_states)])
    ls = np.concatenate((ls,cs), axis=2)
    print(ls.shape)
    ls = np.moveaxis(ls, 0, -1)
    print(ls.shape)
    print(close_counter)
    return ls

# def check_failed_states(ls):
    
#     def transform_coordinates(ball):
#         lx, ly = 47, 47
#         ux = 688
#         uy = 363
        
#         x,y = ball
#         x = int((ux - lx - 2)*x + lx) + 1
#         y = int((uy - ly - 2)*y + ly) + 1
        
#         return np.array([x,y])
    
#     fs = []
#     for state in range(ls.shape[2]):
#         b1, b2 = ls[:,:,state][:,:2]
        
#         b1 = transform_coordinates(b1)
#         b2 = transform_coordinates(b2)
        
#         dist = np.sqrt((b1[0] - b2[0])**2 + (b1[1] - b2[1])**2)
        
#         if dist < 14:
#             fs.append([state, dist, b1, b2])
    
#     return fs


    