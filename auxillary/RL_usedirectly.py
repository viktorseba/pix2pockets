from stable_baselines3 import PPO, TD3, A2C, DDPG, SAC
# from sb3_contrib.ppo_mask import MaskablePPO

def load_RL_no_env(model_path):
    algo = model_path.split('/')[-1].split('.')[0]
    if algo == 'PPO': model = PPO.load(model_path)
    elif algo == 'TD3': model = TD3.load(model_path)
    elif algo == 'DDPG': model = DDPG.load(model_path)
    elif algo == 'SAC': model = SAC.load(model_path)
    elif algo == 'A2C': model = A2C.load(model_path)
    # elif algo == 'PPO_masked': model = MaskablePPO.load(model_path)
    return model
