import argparse
from email.policy import default
from typing import DefaultDict
import torch
import numpy as np
from collections import defaultdict
from widowx_real_env import *

TARGET_POINT = np.array([0.28425417, 0.04540814, 0.07545623])  # mean

variant=defaultdict(lambda: False)

def get_env_params(start_transform):
    env_params = {
        'fix_zangle': True,  # do not apply random rotations to start state
        'move_duration': 0.2,
        'adaptive_wait': True,
        'move_to_rand_start_freq': 1,
        'override_workspace_boundaries': [[0.100, -0.25, 0.0, -1.57, 0], [0.41, 0.143, 0.33, 1.57, 0]],

        'action_clipping': 'xyz',
        'catch_environment_except': True,
        'target_point': TARGET_POINT,
        'add_states': variant.add_states,
        'from_states': variant.from_states,
        'reward_type': variant.reward_type,
        'start_transform': None if variant.start_transform == '' else start_transforms[start_transform],
        'randomize_initpos': 'full_area'
    }
    return env_params

def eval(policy, env, num_episodes=1):
    """
    Evaluate a policy.
    :param policy: policy to be evaluated
    :param env: environment to be evaluated on
    :param num_episodes: number of episodes to run
    :param render: whether to render the environment
    :return: mean reward
    """
    rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        
        print(f'Episode: {i}')
        import ipdb; ipdb.set_trace()
        
        while not done:
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    
    return torch.mean(torch.tensor(rewards))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path', type=str, default=None)
    argparser.add_argument('--start_transform', type=str, default=None)
    argparser.add_argument('--num_tasks', type=int, default=0)
    argparser.add_argument('--num_trajectory', type=int, default=0)
    args = argparser.parse_args()
    
    model = torch.load(args.path)
    
    from widowx_real_env import JaxRLWidowXEnv
    env_params = get_env_params(args.start_transform)
    env = JaxRLWidowXEnv(env_params, num_tasks=args.num_tasks)
    
    eval(model, env, num_episodes=args.num_trajectory)