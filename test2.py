import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import json
import gym
import argparse

from environments.parallel_envs import make_vec_envs, make_env
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_policy(load_path, iter):
    with open(os.path.join(load_path, "config.json"),'r', encoding='UTF-8') as f:
        data_dict = json.load(f)
    args = argparse.ArgumentParser()
    for key, val in data_dict.items():
        args.add_argument("-" + key, default=val)
        
    args = args.parse_args()
    
    policy = torch.load(os.path.join(load_path, 'models/policy{}.pt'.format(iter)))
    # encoder = torch.load(os.path.join(load_path, 'models/encoder{}.pt'.format(iter)))
    encoder = None

    # reset environment
    ego_state = np.array([-0.4, 0, 0, -0.3, 0.00105598, 0])

    ref_state1 = np.array([0] * 20)
    ref_state2 = np.array([i * 0.05 for i in range(20)])
    ref_stata3 = np.array([0] * 20)

    state = np.concatenate((ego_state, ref_state1, ref_state2, ref_stata3)).astype(np.float32)
    sate = np.expand_dims(state, 0)
    state = torch.from_numpy(state).float().to(device).unsqueeze(0)
    # state = np.concatenate([state, [0.0]], axis=0)

    if encoder is not None:
        # reset latent state to prior
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(1)
    else:
        latent_sample = latent_mean = latent_logvar = hidden_state = None



    with torch.no_grad():
        _, action = utl.select_action(args=args,
                                        policy=policy,
                                        state=state,
                                        belief=None,
                                        task=None,
                                        latent_sample=latent_sample,
                                        latent_mean=latent_mean,
                                        latent_logvar=latent_logvar,
                                        deterministic=True)

    print(action)
    
    # print(test_return, devi_p, devi_v, devi_phi)
    # print(test_return)
    # print('-' * 100)


def main():
    load_path = "./logs/logs_MultiGoalEnv-v0/varibad_74__10:01_10:50:49"
    iter = 15499
    test_policy(load_path, iter)


if __name__ == "__main__":
    main()