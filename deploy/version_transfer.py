import torch
import os
import torch.nn as nn
from models.policy import Policy
from models.encoder import RNNEncoder
import numpy as np
import argparse
import json

def convert(load_path, iter):
    # Load trained policy
    policy_path = ''
    encoder_path = ''
    with open(os.path.join(load_path, "config.json"),'r', encoding='UTF-8') as f:
        data_dict = json.load(f)

    args = argparse.ArgumentParser()

    for key, val in data_dict.items():
        args.add_argument("-" + key, default=val)
        args = args.parse_args()

    policy = Policy(
            args=args,
            #
            pass_state_to_policy=args.pass_state_to_policy,
            pass_latent_to_policy=args.pass_latent_to_policy,
            pass_belief_to_policy=args.pass_belief_to_policy,
            pass_task_to_policy=args.pass_task_to_policy,
            dim_state=args.state_dim,
            dim_latent=args.latent_dim * 2 if args.use_dist_latent else args.latent_dim,
            dim_belief=args.belief_dim,
            dim_task=args.task_dim,
            #
            hidden_layers=args.policy_layers,
            activation_function=args.policy_activation_function,
            policy_initialisation=args.policy_initialisation,
            #
            action_space=np.one(2),
            init_std=args.policy_init_std,
        )
    
    encoder = RNNEncoder(
            args=args,
            layers_before_gru=args.encoder_layers_before_gru,
            hidden_size=args.encoder_gru_hidden_size,
            layers_after_gru=args.encoder_layers_after_gru,
            latent_dim=args.latent_dim,
            action_dim=args.action_dim,
            action_embed_dim=args.action_embedding_size,
            state_dim=args.state_dim,
            state_embed_dim=args.state_embedding_size,
            reward_size=1,
            reward_embed_size=args.reward_embedding_size,
        )
    
    policy.load_state_dict(torch.load(policy_path))
    encoder.load_state_dict(torch.load(encoder_path))
    
def cal_action(encoder, policy, args):
    latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(1)

