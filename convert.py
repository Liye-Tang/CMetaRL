import os
import time
import shutil

import gym
import numpy as np
import torch
import json
import utils.gol as gol
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

gol._init()
gol.set_value('device', 'cpu')

from algorithms.a2c import A2C
from algorithms.online_storage import OnlineStorage
from algorithms.ppo import PPO
from environments.parallel_envs import make_vec_envs, make_env
from models.policy import Policy
from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.tb_logger import TBLogger
from vae import VaribadVAE


import torch.nn.functional as F

device = 'cpu'
load_path = "./logs/logs_MobileGoalClusterEnv-v0/cluster_73__08:09_15:45:47"
iter = '19999'
with open(os.path.join(load_path, "config.json"),'r', encoding='UTF-8') as f:
    data_dict = json.load(f)
args = argparse.ArgumentParser()
for key, val in data_dict.items():
    args.add_argument("-" + key, default=val)
    
args = args.parse_args()

# if device == "cpu":
#     policy = torch.load(os.path.join(load_path, 'models/policy{}.pt'.format(iter)))
# else:
policy = torch.load(os.path.join(load_path, 'models/policy{}.pt'.format(iter)), map_location=torch.device('cpu'))

# if args.disable_metalearner:
#     encoder = None
# else:
#     if device == "cpu":
#         encoder = torch.load(os.path.join(load_path, 'models/encoder{}.pt'.format(iter)))
#     else:
encoder = torch.load(os.path.join(load_path, 'models/encoder{}.pt'.format(iter)), map_location=torch.device('cpu'))

# if device == "cpu":
#     proto_proj = torch.load(os.path.join(load_path, 'models/proto_proj{}.pt'.format(iter)))
# else:
#     proto_proj = torch.load(os.path.join(load_path, 'models/proto_proj{}.pt'.format(iter)), map_location=torch.device('cpu'))

policy.eval()
encoder.eval()

legend = 'cluster_0809_154547'
os.makedirs('./weights/{}'.format(legend), exist_ok=True)
torch.save(policy.state_dict(), './weights/{}/policy.pkl'.format(legend), _use_new_zipfile_serialization=False)
torch.save(encoder.state_dict(), './weights/{}/encoder.pkl'.format(legend), _use_new_zipfile_serialization=False)
utl.save_obj(policy.state_rms, './weights/{}'.format(legend), f"state_rms")
utl.save_obj(policy.latent_rms, './weights/{}'.format(legend), f"latent_rms")
shutil.copyfile(os.path.join(load_path, "config.json"), './weights/{}/config.json'.format(legend))