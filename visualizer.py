import os
import time

import gym
import imageio
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

device_name = gol.get_value("device")
device = torch.device(device_name if torch.cuda.is_available() else "cpu")


class TestPolicy():
    def __init__(self,
                 load_path = None,
                 iter = None):
        self.load_path = load_path
        self.iter = iter
        
        self.task_color_list = list(mcolors.BASE_COLORS.keys()) * 3
        self.context_color_list = list(mcolors.TABLEAU_COLORS.keys()) * 3
        
        with open(os.path.join(self.load_path, "config.json"),'r', encoding='UTF-8') as f:
            data_dict = json.load(f)
        args = argparse.ArgumentParser()
        for key, val in data_dict.items():
            args.add_argument("-" + key, default=val)
            
        args = args.parse_args()
        self.env = make_env(env_id=args.env_name, seed=args.seed, rank=0,
                            episodes_per_task=args.max_rollouts_per_task,
                            tasks=None, add_done_info=None
                            )()
        self.args = args
        # self.iter = ''  #TODO: only for test
        
        if device == "cpu":
            self.policy = torch.load(os.path.join(self.load_path, 'models/policy{}.pt'.format(self.iter)))
        else:
            self.policy = torch.load(os.path.join(self.load_path, 'models/policy{}.pt'.format(self.iter)), map_location=torch.device('cpu'))
        
        if args.disable_metalearner:
            self.encoder = None
        else:
            if device == "cpu":
                self.encoder = torch.load(os.path.join(self.load_path, 'models/encoder{}.pt'.format(self.iter)))
            else:
                self.encoder = torch.load(os.path.join(self.load_path, 'models/encoder{}.pt'.format(self.iter)), map_location=torch.device('cpu'))
        
        if args.disable_cluster:
            self.proto_proj = None
        else:
            if device == "cpu":
                self.proto_proj = torch.load(os.path.join(self.load_path, 'models/proto_proj{}.pt'.format(self.iter)))
            else:
                self.proto_proj = torch.load(os.path.join(self.load_path, 'models/proto_proj{}.pt'.format(self.iter)), map_location=torch.device('cpu'))
  
    def cal_context_cls(self, context):
        latent = torch.from_numpy(context)
        # latent = latent.reshape(-1, self.latent_dim)
        embedding = F.normalize(latent, dim=-1, p=2)
        scores = self.proto_proj(embedding)
        log_prob = F.log_softmax(scores / self.args.temperature, dim=-1)
        context_cls = torch.max(log_prob, dim=-1)[1].unsqueeze(-1)
        return context_cls.numpy()[0]
    
    def get_context_cls_list(self, context_list):
        context_cls_list = [self.cal_context_cls(context) for context in context_list]
        return context_cls_list
    
    def run_an_episode(self, num_steps, render=False, task=None, episode_ID=1):
        context_list = []
        TAR = 0
        state = self.env.reset(task=task)
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)

        if self.encoder is not None:
            # reset latent state to prior
            latent_sample, latent_mean, latent_logvar, hidden_state = self.encoder.prior(1)
        else:
            latent_sample = latent_mean = latent_logvar = hidden_state = None

        latent_cls_prob = None
        if self.args.pass_latent_cls_to_policy:
            # latent = latent.reshape(-1, self.latent_dim)
            embedding = F.normalize(latent_mean, dim=-1, p=2)
            scores = self.proto_proj(embedding)
            latent_cls_prob = F.softmax(scores / self.args.temperature, dim=-1).squeeze(0)
            
        for step_idx in range(num_steps):
            with torch.no_grad():
                _, action = utl.select_action(args=self.args,
                                              policy=self.policy,
                                              state=state,
                                              belief=None,
                                              task=None,
                                              latent_sample=latent_sample,
                                              latent_mean=latent_mean,
                                              latent_logvar=latent_logvar,
                                              latent_cls_prob=latent_cls_prob,
                                              deterministic=True)
                # print(action)
                # action = torch.tensor([[0.1, 0.1]])

            state, rew_raw, done, info = self.env.step(action.cpu().numpy()[0])
            state = state + np.concatenate((np.random.random(5) * 0.1, np.array([0])))
            TAR += rew_raw
            # print(state[2])
            if render:
                self.env.render()
            
            state = torch.from_numpy(state).float().to(device).unsqueeze(0)
            rew_raw = torch.from_numpy(np.array([rew_raw])).float().to(device).unsqueeze(0)

            if self.encoder is not None:
                latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder=self.encoder,
                                                                                                next_obs=state,
                                                                                                action=action,
                                                                                                reward=rew_raw,
                                                                                                done=None,
                                                                                                hidden_state=hidden_state)
                latent_sample, latent_mean, latent_logvar, hidden_state = latent_sample.unsqueeze(0), latent_mean.unsqueeze(0), latent_logvar.unsqueeze(0), hidden_state.unsqueeze(0)
            
            if step_idx > 100:
                context_list.append(latent_mean[0, 0].numpy())
            if done:
                break
        # print(TAR)
        # with imageio.get_writer(uri='./test_results/test{}_n.gif'.format(episode_ID), mode='I', fps=10) as writer:
        #     for i in range(200):
        #         writer.append_data(imageio.imread('./test_results/{}.jpg'.format(i)))
        #         os.remove('./test_results/{}.jpg'.format(i))
        return context_list, TAR

    def run_n_episode(self, n):
        for i in range(n):
            self.run_an_episode()

    def get_context_and_cls_for_task(self, num_steps=200, start_step_idx=100, task=None):
        context_list, _ = self.run_an_episode(num_steps=num_steps)
        context_cls_list = self.get_context_cls_list(context_list) if not self.args.disable_cluster else None
        return context_list, context_cls_list
    
    def get_task_cls_and_context_cls(self, num_tasks=100):
        # sample some tasks [task1(array), task2(array), ...]
        task_list = []
        task_cls_list = []
        for i in range(num_tasks):
            self.env.reset_task(task=None)
            task_list.append(self.env.get_task())
            task_cls_list.append(self.env.get_task_cls())
        
        # get context list and cor cls list for per task, [[c11, c12, ...], [c21, c22, ...], ...], [[cls11, cls12, ...], [cls21, cls22, ...], ...]
        task_context_list = []
        task_context_cls_list = []
        for task in task_list:
            context_list, context_cls_list = self.get_context_and_cls_for_task(task=task)
            task_context_list.append(context_list)
            task_context_cls_list.append(context_cls_list)
        return task_list, task_cls_list, task_context_list, task_context_cls_list

    def visualize_context(self, num_tasks):
        task_list, task_cls_list, task_context_list, task_context_cls_list = self.get_task_cls_and_context_cls(num_tasks=num_tasks)
        total_context_list = []
        total_context_cls_list = []
        total_task_list = []
        for context_list in task_context_list:
            total_context_list += context_list
        for context_cls_list in task_context_cls_list:
            total_context_cls_list += context_cls_list
        for task_cls in task_cls_list:
            total_task_list += [task_cls] * 99
        total_task_color_list = [self.task_color_list[task_cls] for task_cls in total_task_list]
        total_context_color_list = [self.context_color_list[context_cls] for context_cls in total_context_cls_list]
        context = np.array(total_context_list)
        X_embedded = TSNE(n_components=2).fit_transform(context)
        plt.figure()
        plt.scatter(X_embedded[: ,0], X_embedded[:, 1], color=total_context_color_list)
        plt.savefig('test1.jpg')
        plt.figure()
        plt.scatter(X_embedded[: ,0], X_embedded[:, 1], color=total_task_color_list)
        plt.savefig('test2.jpg')
        
def main():
    # the policy and configuration path
    load_path = "logs/logs_MobileGoalClusterEnv-v0/cluster_173__07:10_12:24:31"
    iter = 29999

    test = TestPolicy(load_path, iter)
    
    # test.run_an_episode(num_steps=200, episode_ID=i)
    # tar_list = []
    # for i in range(1):
    #     _, tar = test.run_an_episode(200)
    #     tar_list.append(tar)
    # mean_tar = np.mean(tar_list)
    # print(mean_tar)

    
    # load_path = 'logs\cluster_73_1609_001007'     
    # iter = 29999
    # test = TestPolicy(load_path, iter)     
    # tar_list = []
    # for i in range(10):
    #     _, tar = test.run_an_episode(200)
    #     tar_list.append(tar)
    # mean_tar = np.mean(tar_list)
    # print(mean_tar)
    # # print(latent_sample, latent_mean, latent_logvar, hidden_state)
    # latent_sample, latent_mean, latent_logvar, hidden_state = test.encoder.prior(1)     
    # print(latent_mean)
    # test.run_an_episode(num_steps=200)
    test.visualize_context(num_tasks=100)
    # print(action)

if __name__ == "__main__":
    for i in range(10):
        main()