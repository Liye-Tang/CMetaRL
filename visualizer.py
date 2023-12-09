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
        # self.iter = ''  # TODO: only for test
        
        if self.args.disable_cluster and not self.args.disable_kl_term:
            self.res_dir = './test_results/variabd/{}'.format(self.load_path[-14:])
        elif self.args.disable_cluster and self.args.disable_kl_term:
            self.res_dir = './test_results/none/{}'.format(self.load_path[-14:])
        elif not self.args.disable_cluster and self.args.disable_kl_term:
            self.res_dir = './test_results/cluster/{}'.format(self.load_path[-14:])
        
        os.makedirs(self.res_dir, exist_ok=True)
        
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
    
    def run_an_episode(self, num_steps, render=False, task=None, episode_ID=1, plot=False):
        context_list = []
        
        # save the key states and actions
        tar = np.zeros(num_steps+1)
        delta_v = np.zeros(num_steps)
        delta_w = np.zeros(num_steps)
        
        delta_phi = np.zeros(num_steps)
        delta_pos = np.zeros(num_steps)
        
        latent_cls = np.zeros(num_steps)
        
        # reset the environment
        state = self.env.reset(task=task)
        self.env.render_init(1)
        target_pos = self.env.goal_pos
        
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
        
        # run the episode and save the results
        for step_idx in range(num_steps):
            # singe cycle
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
            
            state, rew_raw, done, info = self.env.step(action.cpu().numpy()[0])
            # state = state + np.concatenate((np.random.random(5) * 0.1, np.array([0])))
            
            if render:
                self.env.render(path=self.res_dir)
                if step_idx == 199:
                    print('ok')
            
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
            
            # save the key info
            tar[step_idx+1] = tar[step_idx] + rew_raw[0]
            delta_v[step_idx] = action.cpu().numpy()[0][0]
            delta_w[step_idx] = action.cpu().numpy()[0][1]
            
            delta_phi[step_idx] = dot_product_angle(state[0][:2], target_pos)
            delta_pos[step_idx] = np.sum(np.square(state.numpy()[0][:2] - target_pos))
            
            if not self.args.disable_cluster:
                latent_cls[step_idx] = self.get_context_cls_list([latent_mean.numpy()])[0]
            
            if done:
                break
        if plot:
            # plot the key figures
            t = np.arange(num_steps)
            # plt.figure(figsize=(16, 8))
            ax = plt.gca()
            # x_major_locator = plt.MultipleLocator(1)
            # y_major_locator = plt.MultipleLocator(10)
            # ax.xaxis.set_major_locator(x_major_locator)
            # ax.yaxis.set_major_locator(y_major_locator)

            plt.figure()
            plt.minorticks_on()
            plt.plot(t, tar[1:])
            plt.text(10, 10, "{}".format(tar[-1]))
            plt.title('tar')
            plt.savefig('{}/tar{}.jpg'.format(self.res_dir, episode_ID), dpi=300) 
            
            plt.figure()
            plt.minorticks_on()
            plt.plot(t, delta_v)
            plt.title('delta_v')
            plt.savefig('{}/delta_v{}.jpg'.format(self.res_dir, episode_ID), dpi=300) 

            plt.figure()
            plt.minorticks_on()
            plt.plot(t, delta_w)
            plt.title('delta_w')
            plt.savefig('{}/delta_w{}.jpg'.format(self.res_dir, episode_ID), dpi=300) 

            plt.figure()
            plt.minorticks_on()
            plt.plot(t, delta_phi)
            plt.title('delta_phi')
            plt.savefig('{}/delta_phi{}.jpg'.format(self.res_dir, episode_ID), dpi=300) 
            
            plt.figure()
            plt.minorticks_on()
            plt.plot(t, delta_pos)
            plt.title('delta_pos')
            plt.savefig('{}/delta_pos{}.jpg'.format(self.res_dir, episode_ID), dpi=300) 
            
            
            plt.figure()
            plt.minorticks_on()
            plt.plot(t, latent_cls)
            plt.title('latent_cls')
            plt.savefig('{}/latent_cls{}.jpg'.format(self.res_dir, episode_ID), dpi=300) 
            
            # get the video
            with imageio.get_writer(uri='{}/test{}.gif'.format(self.res_dir, episode_ID), mode='I', fps=10) as writer:
                for i in range(200):
                    writer.append_data(imageio.imread('{}/{}.jpg'.format(self.res_dir, i)))
                    os.remove('{}/{}.jpg'.format(self.res_dir, i))

        return context_list, 0

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


def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)
        return angle
    return 0
        
def main():
    # the policy and configuration path
    iter = 29999
    # load the three tests
    load_path = "logs/logs_MobileGoalClusterEnv-v0/cluster_173__24:10_19:25:51"
    cluster_test = TestPolicy(load_path, iter)
    
    load_path = 'logs/logs_MobileGoalClusterEnv-v0/cluster_173__07:10_12:23:49'
    none_test = TestPolicy(load_path, iter)
    
    load_path = 'logs/logs_MobileGoalClusterEnv-v0/cluster_173__07:10_12:22:59'
    varibad_test = TestPolicy(load_path, iter)

    r = 8
    a_list = [np.pi / 6 * i for i in range(12)]
    goal_pos_list = [np.stack((r * np.cos(a), r * np.sin(a)), axis=-1) for a in a_list]

    for i, goal_pos in enumerate(goal_pos_list):
        # if i == 5:
        cluster_test.run_an_episode(num_steps=200, task=np.expand_dims(goal_pos, axis=0), episode_ID=i, render=True, plot=True)
        # none_test.run_an_episode(num_steps=200, task=np.expand_dims(goal_pos, axis=0), episode_ID=i)
        # varibad_test.run_an_episode(num_steps=200, task=np.expand_dims(goal_pos, axis=0), episode_ID=i)


def plot_cls():
    # the policy and configuration path
    iter = 29999
    # load the three tests
    load_path = "logs/logs_MobileGoalClusterEnv-v0/cluster_173__07:10_12:24:31"
    test = TestPolicy(load_path, iter)
    test.visualize_context()


if __name__ == "__main__":
    main()