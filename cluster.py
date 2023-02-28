import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Cluster:
    def __init__(self, args, encoder, rollout_storage, logger, get_iter_idx):
        self.args = args
        self.latent_dim = self.args.latent_dim  # the input dim: D
        self.num_prototypes = self.args.num_prototypes  # the number of the prototypes: K
        self.encoder = encoder
        self.logger = logger
        self.rollout_storage = rollout_storage

        # warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
        # cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
        #                     math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
        # self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.iters = 0
        self.get_iter_idx = get_iter_idx


        # self.projection_head = nn.Sequential(
        #     nn.Linear(num_out_filters * block.expansion, hidden_mlp),
        #     nn.BatchNorm1d(hidden_mlp),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_mlp, output_dim),
        # )

        # protomatrix C = [c1, c2, ..., ck]
        # the weight matrix is C [D, K]
        self.proto_proj = nn.Linear(in_features=self.latent_dim, out_features=self.num_prototypes, bias=False).to(device)
        # self.optimiser_proto = torch.optim.Adam([*self.proto_proj.parameters()], lr=self.args.lr_proto)
        # self.optimiser_cluster = torch.optim.Adam([*self.encoder.parameters()], lr=self.args.lr_cluster)
        self.optimiser_cluster = torch.optim.Adam([*self.proto_proj.parameters(), *self.encoder.parameters()], lr=self.args.lr_cluster)

    def compute_cluster_loss(self, update=False):

        with torch.no_grad():
            w = self.proto_proj.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.proto_proj.weight.copy_(w)

        # get a mini-batch
        vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, \
        trajectory_lens = self.rollout_storage.get_batch(batchsize=self.args.cluster_batch_num_trajs)
        # vae_prev_obs will be of size: max trajectory len x num trajectories x dimension of observations

        # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
        _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                        states=vae_next_obs,
                                                        rewards=vae_rewards,
                                                        hidden_state=None,
                                                        return_prior=False,
                                                        detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None,
                                                        )
        latent_mean = latent_mean[5:]
        latent = latent_mean.reshape(-1, self.latent_dim)

        embedding = F.normalize(latent, dim=1, p=2)
        scores = self.proto_proj(embedding)
        q = self.sinkhorn(scores)
        
        cluster_loss = 0
        cluster_loss -= torch.mean(torch.sum(q * F.log_softmax(scores / self.args.temperature, dim=1), dim=1))

        if update:
            self.optimiser_cluster.zero_grad()
            cluster_loss.backward()

            # clip gradients
            if self.args.proto_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.proto_proj.parameters(), self.args.proto_max_grad_norm)
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)

            # update
            self.optimiser_cluster.step()
        
        if self.get_iter_idx() % 100 == 0:
            print(self.proto_proj.weight.data.clone())
        
        self.log(cluster_loss)
    
    @torch.no_grad()
    def sinkhorn(self, scores):
        Q = torch.exp(scores / self.args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.args.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            # dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
    
    def log(self, cluster_loss):
        curr_iter_idx = self.get_iter_idx()

        if curr_iter_idx % self.args.log_interval == 0:

            self.logger.add('cluster_losses/cluster', cluster_loss, curr_iter_idx)