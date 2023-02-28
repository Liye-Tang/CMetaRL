import gym
import numpy as np
from gym.utils import seeding

import matplotlib.pyplot as plt
import random


from environments.pathfollow.ref_path import ReferencePath
from environments.pathfollow.dynamics_and_models import VehicleDynamics, EnvironmentModel
from environments.pathfollow.utils import *


class MultiGoalEnv(gym.Env):
    def __init__(self,
                 goal_point=None,
                 start_point=(0, 0, 90),
                 ):
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [-np.inf] * (Para.EGO_DIM + Para.N * 3)
            ),
            high=np.array(
                [np.inf] * (Para.EGO_DIM + Para.N * 3)
            ),
            dtype=np.float32)   # TODO
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)    # TODO
        self.seed()

        self.obs = None
        self.action =None
        self.ego_state = None
        self.closest_point = None
        self.future_n_point = None
        self.area_index = None
        self.obs_scale = [1/10, 1/10, 1/10, 1/10, 1/10, 1/180] + [1/10] * Para.N + [1/10] * Para.N + [1/180] * Para.N
        self.task_dim = 3

        self.start_point = start_point
        self.goal_point = goal_point

        self._max_episode_steps = Para.MAX_STEPS

        self.done_type = None
        self.done = False

        self.env_model = EnvironmentModel()
        self.ego_dynamic = VehicleDynamics()
        plt.ion()

        if self.goal_point is None:
            self.reset_task()
            self.reset()
        else:
            self.ref_path = ReferencePath(self.goal_point)

    def step(self, action):
        info = {}
        self.action = action_denormalize(action)
        reward, reward_info = self.compute_reward(self.obs, self.action, self.closest_point)
        info.update(reward_info)
        self.ego_state, ego_param = self.get_next_ego_state(self.action)
        self.update_obs()
        self.done_type, self.done = self.judge_done()
        return self.obs * self.obs_scale, reward, self.done, info

    def reset(self):
        # self.generate_goal_point()
        self.generate_ego_state()
        # print(self.ego_state)
        self.update_obs()
        return self.obs * self.obs_scale

    def reset_task(self, task=None):
        if task is None:
            goal_x = random.uniform(Para.GOAL_X_LOW, Para.GOAL_X_UP)
            goal_y = random.uniform(Para.GOAL_Y_LOW, Para.GOAL_Y_UP)
            if goal_x > 0:
                goal_phi = random.uniform(Para.GOAL_PHI_LOW, 90)
            else:
                goal_phi = random.uniform(90, Para.GOAL_PHI_UP)
            self.goal_point = goal_x, goal_y, goal_phi
        else:
            self.goal_point = task
        self.ref_path = ReferencePath(self.goal_point)
        return self.goal_point

    def get_task(self):
        return self.goal_point

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    def compute_reward(self, obs, action, closest_point):
        obses, actions, closest_points = obs[np.newaxis, :], action[np.newaxis, :], closest_point[np.newaxis, :]
        reward, reward_dict = self.env_model.compute_rewards(obses, actions, closest_points)
        for k, v in reward_dict.items():
            reward_dict[k] = v.numpy()[0]
        # print('reward_dict', reward_dict)
        return reward.numpy()[0], reward_dict

    def get_next_ego_state(self, action):
        ego_states, actions = self.ego_state[np.newaxis, :], action[np.newaxis, :]
        next_ego_states, next_ego_params = self.ego_dynamic.prediction(ego_states, actions, Para.FREQUENCY)
        next_ego_state, next_ego_param = next_ego_states.numpy()[0], next_ego_params.numpy()[0]
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.
        # next_ego_state[-1] = deal_with_phi(next_ego_state[-1])
        return next_ego_state, next_ego_param

    def judge_done(self):
        """
        :return:
         1: good done: enter area 2
         2: bad done: deviate too much
         3: not done
        """
        if self._deviate_too_much():
            return 'deviate too much', 1
        # elif self.area_index == 2:
        #     return 'good done', 2
        else:
            return 'not_done', 0

    def update_obs(self):
        ego_pos = self.ego_state[-3:]
        closest_point, self.area_index, index = self.ref_path.find_closest_point(ego_pos)
        # print(self.area_index)
        self.closest_point = np.asarray(closest_point, dtype=np.float32)
        self.future_n_point = self.ref_path.get_n_future_point(self.closest_point, index, n=Para.N)
        # self.obs = np.concatenate((self.ego_state, self.goal_point), axis=0, dtype=np.float32)
        rela_ego_state = self.convert_to_rela(self.ego_state)
        rela_ref_x, rela_ref_y, rela_ref_phi = \
            shift_and_rotate_coordination(self.future_n_point[0], self.future_n_point[1], self.future_n_point[2],
                                          self.future_n_point[0][0], self.future_n_point[1][0], self.future_n_point[2][0] - 90)
        self.obs = np.concatenate(
            (rela_ego_state, rela_ref_x, rela_ref_y, rela_ref_phi),
            axis=0)

    def _deviate_too_much(self):
        # print(cal_eu_dist(
        #     self.ego_state[-3],
        #     self.ego_state[-2],
        #     self.closest_point[0],
        #     self.closest_point[1]
        # ))
        # print(abs(self.ego_state[-1] - self.closest_point[2]))
        return True if cal_eu_dist(
            self.ego_state[-3],
            self.ego_state[-2],
            self.closest_point[0],
            self.closest_point[1]
        ) > Para.POS_TOLERANCE or abs(self.ego_state[-1] - self.closest_point[2]) > Para.ANGLE_TOLERANCE else False

    def generate_ego_state(self):
        whole_ref_len = len(self.ref_path.whole_path[0])
        random_index = int(random.uniform(150, whole_ref_len))
        # random_index = int(random.uniform(150, 200))
        ref_x, ref_y, ref_phi, ref_v = self.ref_path.idx2whole(random_index)

        # add some noise
        ego_state = [0] * 6
        ego_state[3] = ref_x + np.clip(random.gauss(Para.MU_X, Para.SIGMA_X), -5, 5)
        ego_state[4] = ref_y + np.clip(random.gauss(Para.MU_Y, Para.SIGMA_Y), -5, 5)
        ego_state[5] = ref_phi + np.clip(random.gauss(Para.MU_PHI, Para.SIGMA_PHI), -30, 30)
        ego_state[0] = random.random() * ref_v
        ego_state[1] = 0
        ego_state[2] = random.random() * 0.2 - 0.1

        self.ego_state = np.array(ego_state, dtype=np.float32)

    def convert_to_rela(self, ego_state):
        ego_vx, _, _, ego_x, ego_y, ego_phi = ego_state
        rela_vx = ego_vx - self.ref_path.exp_v
        rela_x, rela_y, rela_phi = shift_and_rotate_coordination(ego_x, ego_y, ego_phi,
                                                                 self.closest_point[0],
                                                                 self.closest_point[1],
                                                                 self.closest_point[2] - 90,)
        return np.concatenate(([rela_vx], ego_state[1: 3],
                               [rela_x, rela_y, rela_phi]),
                              axis=0)

        # return np.concatenate((ego_state[:3], [rela_x, rela_y, rela_phi]), axis=0)

    def render(self, mode="human"):
        if mode == 'human':
            # basic render settings
            patches = []
            plt.clf()
            ax = plt.axes([0.05, 0.05, 0.9, 0.9])
            ax.axis('equal')

            # plot ref path
            self.ref_path.plot_path(ax)

            # plot ego vehicle
            patches.append(
                draw_rotate_rec(self.ego_state[-3], self.ego_state[-2], self.ego_state[-1], Para.L, Para.W)
            )

            # plot the close points
            ax.scatter(self.future_n_point[0], self.future_n_point[1], color='black')
            ax.scatter(self.closest_point[0], self.closest_point[1], color='blue')

            # plot the whole fig
            ax.add_collection(PatchCollection(patches, match_original=True))
            plt.show()
            plt.pause(0.001)


def test():
    import tensorflow as tf
    env = MultiGoalEnv()
    env_model = EnvironmentModel()
    i = 0
    while i < 1000:
        for j in range(1000):
            i += 1
            action = np.array([0, 0.6 + random.random() * 0.8], dtype=np.float32)  # random.rand(1)*0.1 - 0.05
            obs, reward, done, info = env.step(action)
            # print(done)
            obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
            # print(reward)
            # obses = tf.convert_to_tensor(np.tile(obs, (1, 1)), dtype=tf.float32)
            # ref_points = tf.convert_to_tensor(np.tile(info['future_n_point'], (1, 1, 1)), dtype=tf.float32)
            # actions = tf.convert_to_tensor(np.tile(actions, (1, 1)), dtype=tf.float32)
            # env_model.reset(obses, ref_points)
            env.render()
            # if j > 88:
            #     for i in range(25):
            #         obses, rewards, rewards4value, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, \
            #         veh2bike4real, veh2person4real, veh2speed4real = env_model.rollout_out(
            #             actions + tf.experimental.numpy.random.rand(2) * 0.05, i)
            #         env_model.render()
            if done:
                print(env.done_type)
                break
        env.reset()
        # env.render(weights=np.zeros(env.other_number,))


if __name__ == '__main__':
    test()
