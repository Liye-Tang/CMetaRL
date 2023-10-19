import random

import numpy as np

from environments.mujoco.ant import AntEnv


class AntGoalClusterEnv(AntEnv):
    def __init__(self, max_episode_steps=200):
        self.seed()
        self.num_cls = 16
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        super(AntGoalClusterEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        # goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal
        goal_reward = -np.sum(np.square(xposafter[:2] - self.goal_pos))  # test the square reward

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            task=self.get_task()
        )

    def sample_tasks(self, num_tasks):
        task_clses = [self.np_random.randint(0, self.num_cls) for _ in range(num_tasks)]
        self.task_cls = task_clses[0]
        a = np.array([self.sample_task_per_cls(task_cls) for task_cls in task_clses])
        
        r_list = [4.4, 3.5, 5.1, 3.0, 4.2, 3.4, 4.7, 3.9, 4.4, 3.5, 4.4, 3.0, 4.5, 3.4, 4.4, 3.8, 4.4, 3.9, 5.1, 3.0, 4.2, 3.4, 4.7, 3.9, 4.4, 3.5, 4.6, 3.2, 4.2, 5.0, 4.5, 3.9] 
        r = [r_list[task_cls] for task_cls in task_clses]
        tasks = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1) + self.np_random.uniform(-0.1, 0.1, (num_tasks, 2))

        return tasks
    
    def sample_task_per_cls(self, task_cls):
        orig_dir_list = [0, np.pi/7, np.pi/4, np.pi/3.5, np.pi/2.3]
        a = orig_dir_list[task_cls%4] + np.pi / 2 * (task_cls//4)
        # a = task_cls * np.pi * 2 / self.num_cls + \
        # self.np_random.uniform(-np.pi * 0.05 / self.num_cls, np.pi * 0.05 / self.num_cls)

        # if task_cls == 0:
        #     a = self.np_random.uniform(1/16, 3/16) * 2 * np.pi
        # elif task_cls == 1:
        #     a = self.np_random.uniform(5/16, 7/16) * 2 * np.pi
        # elif task_cls == 2:
        #     a = self.np_random.uniform(9/16, 11/16) * 2 * np.pi
        # else:
        #     a = self.np_random.uniform(13/16, 15/16) * 2 * np.pi 
        return a
        
    def get_task_cls(self):
        return self.task_cls
    
    def set_task(self, task):
        self.goal_pos = task

    def get_task(self):
        return np.array(self.goal_pos)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])


class AntGoalOracleEnv(AntGoalClusterEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.goal_pos,
        ])


def test_env():
    env = AntGoalClusterEnv()
    for i in range(10):
        state = env.reset()
        task = env.reset_task()
        print(state, task)


if __name__ == '__main__':
    test_env()
