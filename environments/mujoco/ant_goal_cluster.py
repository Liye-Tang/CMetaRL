import random

import numpy as np

from environments.mujoco.ant import AntEnv


class AntGoalClusterEnv(AntEnv):
    def __init__(self, max_episode_steps=200):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        super(AntGoalClusterEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal

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
        a = np.array([random.uniform(1/16, 3/16) for _ in range(num_tasks)]) * 2 * np.pi
        r = 3
        group1 = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        
        a = np.array([random.uniform(5/16, 7/16) for _ in range(num_tasks)]) * 2 * np.pi
        r = 3
        group2 = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        
        a = np.array([random.uniform(9/16, 11/16) for _ in range(num_tasks)]) * 2 * np.pi
        r = 3
        group3 = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        
        a = np.array([random.uniform(13/16, 15/16) for _ in range(num_tasks)]) * 2 * np.pi
        r = 3
        group4 = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        
        return random.choice([group1, group2, group3, group4])
    
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
