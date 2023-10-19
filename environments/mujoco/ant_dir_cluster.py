import random

import numpy as np

from environments.mujoco.ant import AntEnv


class AntDirClusterEnv(AntEnv):
    """
    Forward/backward ant direction environment
    """

    def __init__(self, max_episode_steps=200):
        self.seed()
        self.num_cls = 16
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        super(AntDirClusterEnv, self).__init__()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self.goal_direction), np.sin(self.goal_direction))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
            task=self.get_task()
        )

    def sample_tasks(self, num_tasks):
        task_clses = [self.np_random.randint(0, self.num_cls) for _ in range(num_tasks)]
        self.task_cls = task_clses[0]
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        a = np.array([self.sample_task_per_cls(task_cls) for task_cls in task_clses])
        return a
    
    def sample_task_per_cls(self, task_cls):
        a = task_cls * np.pi * 2 / self.num_cls + \
        self.np_random.uniform(-np.pi * 0.1 / self.num_cls, np.pi * 0.1 / self.num_cls)
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
        self.get_task_cls(task)
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_direction = task
        
    def get_task_cls(self, task):
        if task > 1/16 * 2 * np.pi and task < 3/16 * 2 * np.pi:
            self.task_cls = 0
        elif task > 5/16 * 2 * np.pi and task < 7/16 * 2 * np.pi:
            self.task_cls = 1
        elif task > 9/16 * 2 * np.pi and task < 11/16 * 2 * np.pi:
            self.task_cls = 2
        else:
            self.task_cls = 3

    def get_task(self):
        return np.array([self.goal_direction])


class AntDir2DEnv(AntDirClusterEnv):
    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        directions = np.array([self.np_random.gauss(mu=0, sigma=1) for _ in range(n_tasks * 2)]).reshape((n_tasks, 2))
        directions /= np.linalg.norm(directions, axis=1)[..., np.newaxis]
        return directions


class AntDirOracleEnv(AntDirClusterEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            [self.goal_direction],
        ])


class AntDir2DOracleEnv(AntDirClusterEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            [self.goal_direction],
        ])
