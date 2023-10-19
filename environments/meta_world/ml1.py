import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
import numpy as np
import gym
import random


class ML1(gym.Env):
    def __init__(self):
        super(ML1, self).__init__()
        # env_list = ['push-v2', 'reach-v2', 'push-back-v2', 'pick-place-v2']
        env_list = ['pick-place-v2']  # only for the test
        self.env_list = [_env_dict.ALL_V2_ENVIRONMENTS[env_name] for env_name in env_list]

        self.observation_space = None
        self.action_space = None
        self._max_episode_steps = 500

        # self.current_env = None
        # self._benchmark = metaworld.ML1('reach-v2')
        # self._kind = 'train'
        # self._classes = self._benchmark.train_classes
        # self._tasks = self._benchmark.train_tasks
        # reset the the task and reset initial state
        self.reset_task()
        self.obs = self.reset()

    def step(self, action):
        self.obs, reward, done, info = self.current_env.step(action)
        return self.obs, reward, done, info

    def reset(self):
        """
        Reset the environment. This should *NOT* automatically reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """
        self.obs = self.current_env.reset()
        return self.obs
    
    def render(self):
        self.current_env.render()

    def get_task(self):
        return 0

    def reset_task(self, task=None):
        """
        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment.
        """
        self.current_env = random.choice(self.env_list)()
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space
        self.current_env.reset_task()
        

if __name__ == '__main__':
    env = ML1()
    for i in range(1):
        env.reset_task()
        env.reset()
        epi_return = 0
        for step in range(500):
            a = env.action_space.sample()
            # print(env.action_space)
            obs, reward, done, info = env.step(a)
            epi_return += reward
        print(epi_return)
        