import metaworld
import numpy as np
import gym
import random


class ML45(gym.Env):
    def __init__(self):
        super(ML45, self).__init__()
        self._benchmark = metaworld.ML45()
        self._kind = 'train'
        self._classes = self._benchmark.train_classes
        self._tasks = self._benchmark.train_tasks
        
        self.current_env = None
        self.current_task = None
        self.obs = None
        
        self.observation_space = None
        self.action_space = None
        self._max_episode_steps = 500
        
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
        obs = self.current_env.reset()
        return obs
    
    def render(self):
        self.current_env.render()
        

    def get_task(self):
        return 0

    def reset_task(self, task=None):
        """
        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment.
        """
        self.current_task = random.choice(self._tasks)
        env_cls = self._classes[self.current_task.env_name]
        self.current_env = env_cls()
        self.current_env.set_task(self.current_task)
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space
        

if __name__ == '__main__':
    env = ML45()
    for i in range(100):
        env.reset_task()
        epi_return =0
        for step in range(500):
            a = env.action_space.sample()
            # print(env.action_space)
            obs, reward, done, info = env.step(a)
            epi_return += reward
        print(epi_return)
        