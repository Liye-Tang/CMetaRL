#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Mobile Robot Environment
#  Update Date: 2022-06-05, Baiyu Peng: create environment
import random
from typing import Any, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

gym.logger.setLevel(gym.logger.ERROR)

class MobileRobotDir(gym.Env):

    def __init__(
            self, **kwargs: Any,
    ):
        self.max_episode_steps = 200

        self.robot = Robot()
        self.dt = 0.2
        self.exp_v = 0.35
        self.state_dim = 5
        self.action_dim = 2
        
        lb_state = np.array([-30, -30, -np.pi/4, -1, -np.pi / 2])
        hb_state = np.array([30, 30, np.pi/4, 1, np.pi / 2])
        
        frequency = 1 / self.dt
        lb_action = np.array([-1.8, -0.8]) / frequency
        hb_action = np.array([1.8, 0.8]) / frequency

        self.action_space = spaces.Box(low=lb_action, high=hb_action)
        self.observation_space = spaces.Box(lb_state, hb_state)
        
        self.seed()
        self._state = self.reset()

        self.goal_direction = None

        self.num_cls = 4
        self.task_cls = None
        self.reset_task()

        self.steps = 0
        self._max_episode_steps = 200

    @property
    def state(self):
        return self._state.reshape(-1)#[:5]

    def reset(self, n_agent=1, init_state: list = None, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        def uniform(low, high):
            return np.random.random([n_agent]) * (high - low) + low

        state = np.zeros([n_agent, self.state_dim])
        
        #state[:, 0] = uniform(0.75, 1.25)  # x坐标
        state[:, 0] = 1  # x坐标
        #state[:, 1] = uniform(-0.7, -0.3)  # y坐标
        state[:, 1] = -0.5  # y坐标
        #state[:, 2] = uniform(-0.2, 0.2)  # 朝向
        state[:, 2] = 0  # 朝向
        state[:, 3] = 0  # 速度
        #state[:, 4] = uniform(-np.pi/18,np.pi/18)  # 角速度
        state[:, 4] = 0  # 角速度

        # 重置状态变量
        self.steps_beyond_done = None
        self.steps = 0
        self._state = state.reshape(n_agent , self.state_dim)
        # 获取状态输入并返回
        # 将车辆状态信息和障碍物信息等按照一定顺序拼接成状态输入，并通过get_state_input方法获得其它所需处理的状态信息
        observation = np.array(self._state).reshape(1, -1)
        return np.array(observation).reshape(-1)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, dict]:
        #  define your forward function here: the format is just like: state_next = f(state,action)
        action = action.reshape(1, -1)
        
        robot_state = self.robot.f_xu(
            self._state[:, :5], action.reshape(1, -1), self.dt, "ego"
        )
        
        goal_direction = (np.cos(self.goal_direction), np.sin(self.goal_direction))
        robot_direction = (np.cos(self._state[:, 2]), np.sin(self._state[:, 2]))
        direction_reward = np.dot(goal_direction, robot_direction)
        velocity_error = self._state[:, 3] - self.exp_v

        self._state = robot_state

        # the absolute obs x is not taken as the obs 
        observation = np.array(self._state.reshape(1,-1))
        # 定义奖励函数
        r_tracking = (5
                      + 6.5 * np.square(direction_reward)    # 目标朝向角度误差
                      - 7 * np.square(velocity_error)   # 期望速度误差
                      - 3 * np.square(self._state[:,4])     # 横摆惩罚
                      )
        # r_action = -0.5 * np.square(action[:, 0]) - 0.5 * np.square(action[:, 1])
        r_action = -1 * np.square(action[:, 0]) - 1 * np.square(action[:, 1])

        reward = r_tracking + r_action

        isdone = self.get_done()
        # if isdone:
        #     reward = -415
        self.info = dict(r_tracking=direction_reward, r_action=r_action)
        self.steps += 1
        return np.array(observation.reshape(-1), dtype=np.float32), float(reward), isdone, {}

    def get_done(self) -> np.ndarray:
        # done = self._state[:, 0] < -2 \
        #        or self._state[:, 1] > 4 \
        #        or self._state[:, 1] < -4 \
        #        or self._state[:, 2] < -np.pi/2 \
        #        or self._state[:, 2] > np.pi/2
        # done = True if self.steps > self._max_episode_steps - 1 else False
        done = False
        return done

    def render(self, mode: str = "human", n_window: int = 1):

        if not hasattr(self, "artists"):
            self.render_init(n_window)
        state = self._state
        r_rob = self.robot.robot_params["radius"]
        # r_obs = self.obses[0].robot_params["radius"]

        def arrow_pos(state):
            x, y, theta = state[0], state[1], state[2]
            return [x, x + np.cos(theta) * r_rob], [y, y + np.sin(theta) * r_rob]
        
        for i in range(n_window):
            for j in range(n_window):
                idx = i * n_window + j
                circles, arrows, texts = self.artists[idx]
                circles[0].center = state[idx, :2]
                arrows[0].set_data(arrow_pos(state[idx, :5]))
                texts[0].set_text('r_t:{}, r_a:{}'.format(self.info['r_tracking'], self.info['r_action']))
            plt.pause(0.025)

    def render_init(self, n_window: int = 1):

        fig, axs = plt.subplots(n_window, n_window, figsize=(9, 9))
        artists = []

        r_rob = self.robot.robot_params["radius"]
        # r_obs = self.obses[0].robot_params["radius"]
        for i in range(n_window):
            for j in range(n_window):
                if n_window == 1:
                    ax = axs
                else:
                    ax = axs[i, j]
                ax.set_aspect(1)
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)

                # plot the direction of the robot
                x = np.linspace(0, 6, 1000)
                y = np.tan(self.goal_direction) * x
                ax.plot(x, y, color="black", linestyle='dashed')
                circles = []
                arrows = []
                texts = []
                circles.append(plt.Circle([0, 0], r_rob, color="red", fill=False))
                arrows.append(ax.plot([], [], "red")[0])
                texts.append(ax.text(4, 1, ''))
                ax.add_artist(circles[-1])
                ax.add_artist(arrows[-1])
                ax.add_artist(texts[-1])
                # for k in range(self.n_obstacle):
                #     circles.append(plt.Circle([0, 0], r_obs, color="blue", fill=False))
                #     ax.add_artist(circles[-1])

                #     arrows.append(ax.plot([], [], "blue")[0])
                artists.append([circles, arrows, texts])
        self.artists = artists
        plt.ion()

    def close(self):
        plt.close("all")

    def reset_task(self, task=None):
        if task is not None:
            self.set_task(task)
        else:
            self.task_cls = random.randint(0, self.num_cls - 1)
            self.goal_direction = self.sample_task_per_cls(self.task_cls)
    
    def sample_task_per_cls(self, task_cls):
        if task_cls == 0:
            a = random.uniform(1/16, 3/16) * 2 * np.pi
        elif task_cls == 1:
            a = random.uniform(5/16, 7/16) * 2 * np.pi
        elif task_cls == 2:
            a = random.uniform(9/16, 11/16) * 2 * np.pi
        else:
            a = random.uniform(13/16, 15/16) * 2 * np.pi 
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


class Robot:
    def __init__(self):
        self.robot_params = dict(
            v_max=1,
            w_max=np.pi / 2,
            v_delta_max=1.8,
            w_delta_max=0.8,
            v_desired=0.18,
            radius=0.74 / 2,  # per second
        )
        
        # self.path = ReferencePath()

    def f_xu(
            self, states: np.ndarray, actions: np.ndarray, T: float, type: str
    ) -> np.ndarray:
        v_delta_max = self.robot_params["v_delta_max"]
        v_max = self.robot_params["v_max"]
        w_max = self.robot_params["w_max"]
        w_delta_max = self.robot_params["w_delta_max"]
        std_type = {
            "ego": [0.0, 0.0],
            "obs": [0.03, 0.02],
            "none": [0, 0],
            "explore": [0.3, 0.3],
        }
        stds = std_type[type]

        x, y, theta, v, w = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
        )
        # v_cmd, w_cmd = actions[:, 0], actions[:, 1]
        #
        # delta_v = np.clip(v_cmd - v, -v_delta_max * T, v_delta_max * T)
        # delta_w = np.clip(w_cmd - w, -w_delta_max * T, w_delta_max * T)

        delta_v, delta_w = actions[:, 0], actions[:, 1]
        delta_v = np.clip(delta_v, -v_delta_max * T, v_delta_max * T)
        delta_w = np.clip(delta_w, -w_delta_max * T, w_delta_max * T)
        v_cmd = (
                np.clip(v + delta_v, 0, v_max)
                #+ np.random.normal(0, stds[0], [states.shape[0]]) * 0.5
        )

        w_cmd = (
                np.clip(w + delta_w, -w_max, w_max)
                #+ np.random.normal(0, stds[1], [states.shape[0]]) * 0.5
        )

        next_state = [
            x + T * np.cos(theta) * v_cmd,
            y + T * np.sin(theta) * v_cmd,
            np.clip(theta + T * w_cmd, -np.pi, np.pi),
            v_cmd,
            w_cmd,
        ]

        return np.stack(next_state, 1)


def main():
    env = MobileRobotDir()
    state = env.reset()

    for i in range(200):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        env.render()

    
    

if __name__ == '__main__':
    main()