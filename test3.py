import matplotlib.pyplot as plt
import numpy as np

from environments.parallel_envs import make_vec_envs, make_env
from utils import helpers as utl

import mujoco_py


def test_policy(load_path, iter):
    import os

    mj_path = mujoco_py.utils.discover_mujoco()
    xml_path = os.path.join(mj_path, 'model', 'arm26.xml')

    model = mujoco_py.load_model_from_path(xml_path)

    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    sim_state = sim.get_state()

    while True:
        sim.set_state(sim_state)

        for i in range(1000):
            if i < 2:
                sim.data.ctrl[:] = 0.0
            else:
                sim.data.ctrl[:] = -1.0
            sim.step()
            viewer.render()

        if os.getenv('TESTING') is not None:
            break
# [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
#   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
#   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
#  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
#  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
#  -2.22862221e-05]


def main():
    load_path = "./logs/logs_MultiGoalEnv-v0/varibad_74__11:01_05:50:13"
    iter = 12499
    for i in range(20):
        test_policy(load_path, iter)


if __name__ == "__main__":
    main()