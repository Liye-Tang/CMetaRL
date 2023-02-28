import numpy as np
from dataclasses import dataclass
import matplotlib
from matplotlib.transforms import Affine2D
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import math

import argparse
from config.vehicle import args_veh_varibad


@dataclass
class Para:

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='vehicle_varibad')
    args, rest_args = parser.parse_known_args()
    args = args_veh_varibad.get_args(rest_args)

    # dim
    EGO_DIM: int = 6
    GOAL_DIM: int = 3
    N = args.N

    # reward hparam
    scale_devi_p: float = args.scale_devi_p
    scale_devi_v: float = args.scale_devi_v
    scale_devi_phi: float = args.scale_devi_phi
    scale_punish_yaw_rate: float = args.scale_punish_yaw_rate  # 0.1
    scale_punish_steer: float = args.scale_punish_steer  # 1
    scale_punish_a_x: float = args.scale_punish_a_x  # 0.1

    reward_shift: float = args.reward_shift

    # action scale factor
    ACC_SCALE: float = 3.0
    ACC_SHIFT: float = 1.0
    STEER_SCALE: float = 0.5
    STEER_SCALE: float = 0.5
    STEER_SHIFT: float = 0

    # done
    POS_TOLERANCE: float = 10.
    ANGLE_TOLERANCE: float = 60.

    # ego shape
    L: float = 4.8
    W: float = 2.

    MAX_STEPS = args.num_max_step

    # goal
    GOAL_X_LOW: float = -40.
    GOAL_X_UP: float = 40.
    GOAL_Y_LOW: float = 40.
    GOAL_Y_UP: float = 60.
    GOAL_PHI_LOW: float = 0.
    GOAL_PHI_UP: float = 180.

    # ref path
    METER_POINT_NUM: int = 30
    START_LENGTH: float = 5.
    END_LENGTH: float = 5.
    EXPECTED_V = args.EXPECTED_V

    # initial obs noise
    MU_X: float = 0
    SIGMA_X: float = 1
    MU_Y: float = 0
    SIGMA_Y: float = 1
    MU_PHI: float = 0
    SIGMA_PHI: float = 5

    # simulation settings
    FREQUENCY: float = 10


def cal_eu_dist(x1, y1, x2, y2):
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))


def action_denormalize(action_norm):
    action = np.clip(action_norm, -1.05, 1.05)
    steer_norm, a_x_norm = action[0], action[1]
    scaled_steer = Para.STEER_SCALE * steer_norm - Para.STEER_SHIFT
    scaled_acc = Para.ACC_SCALE * a_x_norm - Para.ACC_SHIFT
    scaled_action = np.array([scaled_steer, scaled_acc], dtype=np.float32)
    return scaled_action


def draw_rotate_rec(x, y, a, l, w):
    return matplotlib.patches.Rectangle((-l / 2 + x, -w / 2 + y),
                                        width=l, height=w,
                                        fill=False,
                                        facecolor=None,
                                        edgecolor='k',
                                        linewidth=1.0,
                                        transform=Affine2D().rotate_deg_around(*(x, y),
                                                                               a))


def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * math.pi / 180
    transformed_x = orig_x * math.cos(coordi_rotate_d_in_rad) + orig_y * math.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * math.sin(coordi_rotate_d_in_rad) + orig_y * math.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d - 90
    # if transformed_d > 180:
    #     while transformed_d > 180:
    #         transformed_d = transformed_d - 360
    # elif transformed_d <= -180:
    #     while transformed_d <= -180:
    #         transformed_d = transformed_d + 360
    # else:
    #     transformed_d = transformed_d
    return transformed_x, transformed_y, transformed_d


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param coordi_shift_x: coordi_shift_x along x axis
    :param coordi_shift_y: coordi_shift_y along y axis
    :return: shifted_x, shifted_y
    """
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def rotate_and_shift_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y, transformed_d \
        = rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d)
    transformed_x, transformed_y = shift_coordination(shift_x, shift_y, coordi_shift_x, coordi_shift_y)

    return transformed_x, transformed_y, transformed_d


def shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def deal_with_phi(phi):
    while phi > 360:
        phi -= 360
    while phi <= 0:
        phi += 360
    return phi