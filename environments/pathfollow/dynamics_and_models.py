import numpy as np
import tensorflow as tf
from environments.pathfollow.utils import *
import math
from environments.pathfollow.ref_path import ReferencePath


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-90000,  # front wheel cornering stiffness [N/rad]
                                   C_r=-90000,  # rear wheel cornering stiffness [N/rad]
                                   a=1.1,  # distance from CG to front axle [m]
                                   b=1.2,  # distance from CG to rear axle [m]
                                   mass=1200,  # mass [kg]
                                   I_z=1600,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=1.0,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
        v_x, v_y, r, x, y, phi = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
        phi = phi * np.pi / 180.
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
        C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
        a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
        b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
        mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
        I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
        miu = tf.convert_to_tensor(self.vehicle_params['miu'], dtype=tf.float32)
        g = tf.convert_to_tensor(self.vehicle_params['g'], dtype=tf.float32)

        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x))
        F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
        miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
        alpha_f = tf.atan((v_y + a * r) / (v_x + 1e-8)) - steer
        alpha_r = tf.atan((v_y - b * r) / (v_x + 1e-8))

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * tf.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (tf.square(a) * C_f + tf.square(b) * C_r) - I_z * v_x),
                      x + tau * (v_x * tf.cos(phi) - v_y * tf.sin(phi)),
                      y + tau * (v_x * tf.sin(phi) + v_y * tf.cos(phi)),
                      (phi + tau * r) * 180 / np.pi]

        return tf.stack(next_state, 1), tf.stack([alpha_f, alpha_r, miu_f, miu_r], 1)

    def prediction(self, x_1, u_1, frequency):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params


class EnvironmentModel(object):
    def __init__(self):
        self.ego_dim = Para.EGO_DIM
        self.goal_dim = Para.GOAL_DIM

    def compute_rewards(self, obses, actions, closest_points):
        obses_ego, obses_goal = self.split_all(obses)
        steers, a_xs = actions[:, 0], actions[:, 1]

        # rewards related to tracking error
        devi_p = -tf.cast(tf.square(obses_ego[:, 3]), tf.float32)
        devi_phi = -tf.cast(tf.square((obses_ego[:, 5]) * math.pi / 180), tf.float32)

        devi_v = -tf.cast(tf.square(obses_ego[:, 0]), tf.float32)

        # rewards related to ego stability
        punish_yaw_rate = -tf.cast(tf.square(obses_ego[:, 2]), tf.float32)

        # rewards related to action
        punish_steer = -tf.square(steers)
        punish_a_x = -tf.square(a_xs)

        rewards = Para.scale_devi_p * devi_p + \
                  Para.scale_devi_v * devi_v + \
                  Para.scale_devi_phi * devi_phi + \
                  Para.scale_punish_yaw_rate * punish_yaw_rate + \
                  Para.scale_punish_steer * punish_steer + \
                  Para.scale_punish_a_x * punish_a_x + 1

        reward_dict = dict(devi_p=devi_p,
                           devi_v=devi_v,
                           devi_phi=devi_phi,
                           punish_steer=punish_steer,
                           punish_a_x=punish_a_x,
                           punish_yaw_rate=punish_yaw_rate,
                           scaled_devi_p=Para.scale_devi_p * devi_p,
                           scaled_devi_v=Para.scale_devi_v * devi_v,
                           scaled_devi_phi=Para.scale_devi_phi * devi_phi,
                           scaled_punish_steer=Para.scale_punish_steer * punish_steer,
                           scaled_punish_a_x=Para.scale_punish_a_x * punish_a_x,
                           scaled_punish_yaw_rate=Para.scale_punish_yaw_rate * punish_yaw_rate,
                           )

        return rewards, reward_dict

    def split_all(self, obses):
        obses_ego = obses[:, :self.ego_dim]
        obses_goal = obses[:, self.ego_dim:self.ego_dim+self.goal_dim]

        return obses_ego, obses_goal
