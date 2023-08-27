from gym.envs.registration import register

# Mujoco
# ----------------------------------------

# - randomised reward functions

register(
    'AntDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDirCluster-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir_cluster:AntDirClusterEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDir2D-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DEnv',
            'max_episode_steps': 200},
    max_episode_steps=200,
)

register(
    'AntGoal-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntGoalCluster-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal_cluster:AntGoalClusterEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVelCluster-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel_cluster:HalfCheetahVelClusterEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HumanoidDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.humanoid_dir:HumanoidDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

# - randomised dynamics

register(
    id='Walker2DRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
    max_episode_steps=200
)

register(
    id='HopperRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
    max_episode_steps=200
)

register(
    id='Walker2DClusterParams-v0',
    entry_point='environments.mujoco.rand_param_envs.walker2d_cluster_params:Walker2DClusterParamsEnv',
    max_episode_steps=200
)

register(
    id='HopperClusterParams-v0',
    entry_point='environments.mujoco.rand_param_envs.hopper_cluster_params:HopperClusterParamsEnv',
    max_episode_steps=200
)

# # 2D Navigation
# # ----------------------------------------
#
register(
    'PointEnv-v0',
    entry_point='environments.navigation.point_robot:PointEnv',
    kwargs={'goal_radius': 0.2,
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)

register(
    'SparsePointEnv-v0',
    entry_point='environments.navigation.point_robot:SparsePointEnv',
    kwargs={'goal_radius': 0.2,
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)

#
# # GridWorld
# # ----------------------------------------

register(
    'GridNavi-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 5, 'num_steps': 15},
)

#
# # vehicle path following task
# # ----------------------------------------

register(
    'MultiGoalEnv-v0',
    entry_point='environments.pathfollow.multigoal:MultiGoalEnv',
    kwargs={},
    max_episode_steps=200
)

register(
    'MultiGoalClusterEnv-v0',
    entry_point='environments.pathfollow.multigoal_cluster:MultiGoalClusterEnv',
    kwargs={},
    max_episode_steps=200
)


register(
    'MultiParamEnv-v0',
    entry_point='environments.veh_param.Multiparam:MultiParamEnv',
    kwargs={},
    max_episode_steps=200
)

register(
    'ML45-v0',
    entry_point='environments.meta_world.ml45:ML45',
    kwargs={},
    max_episode_steps=500
)

register(
    'ML10-v0',
    entry_point='environments.meta_world.ml10:ML10',
    kwargs={},
    max_episode_steps=500
)

register(
    'ML1-v0',
    entry_point='environments.meta_world.ml1:ML1',
    kwargs={},
    max_episode_steps=500
)

register(
    'MobileDirClusterEnv-v0',
    entry_point='environments.mujoco.mobile_robot_dir_cluster:MobileRobotDir',
    kwargs={},
    max_episode_steps=200
)

register(
    'MobileGoalClusterEnv-v0',
    entry_point='environments.mujoco.mobile_robot_goal_cluster:MobileRobotGoal',
    kwargs={},
    max_episode_steps=200
)