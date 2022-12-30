# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "SAC"
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
interm_dir = FILE_PATH + "/logs/intermediate_models/"
log_dir = interm_dir + '122922223535'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False
env_config['motor_control_mode'] = 'CPG'
env_config['competition_env'] = False

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time
env.move_reverse = False

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
NUM_STEPS = 10000
TIME_STEP = 0.001
t = np.arange(NUM_STEPS)*TIME_STEP

# initialize matrices for the plots
XYZ_base = np.zeros((3,NUM_STEPS))
torques = np.zeros((12, NUM_STEPS))
foot_pos = np.zeros((12, NUM_STEPS))
CPG_r = np.zeros((4, NUM_STEPS))
CPG_theta = np.zeros((4, NUM_STEPS))
RollPitchYaw = np.zeros((3, NUM_STEPS))

for i in range(NUM_STEPS):
    action, _states = model.predict(obs, deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0
        #break

    # [TODO] save data from current robot states for plots
    XYZ_base[:,i] = env.envs[0].robot.GetBasePosition()
    torques[:,i] = env.envs[0].robot.GetMotorTorques()
    foot_pos[0:3,i] = env.envs[0].robot.ComputeJacobianAndPosition(0)[1]
    foot_pos[3:6,i] = env.envs[0].robot.ComputeJacobianAndPosition(1)[1]
    foot_pos[6:9,i] = env.envs[0].robot.ComputeJacobianAndPosition(2)[1]
    foot_pos[9:12,i] = env.envs[0].robot.ComputeJacobianAndPosition(3)[1]
    CPG_r[:,i] = env.envs[0].get_cpg_r()
    CPG_theta[:,i] = env.envs[0].get_cpg_theta()
    RollPitchYaw[:,i] = env.envs[0].robot.GetBaseOrientationRollPitchYaw()
    
    
# [TODO] make plots:
fig = plt.figure()
plt.plot(t, XYZ_base[0,:])
plt.show()