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

""" Run CPG """
import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv


ADD_CARTESIAN_PD = "both"
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=False,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="CPG",
                    add_noise=False,    # start in ideal conditions
                    move_reverse=False
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP, omega_swing=8*2*np.pi, omega_stance=2.5*2*np.pi)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
cpg_pos_states = np.zeros((TEST_STEPS,2,4))
cpg_speed_states = np.zeros((TEST_STEPS,2,4))
r = np.zeros(TEST_STEPS)
theta = np.zeros(TEST_STEPS)
foot_pos = np.zeros([3,TEST_STEPS])
des_foot_pos = np.zeros([1,TEST_STEPS])
leg_torques = np.zeros((TEST_STEPS,12))

############## Sample Gains
# joint PD gains
kp=np.array([480,480,480])
kd=np.array([6,6,6])
# Cartesian PD gains
kpCartesian = np.diag([8000]*3)
kdCartesian = np.diag([90]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()
  r = cpg.X[0,0]
  theta = cpg.X[1,0]
  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()
  des_joint_vel = np.zeros(3)

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    # leg_q = np.zeros(3) # [TODO] 
    leg_q = env.robot.ComputeInverseKinematics(i ,leg_xyz)
    # Add joint PD contribution to tau for leg i (Equation 4)
    # tau += np.zeros(3) # [TODO] 
    if ADD_CARTESIAN_PD == "joint" or ADD_CARTESIAN_PD == "both":
      tau += kp*(leg_q-q[i*3:i*3+3]) + kd*(des_joint_vel - dq[i*3:i*3+3])

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD == "cart" or ADD_CARTESIAN_PD == "both":
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      # [TODO] 
      J, p = env.robot.ComputeJacobianAndPosition(i)
      # Get current foot velocity in leg frame (Equation 2)
      # [TODO] 
      v = J @ dq[i*3:i*3+3]
      des_v = J @ des_joint_vel
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      # tau += np.zeros(3) # [TODO]
      tau += np.transpose(J) @ (kpCartesian@(leg_xyz-p) + kdCartesian@(des_v - v))

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

    # fill the matrices for the plots
    foot_pos[:,j] = env.robot.ComputeJacobianAndPosition(0)[1]
    des_foot_pos[:,j] = xs[0]

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action)

  # [TODO] save any CPG or robot states
  cpg_pos_states[j,:,:] = cpg.X 
  cpg_speed_states[j,:,:] = cpg.X_dot
  leg_torques[j,:] = action


##################################################### 
# PLOTS
#####################################################



fig, ax = plt.subplots(4, 2)

# make a plot with different y-axis using second axis object

for i in range(4):
  # one plot per leg
  a1 = ax[i][0].plot(t, cpg_pos_states[:, 0, i], label = 'r')
  #a2 = ax[i][0].plot(t, cpg_pos_states[:, 1, i], label = 'theta')
  #secax = ax.secondary_xaxis('right', functions=cpg_pos_states[:, 1, i])
  #secax.set_ylabel('angle [rad]')
  #a2 = ax2.plot(t, cpg_pos_states[:, 1, i], label = 'theta')
  a3 = ax[i][1].plot(t, cpg_speed_states[:, 0, i], label = 'rdot')
  #a4 = ax[i][1].plot(t, cpg_speed_states[:, 1, i], label = 'thetadot')

  ax2=ax[i][0].twinx()
  ax2_1=ax[i][1].twinx()
  ax2.set_ylabel('Phase [rad]', color='tab:orange')
  ax2_1.set_ylabel('Phase speed [rad/s]', color='tab:orange')
  a2 = ax2.plot(t, cpg_pos_states[:, 1, i], color='tab:orange', label='theta')
  a4 = ax2_1.plot(t, cpg_speed_states[:, 1, i], color= 'tab:orange', label='theta_dot')
  
  ax[i][0].set_xlim(0, 0.7)
  ax[i][1].set_xlim(0, 0.7)
  ax[i][0].set_xlabel('Time [s]')
  ax[i][1].set_xlabel('Time [s]')
  ax[i][0].set_ylabel('Amplitude', color='tab:blue')
  ax[i][1].set_ylabel('Amplitude speed', color='tab:blue')

fig.suptitle("CPG states for trot gait", fontweight ="bold", fontsize = 15)
#ax[0, 0].set_title("CPG Position States for each leg (FR, FL, RR, RL)", fontsize = 8)
#ax[0, 1].set_title("CPG Speed States for each leg (FR, FL, RR, RL)", fontsize = 8)

lgs1 = a1+a2
lgs2 = a3+a4
labs1 = [l.get_label() for l in lgs1]
labs2 = [l.get_label() for l in lgs2]
fig.legend(lgs1, labs1, "upper left")
fig.legend(lgs2, labs2, loc="upper right")
#fig.legend([a1, a2], labels=["r", "theta"], loc="upper left")
#fig.legend([a3, a4], labels=["r_dot", "theta_dot"], loc="upper right")

plt.show()

# PLOT WITH AND WITHOUT CARTESIAN PD
#fig = plt.figure()
fig, ax = plt.subplots()
b1 = ax.plot(t,des_foot_pos[0,:], label = 'Desired Foot Position')
#ax2 = ax.twinx()
b2 = ax.plot(t,foot_pos[0,:], color= 'tab:orange', label = 'Actual Foot Position')
if ADD_CARTESIAN_PD == "cart":
  plt.title("Plot comparing the desired foot position vs actual foot position with Cartesian PD", fontsize = 10)
if ADD_CARTESIAN_PD == "joint":
  plt.title("Plot comparing the desired foot position vs actual foot position with Joint PD", fontsize = 10)
if ADD_CARTESIAN_PD == "both":
  plt.title("Plot comparing the desired foot position vs actual foot position with Joint PD and Cartesian PD", fontsize = 10)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [m]')
ax.set_ylim(-0.06, 0.06)
ax.set_xlim(0, 3)

#lgs = b1+b2
#labs = [l.get_label() for l in lgs]
#ax.legend(lgs, labs, loc="upper right")
plt.legend([b1, b2], labels=["Desired foot position", "Actual foot position"],  loc="upper right")
plt.show()

# TORQUE PLOTS WITH AND WITHOUT CARTESIAN PD
#fig = plt.figure()
fig, ax = plt.subplots()
#b1 = ax.plot(t, leg_torques[:,0:3], label = 'Desired Foot Position')
#ax2 = ax.twinx()
c1 = ax.plot(t, leg_torques[:,1])
c2 = ax.plot(t, leg_torques[:,2])
#c3 = ax.plot(t, leg_torques[:,2])
if ADD_CARTESIAN_PD == "cart":
  plt.title("Thigh and calf torques with Cartesian PD", fontsize = 10)
if ADD_CARTESIAN_PD == "joint":
  plt.title("Thigh and calf torques with Joint PD", fontsize = 10)
if ADD_CARTESIAN_PD == "both":
  plt.title("Thigh and calf torques with Joint PD and Cartesian PD", fontsize = 10)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
#ax.set_ylim(-0.06, 0.06)
ax.set_xlim(0, 0.3)

plt.legend([c1, c2], labels=["torque on thigh", "torque on calf"],  loc="upper right")  
plt.show()
