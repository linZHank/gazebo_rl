#!/usr/bin/env python
from __future__ import print_function

import gym
import numpy as np
import time
import random
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, LinkStates

# import our training environment
from envs import crib_nav_task_env # need write task env
from utils import obs_to_state

rospy.init_node('env_test', anonymous=True, log_level=rospy.DEBUG)    
env = gym.make('CribNav-v0')

# test env with random sampled actions
for episode in range(10):
  state, info = env.reset()
  done = False
  for step in range(32):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print("Episode : {}, Step: {}, \nCurrent position: {}, Goal position: {}, Reward: {:.4f}".format(
      episode,
      step,
      info["current_position"],
      info["goal_position"],
      reward
    ))
    if done:
      break
    
