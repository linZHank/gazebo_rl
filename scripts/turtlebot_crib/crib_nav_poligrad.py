#! /usr/bin/env python

"""
Model based control for turtlebot with vanilla policy gradient in crib environment

Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import gym
import rospy
import random
import os
import time
import datetime
import matplotlib.pyplot as plt

import envs.crib_nav_task_env
from utils import bcolors, obs_to_state

    
def mlp(x, sizes, activation=tf.tanh, output_activation=None):
  # Build a feedforward neural network.
  for size in sizes[:-1]:
    x = tf.layers.dense(x, units=size, activation=activation)
  return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

if __name__ == "__main__":
  rospy.init_node("crib_nav_vpg", anonymous=True, log_level=rospy.WARN)
  env_name = 'CribNav-v0'
  env = gym.make(env_name)
  state_dim = obs_to_state(np.random.rand(env.observation_space.shape[0]), np.random.rand(2)).shape[0]
  num_actions = env.action_space.shape[0]
  # hyper parameters
  hidden_layer_size = [32]
  num_episodes = 50
  num_steps = 100
  lr = 1e-2
  batch_size = 5000
  rospy.loginfo("CribNav environment set")

  # make placeholders
  state_ph = tf.placeholder(shape=(None, state_dim), dtype=tf.float32)
  logits = mlp(state_ph, sizes=hidden_layer_size+[num_actions])
  action_ = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
  weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
  action_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
  action_masks = tf.one_hot(action_ph, num_actions)
  log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
  loss = -tf.reduce_mean(weights_ph * log_probs)
  # make train op
  train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
  
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  for ep in range(num_episodes):
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_weights = []
    ep_returns = []
    ep_lens = []
    obs, info = env.reset()
    print(
      bcolors.WARNING,
      "Env reset ...",
      "\nEpisode: {}".format(ep),
      "\nRobot init position: {}".format(obs[:2]),
      "\nGoal position: {}".format(info["goal_position"]),
      bcolors.ENDC
    )
    state = obs_to_state(obs, info["goal_position"])
    done = False

    for st in range(num_steps):
      batch_states.append(state)
      action_id = sess.run(action_, feed_dict={state_ph: state.reshape(1,-1)})[0]
      if not action_id:
        action = np.array([env.action_space.high[0], env.action_space.low[1]]) # id=0 => [high_lin, low_ang]
      else:
        action = env.action_space.high # id=1 => [high_lin, high_ang]
      print(bcolors.WARNING, "action: {}".format(action), bcolors.ENDC)
      obs, rew, done, info = env.step(action)
      state = obs_to_state(obs, info["goal_position"])
      # save action, reward
      batch_actions.append(action_id)
      batch_rewards.append(rew)
      if done: 
        # if episode is over, record info about episode
        ret = sum(batch_rewards)
        ep_returns.append(ret)
        ep_lens.append(st+1)
        # the weight for each logprob(a|s) is R(tau)
        batch_weights += [ret] * (st+1)
        # end experience loop if we have enough of it
        if len(batch_states) > batch_size:
          break
        break

    batch_loss, _ = sess.run(
      [loss, train_op],
      feed_dict={
        state_ph: np.array(batch_states),
        action_ph: np.array(batch_actions),
        weights_ph: np.array(batch_weights)
      }
    )
    print(
      "epoch: {:3d} \t loss: {:.3f} \t return: {:.3f} \t ep_len: {:.3f}".format(
        ep, batch_loss, np.mean(ep_returns), np.mean(ep_lens)
      )
    )
