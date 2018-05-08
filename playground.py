# File to play around with open AI gym environment
#
# Revision History:
# 05/03/18    Tim Liu    started file; wrote cart_pole

import gym
from gym_extensions.continuous import mujoco


def cart_pole():
    env = gym.make('CartPole-v0')  #create environment
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
    print("How'd I do?")
    return

def stand():
    #check if extension installed correctly
    env = gym.make("RoboschoolHumanoid-v0")  #create environment
    env.reset()
    return