# File to play around with open AI gym environment
#
# Revision History:
# 05/03/18    Tim Liu    started file; wrote cart_pole

import gym
import numpy as np
#from gym_extensions.continuous import gym_navigation_2d
#from gym_extensions.continuous import mujoco



def cart_pole():
    env = gym.make('CartPole-v0')  #create environment
    env.reset()
    for i in range(100):
        print("Step: ", i)
        action = env.action_space.sample()
        print("action: ", action)
        observation, reward, done, info = env.step(action) # take a random action
        #dump all of the matrices
        print("obs: ", observation)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)

    print("How'd I do?")
    return

def cart_pole2():
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            print(action)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break  
    print("How'd I do?")
    return

def stand():
    #check if extension installed correctly
    env = gym.make("RoboschoolHumanoid-v0")  #create environment
    env.reset()
    return

def mi_pend():
    #create mujoco inverted pendulum    
    env = gym.make('InvertedPendulum-v1')
    for i in range(10):
        env.reset()
        for i in range(1000):
            a = (np.random.rand(*env.action_space.shape) - 0.5) * 1.1
            # returns observation, reward, done, info
            o, r, d, i = env.step(a)  
            env.render()    