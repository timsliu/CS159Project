# Actor critic multitask discrete
#
# This file attempts vanilla multitasking (one neural net two heads)
# on two discrete environments
#
# Revision History
# 05/19/18    Tim Liu    created second environment env2
# 05/19/18    Tim Liu    added second action head to Policy class
# 05/19/18    Tim Liu    modified forward module for second action; returns
#                        a third argument for second head
# 05/19/18    Tim Liu    added conditional statement to main to alternate
#                        training the two different environments
# 05/22/18    Tim Liu    swapped to discrete environments - was previously
#                        using continuous ones

import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import mujoco_py



parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

#first environment to train on
env1 = gym.make('CartPole-v1')
#second environment to train on
env2 = gym.make('CartPoleLowG-v1')
#seed both environments with the same randomness
env1.seed(args.seed)
env2.seed(args.seed)
#now seed the torch randomness
torch.manual_seed(args.seed)

#save actions for each of the two heads
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


#class for the policy being used; inherits from a neural network class
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        #head for action 1
        self.action1_head = nn.Linear(128, 2)
        #head for action 2
        self.action2_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        # add second action_score for second environment
        action1_scores = self.action1_head(x)
        action2_scores = self.action2_head(x)
        state_values = self.value_head(x)
        # converts action scores to probabilities
        return F.softmax(action1_scores, dim=-1), \
               F.softmax(action2_scores, dim=-1),  \
               state_values

#instantiate a new object from the policy class - keep single policy for both
model = Policy()
#choose the optimizer to use; lr is the learning rate
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state, env):
    '''given a state, this function chooses the action to take
    arguments: state - observation matrix specifying the current model state
               env - integer specifying which environment to sample action
                         for
    return - action to take'''
    state = torch.from_numpy(state).float()
    # retrain the model
    probs1, probs2, state_value = model(state)
    #select which probability to use based on passed argument
    if env == 1:
        probs = probs1
    if env == 2:
        probs = probs2
    # creates a multinomial distribution out of probabilities
    m = Categorical(probs)
    # samples an action according to the policy distribution
    action = m.sample()
    
    # save the actions that we took for each environment 
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))        
    # returns the action as a number converted to python type
    return action.item()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    # convert to a tensor form
    rewards = torch.tensor(rewards)
    #normalize all of them
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    #may need to change this so it works on both sets of saved_actions
    for (log_prob, value), r in zip(saved_actions, rewards):
        # reward is the delta param
        reward = r - value.item()
        # theta
        policy_losses.append(-log_prob * reward)
        # https://pytorch.org/docs/master/nn.html#torch.nn.SmoothL1Loss
        # feeds a weird difference between value and the reward
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
               
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # compute gradients
    loss.backward()
    # train the NN
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        # random initialization
        state1 = env1.reset()
        state2 = env2.reset()
        fail = 0
        for t in range(10000):  # Don't infinite loop while learning
            if t % 2 == 0:
                state = state1
                #action in environment 1 half the time
                action = select_action(state, 1)
                #run the simulator and get the next step
                state1, reward, done, _ = env1.step(action)
                #make the image if the argument is set to do that
                if args.render:
                    env1.render()
                model.rewards.append(reward)
                #check if the simulation is over (we've fallen over)
                if done:
                    fail = 1
                    break                
            if t % 2 == 1:
                state = state2
                #action in environment 2 half the time
                #sample an action
                #action in environment 1 half the time
                action = select_action(state, 2)
                #run the simulator and get the next step
                state2, reward, done, _ = env2.step(action)
                #make the image if the argument is set to do that
                if args.render:
                    env2.render()
                model.rewards.append(reward)
                #check if the simulation is over (we've fallen over)
                if done:
                    fail = 2
                    break                
        # terminates if either one falls over
        t = int(t/2)  #divide by two because we alternated between two environments
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        #print out some diagnostic information
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
            print("Failed environment: %d"  %fail)
        #reached reward threshold
        if running_reward > env1.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
