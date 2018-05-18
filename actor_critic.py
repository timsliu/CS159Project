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


env = gym.make('HalfInvertedPendulum-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


#class for the policy being used; inherits from a neural network class
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        # converts action scores to probabilities
        return F.softmax(action_scores, dim=-1), state_values

#instantiate a new object from the policy class
model = Policy()
#choose the optimizer to use; lr is the learning rate
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    '''given a state, this function chooses the action to take
    arguments: state - observation matrix specifying the current model state
    return - action to take'''
    state = torch.from_numpy(state).float()
    # retrain the model
    probs, state_value = model(state)
    # creates a multinomial distribution out of probabilities
    m = Categorical(probs)
    # samples an action according to the policy distribution
    action = m.sample()
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
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        # reward is the delta param
        reward = r - value.item()
        # theta
        policy_losses.append(-log_prob * reward)
        # https://pytorch.org/docs/master/nn.html#torch.nn.SmoothL1Loss
        # feeds a weird difference between value and the reward
        # WTF
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    # WTF
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
        print(i_episode)
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            #sample an action
            action = select_action(state)
            #run the simulator and get the next step
            state, reward, done, _ = env.step(action)
            #make the image if the argument is set to do that
            if args.render:
                env.render()
            model.rewards.append(reward)
            #check if the simulation is over (we've fallen over)
            if done:
                break
        # I parameter
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        #print out some diagnostic information
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
