# actor critic multitasking continous
#
# This program implements actor critic multitask learning for a continuous
# environment using vanilla multitasking (hard parameter sharing).
#
# USAGE: --env ['env1', ..] If nothing passed, assumes 'InvertedPendulum-v2' and
# 'HalfInvertedPendulum-v0'
#
#
#
# Revision History
# Ayya Alieva 05/26/18   Fixed errors and made it usable for many envs
# Tim Liu    05/23/18    copied from a2c_cont.py and renamed
# Tim Liu    05/23/18    added second environment env2
# Tim Liu    05/23/18    added second sigma and mu head for Policy class
# Tim Liu    05/23/18    modified forward to return second sigma and mu
# Tim Liu    05/23/18    added second argument to select_action for choosing
#                        which head to sample
# Tim Liu    05/23/18    changed main loop to allow for multitasking
# Tim Liu    05/26/18    changed print statement in select_action for if
#                        sigma is NaN to reflect new sigma head attribute names



import argparse
import math
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.distributions.normal import Normal


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--envs', action='append', nargs='+', type=str)


args = parser.parse_args()

num_envs = 0
# number of environments
if args.envs:
    envs_names = args.envs[0]
    num_envs = len(envs_names)

pi = Variable(torch.FloatTensor([math.pi]))

if num_envs == 0:
    # first environment
    env1 = gym.make('InvertedPendulum-v2')
    # second environment
    env2 = gym.make('HalfInvertedPendulum-v0')
    num_envs = 2
    envs = [env1, env2]

else:
    envs = [gym.make(envs_names[i]) for i in range(num_envs)]

for env in envs:
    env.seed(args.seed)


torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.num_envs = num_envs
        # shared layer
        self.affine1 = nn.Linear(4, 128)
        # not shared layers
        self.mu_heads = nn.ModuleList([nn.Linear(128, 1) for i in range(self.num_envs)])
        self.sigma2_heads = nn.ModuleList([nn.Linear(128, 1) for i in range(self.num_envs)])
        self.value_heads = nn.ModuleList([nn.Linear(128, 1) for i in range(self.num_envs)])

        self.apply(weights_init)
        for i in range(self.num_envs):
            mu = self.mu_heads[i]
            sigma = self.sigma2_heads[i]
            value = self.value_heads[i]
            mu.data = normalized_columns_initializer(mu.weight.data, 0.01)
            mu.bias.data.fill_(0)
            sigma.bias.data.fill_(0)
            value.weight.data = normalized_columns_initializer(value.weight.data, 1.0)
            value.bias.data.fill_(0)

        # initialize lists for holding run information
        self.saved_actions = [[] for i in range(num_envs)]
        self.entropies = []
        self.rewards = [[] for i in range(num_envs)]


    def forward(self, x, env_idx):
        '''updated to have 5 return values (2 for each action head one for
        value'''
        x = F.relu(self.affine1(x))
        mu = self.mu_heads[env_idx](x)
        sigma2 = self.sigma2_heads[env_idx](x)
        value = self.value_heads[env_idx](x)
        sigma = F.softplus(sigma2)
        return mu, sigma, value
        # torch.exp(self.sigma2_head(x))

# for debugging
test = False

model = Policy()
# learning rate - might be useful to change
optimizer = optim.Adam(model.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state, env_idx):
    '''given a state, this function chooses the action to take
    arguments: state - observation matrix specifying the current model state
               env - integer specifying which environment to sample action
                         for
    return - action to take'''

    state = torch.from_numpy(state).float()
    # retrain the model - returns
    # pass the number of environment
    mu, sigma, state_value = model(state, env_idx)
    # mu1 sigma 1 correspond to environment 1
    # mu2 sigma 2 correspond to environment 2


    # debugging
    if False:
        print(mu)
        print(state_value)
        print(model.sigma2_head_env1.weight)
        print(model.sigma2_head_env2.weight)
        print(model.affine1.weight)
    # if sigma is nan
    if sigma != sigma:
        print(mu)
        print(state_value)
        # print out the weights
        print(model.sigma2_head_env1.weight)
        print(model.sigma2_head_env2.weight)
        sigma = torch.tensor(float(0.1))
        print('sigma is nan')
        exit()

    # creates a multivariate distribution
    # samples an action according to the policy distribution
    prob = Normal(mu, sigma.sqrt())
    entropy = 0.5*((sigma*2*pi).log()+1)
    action = prob.sample()
    log_prob = prob.log_prob(action)
    model.entropies.append(entropy)
    #print(env_idx)
    #print(model.saved_actions)
    saved_actions = model.saved_actions[env_idx]
    saved_actions.append(SavedAction(log_prob, state_value))
    # returns the action as a number converted to python type and bounded within -1, 1

    return action.item()


def finish_episode(state):
    policy_losses = []
    value_losses = []
    for env_idx in range(num_envs):
        saved_actions = model.saved_actions[env_idx]
        model_rewards = model.rewards[env_idx]
        R = torch.zeros(1, 1)
        R = Variable(R)
        rewards = []
        # compute the reward for each state in the end of a rollout
        for r in model_rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        if rewards.std() != rewards.std() or len(rewards) == 0:
            rewards = rewards - rewards.mean()
        else:
            rewards = (rewards - rewards.mean()) / rewards.std()

        for (log_prob, value), r in zip(saved_actions, rewards):
            # reward is the delta param
            value += Variable(torch.randn(value.size()))
            reward = r - value.item()
            # theta
            # need gradient descent - so negative
            policy_losses.append(-log_prob * reward)
            # https://pytorch.org/docs/master/nn.html#torch.nn.SmoothL1Loss
            # feeds a weird difference between value and the reward
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    # sum of 2 losses?
    loss = (torch.stack(policy_losses).sum() + 0.5*torch.stack(value_losses).sum() \
            - torch.stack(model.entropies).sum() * 0.0001)

    # Debugging
    if False:
        print(loss, 'loss')
    # compute gradients
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 40)

    # Debugging
    if False:
        print('grad')
        for i in range(num_envs):
            print(i)
            print(model.mu_head[i].weight.grad)
            print('##')
            #print(model.sigma2_head[i].weight.grad)
            print('##')
            #print(model.value_head[i].weight.grad)
            print('##')
            #print(model.affine1.weight.grad)


    # train the NN
    optimizer.step()
    model.saved_actions = [[] for i in range(num_envs)]
    del model.entropies[:]
    model.rewards = [[] for i in range(num_envs)]



def main():
    running_reward = 10
    trained = False
    for i_episode in count(1):
        # random initialization
        states = [env.reset() for env in envs]
        done = False
        for t in range(10000):  # Don't infinite loop while learning
            for env_idx, env in enumerate(envs):
                state = states[env_idx]  # variable used for finishing
                # train environment 1 half the time
                action = select_action(state, env_idx)
                state, reward, done, _ = env.step(action)
                reward = max(min(reward, 1), -1)
                model.rewards[env_idx].append(reward)
                states[env_idx] = state
                if args.render:
                    env.render()
                if done:
                    break
            if done:
                break


        #t = int(t/2)  #divide by two because we alternated between two environments
        # update our running reward
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode(state)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))

        for env_idx, env in enumerate(envs):
            if running_reward > env.spec.reward_threshold:
                print("{} solved!".format(envs_names[env_idx]))
                trained = True
            else:
                trained = False
        if trained:
            print("All solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
