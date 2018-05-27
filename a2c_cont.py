# Revision History
#
# Tim Liu    05/26/18    Reduced learning rate to 1e-3
# Tim Liu    05/26/18    Added second environment to comment out and run on


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
parser.add_argument('--entropy-coef', type=float, default=0.01)
args = parser.parse_args()
pi = Variable(torch.FloatTensor([math.pi]))

#uncomment one of these to switch between environments
env = gym.make('InvertedPendulum-v2')
#env = gym.make('HalfInvertedPendulum-v0')
Max_action = env.action_space.high
Min_action = env.action_space.low
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


def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        # 2 outputs because 2D space
        self.mu_head = nn.Linear(128, 1)
        self.sigma2_head = nn.Linear(128, 1)
        self.value_head = nn.Linear(128, 1)
        self.apply(weights_init)
        self.mu_head.weight.data = normalized_columns_initializer(self.mu_head.weight.data, 0.01)
        self.sigma2_head.weight.data = normalized_columns_initializer(self.sigma2_head.weight.data, 0.01)
        self.mu_head.bias.data.fill_(0)
        self.sigma2_head.bias.data.fill_(0)
        self.value_head.weight.data = normalized_columns_initializer(self.value_head.weight.data, 1.0)
        self.value_head.bias.data.fill_(0)
        self.saved_actions = []
        self.entropies = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.mu_head(x), F.softplus(self.sigma2_head(x)), self.value_head(x)
        # torch.exp(self.sigma2_head(x))

# for debugging
test = False

model = Policy()
# learning rate - might be useful to change
optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    # retrain the model
    mu, sigma, state_value = model(state)

    # debugging
    if False:
        print(mu)
        print(state_value)
        print(model.sigma2_head.weight)
        print(model.affine1.weight)
    # if sigma is nan
    if sigma != sigma:
        print(mu)
        print(state_value)
        print(model.sigma2_head.weight)
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
    model.saved_actions.append(SavedAction(log_prob, state_value))
    # returns the action as a number converted to python type and bounded within -1, 1

    return action.item()


def finish_episode(state):
    R = torch.zeros(1, 1)
    R = Variable(R)
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    # compute the reward for each state in the end of a rollout
    for r in model.rewards[::-1]:
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
        print(model.sigma2_head.weight.grad)
        print(model.affine1.weight.grad)

    # train the NN
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]
    del model.entropies[:]


def main():
    running_reward = 10

    for i_episode in count(1):
        # random initialization
        state = env.reset()

        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            reward = max(min(reward, 1), -1)
            if args.render:
                env.render()
            model.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode(state)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
