# actor critic multitasking continous
#
# This program implements actor critic multitask learning for a continuous
# environment using vanilla multitasking (hard parameter sharing).
#
# Revision History
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
parser.add_argument('--entropy-coef', type=float, default=0.01)
args = parser.parse_args()
pi = Variable(torch.FloatTensor([math.pi]))

# first environment
env1 = gym.make('InvertedPendulum-v2')
env1.seed(args.seed)

# second environment
env2 = gym.make('HalfInvertedPendulum-v0')
env2.seed(args.seed)

# take max_action from the first environment
Max_action = env1.action_space.high
Min_action = env1.action_space.low
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
        # shared layer
        # starting layer has 4 nodes because 4 parameters describe the states
        self.affine1 = nn.Linear(4, 128) 

        # 2 outputs because 2D space
        # mu and sigma head for environment 1
        self.mu_head_env = nn.Linear(128, 1)
        self.sigma_head_env = nn.Linear(128, 1)

        '''
        # mu and sigma head for environment 2
        self.mu_head_env2 = nn.Linear(128, 1)
        self.sigma2_head_env2 = nn.Linear(128, 1)
        '''
        # define the value heads
        # is this the critic
        self.value_head_env = nn.Linear(128, 1)

        #self.value_head_env2 = nn.Linear(128, 1)

        # initialize environment head
        self.apply(weights_init)
        self.mu_head_env.weight.data = normalized_columns_initializer\
            (self.mu_head_env.weight.data, 0.01)
        self.sigma_head_env.weight.data = normalized_columns_initializer\
            (self.sigma_head_env.weight.data, 0.01)
        self.mu_head_env.bias.data.fill_(0)
        self.sigma_head_env.bias.data.fill_(0)

        # initialize environment 2 head
        '''self.apply(weights_init)
        self.mu_head_env2.weight.data = normalized_columns_initializer\
            (self.mu_head_env2.weight.data, 0.01)
        self.sigma2_head_env2.weight.data = normalized_columns_initializer\
            (self.sigma2_head_env2.weight.data, 0.01)
        self.mu_head_env2.bias.data.fill_(0)
        self.sigma2_head_env2.bias.data.fill_(0)'''

        #initialization for the value heads
        self.value_head_env.weight.data = normalized_columns_initializer(self.value_head_env.weight.data, 1.0)
        self.value_head_env.bias.data.fill_(0)

        '''
        self.value_head_env2.weight.data = normalized_columns_initializer(self.value_head_env2.weight.data, 1.0)
        self.value_head_env2.bias.data.fill_(0)
        '''

        # initialize lists for holding run information
        self.saved_actions = []
        self.entropies = [] #why???
        self.rewards_env = []

        '''
        self.saved_actions_env2 = []
        self.rewards_env2 = []
        '''

    def forward(self, x):
        '''updated to have 3 return values (2 for each action head one for
        value'''
        x = F.relu(self.affine1(x))
        return self.mu_head_env(x), F.softplus(self.sigma_head_env(x)),\
               self.value_head_env(x)
        # torch.exp(self.sigma2_head(x))

# for debugging
test = False

model1 = Policy()
model2 = Policy()

# learning rate - might be useful to change
optimizer = optim.Adam(model1.parameters(), lr=1e-4)
eps = np.finfo(np.float32).eps.item()


def select_action(state, env):
    '''given a state, this function chooses the action to take
    arguments: state - observation matrix specifying the current model state
               env - integer specifying which environment to sample action
                         for
    return - action to take'''

    state = torch.from_numpy(state).float()
    # retrain the model - returns

    '''
    mu1, sigma1, mu2, sigma2, state_value1, state_value2 = model(state)
    '''

    # mu1 sigma 1 correspond to environment 1
    # mu2 sigma 2 correspond to environment 2

    # decide which mu and sigma to use depending on the env being trained
    if env == 1:
        model = model1
        mu, sigma, state_value = model1.forward(state)
        saved_actions = model1.saved_actions
    if env == 2:
        model = model2
        mu, sigma, state_value = model2.forward(state)
        saved_actions = model2.saved_actions

    # debugging
    if False:
        print(mu)
        print(state_value)
        print(model.sigma_head_env.weight)
        print(model.affine1.weight)
    # if sigma is nan
    if sigma != sigma:
        print(mu)
        print(state_value)
        # print out the weights
        print(model.sigma_head_env.weight)
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
    saved_actions.append(SavedAction(log_prob, state_value))
    # returns the action as a number converted to python type and bounded within -1, 1

    return action.item()

def finish_episode(state, env):
    num_envs = 2
    policy_losses = []
    value_losses = []

    if env % 2 == 0:
        model = model2
        saved_actions = model2.saved_actions
        model_rewards = model2.rewards_env
    else:
        model = model1
        saved_actions = model1.saved_actions
        model_rewards = model1.rewards_env

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

    lmda = 0.01

    affine1loss = model1.affine1.weight - model2.affine1.weight 
    mu_head_loss = model1.mu_head_env.weight - model2.mu_head_env.weight
    sigma_head_loss = model1.sigma_head_env.weight - model2.sigma_head_env.weight
    value_head_loss = model1.value_head_env.weight - model2.value_head_env.weight

    for weight in affine1loss:
        for w in weight:
            loss += (lmda/2) * (math.pow(w, 2))

    for weight in mu_head_loss:
        for w in weight:
            loss += (lmda/2) * (math.pow(w, 2))

    for weight in sigma_head_loss:
        for w in weight:
            loss += (lmda/2) * (math.pow(w, 2))

    for weight in value_head_loss:
        for w in weight:
            loss += (lmda/2) * (math.pow(w, 2))

    # Debugging
    if False:
        print(loss, 'loss')
    # compute gradients
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 20)

    # Debugging
    if False:
        print('grad')
        print(model.sigma2_head.weight.grad)
        print(model.affine1.weight.grad)

    # train the NN
    optimizer.step()
    del model.saved_actions[:]
    del model.entropies[:]
    del model.rewards_env[:]
    del model.saved_actions[:]
    del model.rewards_env[:]


def main():
    running_reward = 10

    for i_episode in count(1):
        # random initialization
        state1 = env1.reset()
        state2 = env2.reset()

        for t in range(10000):  # Don't infinite loop while learning
            if t % 2 == 0:
                state = state1  # variable used for finishing
                # train environment 1 half the time
                action = select_action(state1, 1)
                state1, reward, done, _ = env1.step(action)
                reward = max(min(reward, 1), -1)
                # add parameter regularization for soft parameter sharing
                model1.rewards_env.append(reward)
                if args.render:
                    env1.render()
                if done:
                    break
            if t % 2 == 1:
                # train environment 2 other half of the time
                state = state2  # variable used for finishing
                action = select_action(state2, 2)
                state2, reward, done, _ = env2.step(action)
                reward = max(min(reward, 1), -1)
                model2.rewards_env.append(reward)
                if args.render:
                    env2.render()
                if done:
                    break
            # clip the rewards (?)
            #reward = max(min(reward, 1), -1)
            # render if arguments specify it

            # keep running list of all rewards

            #model.rewards_env1.append(reward)

        t = int(t/2)  #divide by two because we alternated between two environments
        # update our running reward
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode(state1, 1)
        finish_episode(state2, 2)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        # for now use env1 reward threshold
        if running_reward > env1.spec.reward_threshold and running_reward > env2.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
