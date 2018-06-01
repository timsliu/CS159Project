# actor critic multitasking continous
#
# This program implements actor critic multitask learning for a continuous
# environment using vanilla multitasking (hard parameter sharing).
#
# USAGE: --env ['env1', ..] If nothing passed, assumes 'InvertedPendulum-v2' and
# 'HalfInvertedPendulum-v0'
#
# Revision History:
# Tim Liu    05/31/18    updated for record keeping
# Tim Liu    05/31/18    removed manual seeding
#


import argparse
import math
import gym
import numpy as np
from itertools import count
from collections import namedtuple

# used for recording run time data (FOR_RECORD)
import visualize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.distributions.normal import Normal
from hard_param import Policy as Teacher
from hard_param import weights_init, normalized_columns_initializer

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--alpha', type=float, default=0.9, metavar='A',
                    help='alpha (default: 0.8)')
parser.add_argument('--beta', type=float, default=0.1, metavar='B',
                    help='beta (default: 0.2)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--envs', action='append', nargs='+', type=str)

args = parser.parse_args()

# global lists for recording run time behavior - initialized by init_list
# (FOR_RECORD)
length_records = []    # length of each run for each episode
rr_records = []        # running rewards for each episode


num_envs = 0
# number of environments
if args.envs:
    envs_names = args.envs[0]
    num_envs = len(envs_names)

pi = Variable(torch.FloatTensor([math.pi]))

if num_envs == 0:
    envs_names = ['InvertedPendulum-v2', 'LongInvertedPendulum-v0']
    # first environment
    env1 = gym.make('InvertedPendulum-v2')
    # second environment
    env2 = gym.make('LongInvertedPendulum-v0')
    num_envs = 2
    envs = [env1, env2]
else:
    envs = [gym.make(envs_names[i]) for i in range(num_envs)]




class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.num_envs = num_envs
        # shared layer
        self.affine1 = nn.Linear(4, 128)
        self.affine_pi0 = nn.Linear(4, 128)
        # not shared layers
        self.mu_heads = nn.ModuleList([nn.Linear(128, 1) for i in
                                       range(self.num_envs+1)])
        self.sigma2_heads = nn.ModuleList([nn.Linear(128, 1) for i in
                                           range(self.num_envs+1)])
        self.value_heads = nn.ModuleList([nn.Linear(128, 1) for i in range(self.num_envs)])

        self.apply(weights_init)
        # +1 for the distilled policy
        for i in range(self.num_envs+1):
            mu = self.mu_heads[i]
            sigma = self.sigma2_heads[i]
            mu.data = normalized_columns_initializer(mu.weight.data, 0.01)
            mu.bias.data.fill_(0)
            sigma.bias.data.fill_(0)
            if i != self.num_envs:
                value = self.value_heads[i]
                value.weight.data = normalized_columns_initializer(value.weight.data, 1.0)
                value.bias.data.fill_(0)


        # initialize lists for holding run information
        #self.div = [[] for i in range(self.num_envs)]
        self.saved_actions = [[] for i in range(self.num_envs)]
        #self.entropies = [[] for i in range(num_envs)]
        #self.entropies = [[] for i in range(self.num_envs)]
        self.rewards = [[] for i in range(self.num_envs)]
        self.log_prob = [[] for i in range(self.num_envs)]
        self.kl = []
        self.ent = []


    def forward(self, y, env_idx):
        '''updated to have 5 return values (2 for each action head one for
        value'''
        x = F.relu(self.affine1(y))
        mu = self.mu_heads[env_idx](x)
        sigma2 = self.sigma2_heads[env_idx](x)
        sigma = F.softplus(sigma2)
        value = self.value_heads[env_idx](x)

        z = F.relu(self.affine_pi0(y))
        mu_dist = self.mu_heads[num_envs](z)
        sigma2_dist = self.sigma2_heads[num_envs](z)
        sigma_dist = F.softplus(sigma2_dist)
        return mu, sigma, value, mu_dist, sigma_dist


# for debugging
test = False

model = Policy()
# learning rate - might be useful to change
optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def select_action(state, env_idx):
    '''given a state, this function chooses the action to take
    arguments: state - observation matrix specifying the current model state
               env - integer specifying which environment to sample action
                         for
    return - action to take'''

    state = torch.from_numpy(state).float()
    mu, sigma, value, mu_t, sigma_t = model(state, env_idx)

    prob = Normal(args.alpha*mu_t + args.beta*mu, args.alpha*sigma_t.sqrt() + \
                  args.beta*sigma.sqrt())

    entropy = 0.5*(((args.alpha*sigma_t + args.beta*sigma)*2*pi).log()+1)
    #prob = Normal(mu, sigma.sqrt())
    #entropy = 0.5*((sigma*2*pi).log()+1)
    action = prob.sample()
    '''new_KL = torch.zeros(1, 1)
    new_KL = Variable(new_KL)
    new_KL = torch.div(sigma_t.sqrt(), args.alpha*sigma_t.sqrt()+ args.beta*sigma.sqrt()).log() + \
             (args.alpha*sigma_t.sqrt()+args.beta*sigma.sqrt()).pow(2) + \
             torch.div(((1-args.alpha)*mu_t-(args.beta)*mu).pow(2),(2*sigma_t)) - 0.5
    '''
    log_prob = prob.log_prob(action)
    #log_prob_pi0 = -torch.div((action - mu_t).pow(2), 2*sigma_t) - 1/2*(2*pi*sigma_t).log()
    model.saved_actions[env_idx].append(SavedAction(log_prob, value))
    #model.entropies[env_idx].append(entropy)


    #model.div[env_idx].append(torch.div(sigma_t.sqrt(), args.alpha*sigma_t.sqrt()+ args.beta*sigma.sqrt()).log() + \
    #         (args.alpha*sigma_t.sqrt()+args.beta*sigma.sqrt()).pow(2) + \
    #         torch.div(((1-args.alpha)*mu_t-(args.beta)*mu).pow(2),(2*sigma_t)) - 0.5)
    #model.log_prob[env_idx].append(log_prob_pi0)
    model.kl.append(torch.div(sigma_t.sqrt(), args.alpha*sigma_t.sqrt()+ args.beta*sigma.sqrt()).log() + \
             (args.alpha*sigma_t.sqrt()+args.beta*sigma.sqrt()).pow(2) + \
             torch.div(((1-args.alpha)*mu_t-(args.beta)*mu).pow(2),(2*sigma_t)) - 0.5)
    model.ent.append(entropy)
    # model.div[env_idx].append(torch.div(tsigma.sqrt(),sigma.sqrt()).log() + torch.div(sigma+(tmu-mu).pow(2),tsigma*2) - 0.5)
    return action.item()


def finish_episode():
    policy_losses = []
    value_losses = []
    entropy_sum = 0
    loss = torch.zeros(1, 1)
    loss = Variable(loss)
    kl = []

    for env_idx in range(num_envs):
        saved_actions = model.saved_actions[env_idx]
        model_rewards = model.rewards[env_idx]
        R = torch.zeros(1, 1)
        R = Variable(R)
        rewards = []
        # compute the reward for each state in the end of a rollout
        i = len(model_rewards) - 1
        for r in model_rewards[::-1]:
            R = r + args.gamma * R #- 0.001*model.entropies[env_idx][i] #+ model.div[env_idx][i] - 0.001*model.entropies[env_idx][i]#- 1/args.beta*saved_actions[i][0] + args.alpha/args.beta*model.log_prob[env_idx][i]
            rewards.insert(0, R)
            i -= 1
        rewards = torch.tensor(rewards)
        if rewards.std() != rewards.std() or len(rewards) == 0:
            rewards = rewards - rewards.mean()
        else:
            rewards = (rewards - rewards.mean()) / rewards.std()
        #gamma = 0
        #for i, reward in enumerate(rewards):
        #    rewards = rewards + gamma * model.div[env_idx][i]
        #    gamma = gamma * args.gamma

        kl = kl + list(map(lambda x, y: x *y, map(lambda i: args.gamma**(i[0]+1), enumerate(list(model.div[env_idx]))), list(model.dix[env_idx])))

        temp = model.div[env_idx]
        ix = 0
        for (log_prob, value), r in zip(saved_actions, rewards):
            # reward is the delta param
            value += Variable(torch.randn(value.size()))
            reward = r - value.item()
            # theta
            # need gradient descent - so negative
            #policy_losses.append(-reward)
            policy_losses.append(-log_prob * reward) #/ length_discount[env_idx])
            # https://pytorch.org/docs/master/nn.html#torch.nn.SmoothL1Loss
            # feeds a weird difference between value and the reward
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))

            temp = model.div[env_idx][ix] *


    loss = (torch.stack(policy_losses).sum() + \
            0.5*torch.stack(value_losses).sum() + 0.002*torch.stack(model.kl).sum()- \
            torch.stack(model.ent).sum() * 0.0001) / num_envs


    # compute gradients
    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 30)

    # Debugging
    if False:
        print('grad')
        for i in range(1):
            #print(i)
            print(model.mu_heads[num_envs].weight.grad)
            print(model.sigma2_heads[num_envs].weight.grad)
            print('##')
            print(model.mu_heads[0].weight.grad)
            #print('##')
            #print(model.value_head[i].weight.grad)
            #print('##')
            #print(model.affine1.weight.grad)

    # train the NN
    optimizer.step()

    #model.div = [[] for i in range(model.num_envs)]
    model.saved_actions = [[] for i in range(model.num_envs)]
    #model.entropies = [[] for i in range(model.num_envs)]
    model.rewards = [[] for i in range(model.num_envs)]
    model.log_prob = [[] for i in range(model.num_envs)]
    model.kl = []
    model.ent = []

def main():

    # initialize the record lists to the proper length (FOR_RECORD)
    length_records = visualize.init_list(envs_names)
    rr_records = visualize.init_list(envs_names)

    running_reward = 10
    run_reward = np.array([10 for i in range(num_envs)])
    roll_length = np.array([0 for i in range(num_envs)])
    trained = False
    trained_envs = np.array([False for i in range(num_envs)])
    for i_episode in range(6000):
        p = np.random.random()
        # roll = np.random.randint(2)
        length = 0
        for env_idx, env in enumerate(envs):
            # Train each environment simultaneously with the distilled policy
            state = env.reset()
            done = False
            for t in range(10000):  # Don't infinite loop while learning
                action = select_action(state, env_idx)
                state, reward, done, _ = env.step(action)
                reward = max(min(reward, 1), -1)
                model.rewards[env_idx].append(reward)
                if args.render:
                    env.render()
                if done:
                    length += t
                    roll_length[env_idx] = t
                    break

        # update our running reward
        running_reward = running_reward * 0.99 + length / num_envs * 0.01
        run_reward = run_reward * 0.99 + roll_length * 0.01

        # call function to record run data (FOR_RECORD)
        length_records, rr_records = visualize.update_records(\
            roll_length, run_reward, length_records, rr_records)

        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tAverage length per environment {}'.format(i_episode, run_reward))

        for env_idx, env in enumerate(envs):
            if run_reward[env_idx] > env.spec.reward_threshold and trained_envs[env_idx] == False:
                print("{} solved!".format(envs_names[env_idx]))
                trained_envs[env_idx] = True
            if run_reward[env_idx] < env.spec.reward_threshold and trained_envs[env_idx] == True:
                print("{} not solved!".format(envs_names[env_idx]))
                trained_envs[env_idx] = False

        if False not in trained_envs:
            print("All solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
            break

    if i_episode == 5999:
        print('Well that failed')

    visualize.pickle_list('distral', envs_names, length_records,\
                              rr_records)

if __name__ == '__main__':
    main()
