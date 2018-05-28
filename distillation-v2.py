# actor critic multitasking continous
#
# This program implements actor critic multitask learning for a continuous
# environment using vanilla multitasking (hard parameter sharing).
#
# USAGE: --env ['env1', ..] If nothing passed, assumes 'InvertedPendulum-v2' and
# 'HalfInvertedPendulum-v0'
#
#




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
from a2c_mlt_extended_try2 import Policy as Teacher
from a2c_mlt_extended_try2 import weights_init, normalized_columns_initializer

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
    envs_names = ['InvertedPendulum-v2', 'HalfInvertedPendulum-v0']
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
# somehow does not load
teacherNNs = []
for i, name in enumerate(envs_names):
    teacher = torch.load(name + '.pt')
    teacher_mod = Teacher()
    teacher_mod.load_state_dict(teacher)
    teacherNN[i] = teacher_mod

torch.manual_seed(args.seed)



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

        self.mu = [[] for i in range(num_envs)]
        self.sigma = [[] for i in range(num_envs)]
        self.tmu = [[] for i in range(num_envs)]
        self.tsigma = [[] for i in range(num_envs)]



    def forward(self, x, env_idx):
        '''updated to have 5 return values (2 for each action head one for
        value'''
        x = F.relu(self.affine1(x))
        mu = self.mu_heads[env_idx](x)
        sigma2 = self.sigma2_heads[env_idx](x)
        value = self.value_heads[env_idx](x)
        sigma = F.softplus(sigma2)
        return mu, sigma, value


# for debugging
test = False

model = Policy()
# learning rate - might be useful to change
optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state, env_idx, roll):
    '''given a state, this function chooses the action to take
    arguments: state - observation matrix specifying the current model state
               env - integer specifying which environment to sample action
                         for
    return - action to take'''

    state = torch.from_numpy(state).float()
    tmodel = teacherNN[env_idx]
    mu, sigma, state_value = model(state, env_idx)
    tmu, tsigma, tvalue = tmodel(state, 0) # only the one environment

    # use student
    if roll == 1:
        prob = Normal(mu, sigma.sqrt())

    else: # use teacher
        prob = Normal(tmu, tsigma.sqrt())

    action = prob.sample()

    model.mu[env_idx].append(mu)
    model.tmu[env_idx].append(tmu)
    model.sigma[env_idx].append(sigma.sqrt())
    model.tsigma[env_idx].append(tsigma.sqrt())
    return action.item()


def finish_episode():
    policy_losses = []
    value_losses = []
    entropy_sum = 0
    #lengths = np.array([len(model.rewards[i]) for i in range(num_envs)])
    #length_discount = lengths/np.sum(lengths)

    optimizer.zero_grad()
    # sum of 2 losses?
    loss = KL_MV_gaussian(torch.tensor(model.tmu), torch.tensor(model.tsigma),
        torch.tensor(model.mu), torch.tensor(model.sigma))

    # Debugging
    if False:
        print(loss, 'loss')
    # compute gradients
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 30)

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
    model.mu = [[] for i in range(num_envs)]
    model.sigma = [[] for i in range(num_envs)]
    model.tmu = [[] for i in range(num_envs)]
    model.tsigma = [[] for i in range(num_envs)]

def KL_MV_gaussian(mu_p, std_p, mu_q, std_q):
    kl = (std_q/std_p).log() + (std_p.pow(2)+(mu_p-mu_q).pow(2)) / \
            (2*std_q.pow(2)) - 0.5
    kl = kl.sum() # sum across all dimensions
    kl = kl.mean() # take mean across all steps
    return kl

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
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
        #self.entropies = [[] for i in range(num_envs)]
        self.entropies = []
        self.rewards = [[] for i in range(num_envs)]

def main():
    running_reward = 10
    run_reward = np.array([10 for i in range(num_envs)])
    roll_length = np.array([0 for i in range(num_envs)])
    trained = False
    trained_envs = np.array([False for i in range(num_envs)])
    for i_episode in count(1):
        roll = np.random.randint(2)
        length = 0
        for env_idx, env in enumerate(envs):
            state = env.reset()
            done = False
            for t in range(10000):  # Don't infinite loop while learning
                action = select_action(state, env_idx, roll)
                state, reward, done, _ = env.step(action)
                reward = max(min(reward, 1), -1)

                if args.render:
                    env.render()
                if done:
                    length += t
                    roll_length[env_idx] = t
                    break

        # update our running reward
        running_reward = running_reward * 0.99 + length / num_envs * 0.01
        run_reward = run_reward * 0.99 + roll_length * 0.01
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


if __name__ == '__main__':
    main()
