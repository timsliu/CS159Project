# actor critic fine tuning version 1
#
# This program implements actor critic multitask learning for a continuous
# environment using vanilla multitasking (hard parameter sharing).
#
# This version implements finetuning. The policy is first trained on a single
# environment and then the shared layer is fixed. Then a new environment is
# trained and the number of episodes to learn the new environment is printed.
# The neural net has a single actor and a single critic head.
#
# USAGE: pass --envs <one environemnt>
#
# The neural net is always first trained on one environment and then the
# network is frozen and trained on a second environment (passed as the 
# argument)
#
#
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
# Ayya       05/26/18    Fixed errors and made it usable for many envs
# Ayya       05/27/18    Changed rollouts in main()
# Tim Liu    05/28/18    began modifying for fine tuning on two fixed envs
# Tim Liu    05/28/18    removed capability to handle arbitray number of envs
# Tim Liu    05/28/18    updated main loop to train env1 first and then env2
# Tim Liu    05/28/18    modified finish_episode to freeze the affine layer
#                        after the first environment is trained
# Tim Liu    05/28/18    removed manual seeding of environment and torch
# Tim Liu    05/28/18    corrected bug so affine layer actually freezes
# Tim Liu    05/31/18    changed so user can pass what env to train





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


# global lists for recording run time behavior - initialized by init_list
# (FOR_RECORD)
length_records = []    # length of each run for each episode
rr_records = []        # running rewards for each episode

args = parser.parse_args()


pi = Variable(torch.FloatTensor([math.pi]))

# read in which second environment to train on
envs_names = args.envs[0]
if len(envs_names) != 1:
    print("Can only train one additional environment!")
    exit()

#first environment
env1 = gym.make('InvertedPendulum-v2')
#second environment
env2 = gym.make(envs_names[0])

# finetuning only tries these two environments
envs_names = ['InvertedPendulum-v2', envs_names[0]]


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
        # shared layer
        self.affine1 = nn.Linear(4, 128)
        # 2 outputs because 2D space
        # mu and sigma head
        self.mu_head = nn.Linear(128, 1)
        self.sigma2_head = nn.Linear(128, 1)
        # define the value head (for the critic)
        self.value_head = nn.Linear(128, 1)

        # initialize sigma and mu head (actor head)
        self.apply(weights_init)
        self.mu_head.weight.data = normalized_columns_initializer\
            (self.mu_head.weight.data, 0.01)
        self.sigma2_head.weight.data = normalized_columns_initializer\
            (self.sigma2_head.weight.data, 0.01)
        self.mu_head.bias.data.fill_(0)
        self.sigma2_head.bias.data.fill_(0)

        #initialization for the value head (critic head)
        self.value_head.weight.data = normalized_columns_initializer(self.value_head.weight.data, 1.0)
        self.value_head.bias.data.fill_(0)

        # initialize lists for holding run information
        self.saved_actions = []
        self.entropies = []
        self.rewards = []


    def forward(self, x):
        '''forward algorithm.
        argument: x - array with state of the environment
        return:  mu_head, sigma, value'''
        x = F.relu(self.affine1(x))
        return self.mu_head(x), F.softplus(self.sigma2_head(x)),\
               self.value_head(x)

# for debugging
test = False

model = Policy()
# learning rate - might be useful to change
optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    '''given a state, this function chooses the action to take
    arguments: state - observation matrix specifying the current model state
    return - action to take'''

    state = torch.from_numpy(state).float()
    mu, sigma, state_value = model(state)

    # if sigma is nan
    if sigma != sigma:
        print(mu)
        print(state_value)
        # print out the weights
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


def finish_episode(freeze):
    '''this function concludes each episode and updates the neural net
    arguments: freeze - boolean value describing whether to freeze the
                        affine layer'''
    policy_losses = []
    value_losses = []
    entropy_sum = 0
    saved_actions = model.saved_actions
    model_rewards = model.rewards
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
        # need gradient descent - so negative
        policy_losses.append(-log_prob * reward) #/ length_discount[env_idx])
        # https://pytorch.org/docs/master/nn.html#torch.nn.SmoothL1Loss
        # feeds a weird difference between value and the reward
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]))) #/ 
    optimizer.zero_grad()
    # sum of 2 losses?
    loss = (torch.stack(policy_losses).sum() + 0.5*torch.stack(value_losses).sum() \
            - torch.stack(model.entropies).sum() * 0.0001)

    # check if the affine layer should be updated
        # do not update the affine layer
    model.affine1.requires_grad = False
    
    # iterate through the layers
    p_num = 0
    for param in model.parameters():
        # freeze the first layer for finetuning
        if ((p_num == 0) and freeze):
            param.requires_grad = False
        p_num += 1
            
    loss.backward()    #compute the gradients
    nn.utils.clip_grad_norm_(model.parameters(), 30)

    # train the NN
    optimizer.step()
    # clear out saved rewards and actions
    del model.rewards[:]
    del model.saved_actions[:]
    del model.entropies[:]
    return



def main():
    '''main function. Loops training on the environments, intermittently
    printing out the current performance. Exits when finished with
    training and prints out summary information'''
    running_reward = 10       # initialize to roughly random performance
    env1_trained = False      # flag for whether first environment is trained
    env1_training_time = 0    # number of episodes to train env1
    env2_training_time = 0    # episodes to train env2
    
    # initialize the record lists to the proper length (FOR_RECORD)
    length_records = visualize.init_list(envs_names)
    rr_records = visualize.init_list(envs_names)    
    
    for i_episode in count(1): # loop and train
        # environment 1 is fully trained
        if env1_trained:
            env = env2      # use the second environment
            freeze = True   # freeze the affine layer
            env2_training_time += 1
            
        # environment 1 is not yet trained
        else:
            env = env1      # keep using the first environment
            freeze = False  # not done training - update the affine layer  
            env1_training_time += 1
            
        state = env.reset()
        for t in range(10000):  # loop for a single episode
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            reward = max(min(reward, 1), -1)
            model.rewards.append(reward)
            if done:
                break
        
        # update our running reward
        running_reward = running_reward * 0.99 + t * 0.01
        
        if env1_trained:
            # call function to record run data (FOR_RECORD)
            length_records, rr_records = visualize.update_fine(\
                t, running_reward, length_records, rr_records, 1)
            # environment 1 trained so fill data for env2 (last arg)
        else:
            # call function to record run data (FOR_RECORD)
            length_records, rr_records = visualize.update_fine(\
                t, running_reward, length_records, rr_records, 0)   
            # update record for environment 1 (last arg)
        
       
        # call function to update our neural net
        finish_episode(freeze)
        
        # intermittently print out the performance
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
            
        # reach here when the first environment is finished training
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            if env1_trained:
                # if env1 already trained and we reach here again then
                # both environments have been trained
                break
            # only reach here after first environment has been trained
            env1_trained = True
            running_reward = 10   # reset the running reward
    # finished with everything - print out summary
    print("Time to train environment 1: ", env1_training_time)
    print("Time to train environment 2: ", env2_training_time)
    
    # save the list (FOR_RECORD)
    visualize.pickle_list('fine_tuning', envs_names, length_records, rr_records)


if __name__ == '__main__':
    main()
