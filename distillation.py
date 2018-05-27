import argparse
import math
import gym
import numpy as np
from itertools import count
from collections import namedtuple
from gym.envs.mujoco.HalfInvertedPendulum import HalfInvertedPendulumEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.distributions.normal import Normal


# first environment
env1 = gym.make('InvertedPendulum-v2')
env1.seed(args.seed)
# second environment
env2 = gym.make('HalfInvertedPendulum-v0')
env2.seed(args.seed)


class Policy(nn.Module):
    def __init__():
        # shared layer for students
        self.affine1 = nn.Linear(4, 128)
        # 2 outputs because 2D space
        # mu and sigma head for environment 1
        self.mu_head_env1 = nn.Linear(128, 1)
        self.sigma2_head_env1 = nn.Linear(128, 1)
        # mu and sigma head for environment 2
        self.mu_head_env2 = nn.Linear(128, 1)
        self.sigma2_head_env2 = nn.Linear(128, 1)
        # define the value heads
        self.value_head_env1 = nn.Linear(128, 1)
        self.value_head_env2 = nn.Linear(128, 1)

        # Teacher policies
        self.teacher_mu1 = nn.Linear(128, 1)
        self.teacher_sigma1 = nn.Linear(128, 1)
        self.teacher_value1 = nn.Linear(128, 1)

        self.teacher_mu2 = nn.Linear(128, 1)
        self.teacher_sigma2 = nn.Linear(128, 1)
        self.teacher_value2 = nn.Linear(128, 1)

        self.saved_actions_student = {'1': [], '2': []}
        self.saved_actions_teacher = {'1': [], '2': []}
        self.rewards = {'1': [], '2': []}
        self.entropies = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.mu_head_env1(x), F.softplus(self.sigma2_head_env1(x)),\
               self.mu_head_env2(x), F.softplus(self.sigma2_head_env2(x)),\
               self.value_head_env1(x), self.value_head_env2(x), \
               self.teacher_mu1(x), self.teacher_sigma1(x), \
               self.teacher_mu2(x), self.teacher_sigma2(x), \
               self.teacher_value1(x), self.teacher_value2(x)


model = Policy()
# learning rate - might be useful to change
optimizer = optim.Adam(model.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()

def select_action(state, env):
    # Randomly use teacher / student for rollout
    # choice = np.random.choice(2, p=rollout_prob)

    state = torch.from_numpy(state).float()
    mu1, s1, mu2, s2, val1, val2, tmu1, ts1, tmu2, ts2, tval1, tval2 = \
            model(state)

    if env == 1:
        prob = Normal(tmu1, ts1.sqrt())
        entropy = 0.5*((ts1*2*pi).log()+1)
        action = prob.sample()
        log_prob = prob.log_prob(action)
        model.entropies.append(entropy)
        model.saved_actions_teacher[1].append(SavedAction(log_prob,
                                                          state_value))

        prob = Normal(mu1, s1.sqrt())
        entropy = 0.5*((s1*2*pi).log()+1)
        action = prob.sample()
        log_prob = prob.log_prob(action)
        model.entropies.append(entropy)
        model.saved_actions_student[1].append(SavedAction(log_prob,
                                                          state_value))
    elif env == 2:
        prob = Normal(tmu2, ts2.sqrt())
        entropy = 0.5*((ts2*2*pi).log()+1)
        action = prob.sample()
        log_prob = prob.log_prob(action)
        model.entropies.append(entropy)
        model.saved_actions_teacher[2].append(SavedAction(log_prob,
                                                          state_value))
        prob = Normal(mu2, s2.sqrt())
        entropy = 0.5*((s2*2*pi).log()+1)
        action = prob.sample()
        log_prob = prob.log_prob(action)
        model.entropies.append(entropy)
        model.saved_actions_student[2].append(SavedAction(log_prob,
                                                          state_value))
    return action.item()


def KL_MV_gaussian(mu_p, std_p, mu_q, std_q):
    kl = (std_q/std_p).log() + (std_p.pow(2)+(mu_p-mu_q).pow(2)) / \
            (2*std_q.pow(2)) - 0.5
    kl = kl.sum(1, keepdim=True) # sum across all dimensions
    kl = kl.mean() # take mean across all steps
    return kl


def finish_episode(state, teacher_student):
    # Compare teacher distribution against student distribution and
    # enforce closeness with KL divergence
    num_envs = 2
    policy_losses = []
    value_losses = []
    for i in range(num_envs):
        if i % 2 == 0 and:
            saved_actions = model.saved_actions_env1
            model_rewards = model.rewards_env1
        else:
            saved_actions = model.saved_actions_env2
            model_rewards = model.rewards_env2

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
            - torch.stack(model.entropies).sum() * 0.0001) + #KL divergence

    # compute gradients
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 40)

    # train the NN
    optimizer.step()
    del model.saved_actions_student[1][:]
    del model.saved_actions_student[2][:]
    del model.entropies[:]
    del model.rewards[1][:]
    del model.rewards[2][:]
    del model.saved_actions_teacher[1][:]
    del model.saved_actions_teacher[2][:]


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
                action, teacher_student = select_action(state1, 1)
                state1, reward, done, _ = env1.step(action)
                reward = max(min(reward, 1), -1)
                model.rewards_env1.append(reward)
                if args.render:
                    env1.render()
                if done:
                    break
            if t % 2 == 1:
                # train environment 2 other half of the time
                state = state2  # variable used for finishing
                action, teacher_student = select_action(state2, 2)
                state2, reward, done, _ = env2.step(action)
                reward = max(min(reward, 1), -1)
                model.rewards_env2.append(reward)
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
        finish_episode(state)
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
