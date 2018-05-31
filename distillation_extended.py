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

parser = argparse.ArgumentParser(description='distillation')
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
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
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
        '''self.teacher_mu1 = nn.Linear(128, 1)
        self.teacher_sigma1 = nn.Linear(128, 1)
        self.teacher_value1 = nn.Linear(128, 1)

        self.teacher_mu2 = nn.Linear(128, 1)
        self.teacher_sigma2 = nn.Linear(128, 1)
        self.teacher_value2 = nn.Linear(128, 1)'''

        self.saved_actions_student = {1: [], 2:[]}
        # self.saved_actions_teacher = {1: [], 2: []}
        self.samples_student = []
        # self.samples_teacher = {1: [], 2: []}
        self.rewards_student = {1: [], 2: []}
        # self.rewards_teacher = {1: [], 2: []}
        self.entropies = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.mu_head_env1(x), F.softplus(self.sigma2_head_env1(x)),\
               self.mu_head_env2(x), F.softplus(self.sigma2_head_env2(x)),\
               self.value_head_env1(x), self.value_head_env2(x)
               # self.teacher_mu1(x), self.teacher_sigma1(x), \
               # self.teacher_mu2(x), self.teacher_sigma2(x), \
               # self.teacher_value1(x), self.teacher_value2(x)


model = Policy()
# learning rate - might be useful to change
optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()

def select_action(state, env, teacher_mod, teacher_student):
    state = torch.from_numpy(state).float()
    mu1, s1, mu2, s2, val1, val2 = model(state)
    tmu1, ts1, tmu2, ts2, tval1, tval2 = teacher_mod(state)

    if env == 1:
        prob = Normal(tmu1, ts1.sqrt())
        entropy = 0.5*((ts1*2*pi).log()+1)
        action = prob.sample()
        log_prob_t = prob.log_prob(action)
        # model.entropies.append(entropy)
        # teacher_mod.saved_actions_env1.append(SavedAction(log_prob,
        #                                                   tval1))
        teacher_mod.saved_actions_env1.append((tmu1, ts1))

        prob = Normal(mu1, s1.sqrt())
        entropy = 0.5*((s1*2*pi).log()+1)
        action = prob.sample()
        log_prob_s = prob.log_prob(action)
        model.entropies.append(entropy)

        model.samples_student.append((mu1, s1))

        if teacher_student == 1:
            # Randomly save student
            model.saved_actions_student[env].append(SavedAction(log_prob_s,
                                                           val1))
        else:
            model.saved_actions_student[env].append(SavedAction(log_prob_t,
                                                           tval1))

    elif env == 2:
        prob = Normal(tmu2, ts2.sqrt())
        entropy = 0.5*((ts2*2*pi).log()+1)
        action = prob.sample()
        log_prob_t = prob.log_prob(action)
        # model.entropies.append(entropy)
        teacher_mod.saved_actions_env1.append(SavedAction(log_prob_t,
                                                          tval2))
        # model.samples_teacher[2].append((tmu2, ts2))
        prob = Normal(mu2, s2.sqrt())
        entropy = 0.5   *((s2*2*pi).log()+1)
        action = prob.sample()
        log_prob_s = prob.log_prob(action)
        model.entropies.append(entropy)
        model.samples_student.append((mu2, s2))

        if teacher_student == 1:
            # Randomly save student or teacher
            model.saved_actions_student[env].append(SavedAction(log_prob_s,
                                                              val2))
        else:
            model.saved_actions_student[env].append(SavedAction(log_prob_t, tval2))
    return action.item()


def KL_MV_gaussian(mu_p, std_p, mu_q, std_q):
    kl = (std_q/std_p).log() + (std_p.pow(2)+(mu_p-mu_q).pow(2)) / \
            (2*std_q.pow(2)) - 0.5
    kl = kl.sum() # sum across all dimensions
    kl = kl.mean() # take mean across all steps
    return kl


def finish_episode(state):
    # Compare teacher distribution against student distribution and
    # enforce closeness with KL divergence
    num_envs = 2
    policy_losses = []
    value_losses = []
    for i in range(num_envs):
        # if i % 2 == 0:
        saved_actions = model.saved_actions_student[i+1]
        model_rewards = model.rewards_student[i+1]
        # else:
        #     saved_actions = model.saved_actions_student[2]
        #     model_rewards = model.rewards_student[2]

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
            # value = value + Variable(torch.randn(value.size()))
            reward = r - value.item()
            # theta
            # need gradient descent - so negative
            policy_losses.append(-log_prob * reward)
            # https://pytorch.org/docs/master/nn.html#torch.nn.SmoothL1Loss
            # feeds a weird difference between value and the reward
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    # sum of 2 losses?
    teacher_mu = [j[0] for j in teacher_mod.saved_actions_env1]
    # teacher_mu2 = [j[0] for j in teacher_mod.saved_actions_env2]
    # teacher_mu = teacher_mu + teacher_mu2
    #
    teacher_sigma = [j[1].sqrt() for j in teacher_mod.saved_actions_env1]
    # teacher_sigma2 = [j[1].sqrt() for j in teacher_mod.saved_actions_env2]
    # teacher_sigma = teacher_sigma + teacher_sigma2
    student_mu = [j[0] for j in model.samples_student]
    # student_mu2 = [j[0] for j in model.samples_studenT]
    # student_mu = student_mu + student_mu2
    student_sigma = [j[1].sqrt() for j in model.samples_student]
    # student_sigma2 = [j[1].sqrt() for j in model.samples_student]
    # student_sigma = student_sigma + student_sigma2

    loss = (torch.stack(policy_losses).sum() + \
            0.5*torch.stack(value_losses).sum() - \
            torch.stack(model.entropies).sum() * 0.0001) + \
            KL_MV_gaussian(torch.tensor(teacher_mu),
                           torch.tensor(teacher_sigma),
                           torch.tensor(student_mu),
                           torch.tensor(student_sigma))
    # compute gradients
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 30)

    # train the NN
    optimizer.step()
    del model.saved_actions_student[1][:]
    del model.saved_actions_student[2][:]
    del model.entropies[:]
    del model.rewards_student[1][:]
    del model.rewards_student[2][:]
    del teacher_mod.saved_actions_env1[:]
    # del model.rewards_teacher[1][:]
    # del model.rewards_teacher[2][:]
    # del model.saved_actions_teacher[1][:]
    # del model.saved_actions_teacher[2][:]
    # del model.samples_teacher[1][:]
    # del model.samples_teacher[2][:]
    del model.samples_student[:]


def main(teacher):
    running_reward = 10

    for i_episode in count(1):
        # random initialization
        state1 = env1.reset()
        state2 = env2.reset()
        teacher_student = np.random.randint(2)

        for t in range(10000):  # Don't infinite loop while learning
            if t % 2 == 0:
                state = state1  # variable used for finishing
                # train environment 1 half the time
                action = select_action(state1, 1, teacher, teacher_student)
                state1, reward, done, _ = env1.step(action)
                reward = max(min(reward, 1), -1)
                model.rewards_student[t%2+1].append(reward)
                if args.render:
                    env1.render()
                if done:
                    break
            if t % 2 == 1:
                # train environment 2 other half of the time
                state = state2  # variable used for finishing
                action = select_action(state2, 2, teacher, teacher_student)
                state2, reward, done, _ = env2.step(action)
                reward = max(min(reward, 1), -1)
                model.rewards_student[t%2+1].append(reward)
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

class TeacherPolicy(nn.Module):
    def __init__(self):
        super(TeacherPolicy, self).__init__()
        # shared layer
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

        # initialize environment 1 head
        self.apply(weights_init)
        self.mu_head_env1.weight.data = normalized_columns_initializer\
            (self.mu_head_env1.weight.data, 0.01)
        self.sigma2_head_env1.weight.data = normalized_columns_initializer\
            (self.sigma2_head_env1.weight.data, 0.01)
        self.mu_head_env1.bias.data.fill_(0)
        self.sigma2_head_env1.bias.data.fill_(0)

        # initialize environment 2 head
        self.apply(weights_init)
        self.mu_head_env2.weight.data = normalized_columns_initializer\
            (self.mu_head_env2.weight.data, 0.01)
        self.sigma2_head_env2.weight.data = normalized_columns_initializer\
            (self.sigma2_head_env2.weight.data, 0.01)
        self.mu_head_env2.bias.data.fill_(0)
        self.sigma2_head_env2.bias.data.fill_(0)

        #initialization for the value heads
        self.value_head_env1.weight.data = normalized_columns_initializer(self.value_head_env1.weight.data, 1.0)
        self.value_head_env1.bias.data.fill_(0)

        self.value_head_env2.weight.data = normalized_columns_initializer(self.value_head_env2.weight.data, 1.0)
        self.value_head_env2.bias.data.fill_(0)

        # initialize lists for holding run information
        self.saved_actions_env1 = []
        self.entropies = []
        self.rewards_env1 = []
        self.saved_actions_env2 = []
        self.rewards_env2 = []

    def forward(self, x):
        '''updated to have 5 return values (2 for each action head one for
        value'''
        x = F.relu(self.affine1(x))
        return self.mu_head_env1(x), F.softplus(self.sigma2_head_env1(x)),\
               self.mu_head_env2(x), F.softplus(self.sigma2_head_env2(x)),\
               self.value_head_env1(x), self.value_head_env2(x)


if __name__ == '__main__':
    teacher = torch.load('teacher_invert_only.pt')
    teacher_mod = TeacherPolicy()
    teacher_mod.load_state_dict(teacher['model_state_dict'])
    main(teacher_mod)
