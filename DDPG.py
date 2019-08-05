#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
import numpy as np
import os
import math
import random
import time

#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.002      # soft replacement
MEMORY_CAPACITY = 100000
DMEMORY_CAPACITY = 17000
BATCH_SIZE = 128
DBATCH_SIZE = 0
N_STATES = 3
RENDER = False
EPSILON = 0.9
ENV_PARAMS = {'obs': 16, 'goal': 3, 'action': 7, 'action_max': 0.2}
# ##############################  DDPG  ####################################


class ANet(nn.Module):   # ae(s)=a
    def __init__(self, init_w = 3e-3):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(ENV_PARAMS['obs'] + ENV_PARAMS['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, ENV_PARAMS['action'])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                # nn.init.normal_(m.weight, 0, 0.1)

        # self.fc3.weight.data.uniform_(-init_w, init_w)
        # self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, s):
        a = F.relu(self.fc1(s))
        x = F.relu(self.fc2(a))
        x = torch.tanh(self.fc3(x))
        return x


class CNet(nn.Module):
    def __init__(self, init_w = 3e-3):
        super(CNet, self).__init__()
        self.fc1 = nn.Linear(ENV_PARAMS['obs'] + ENV_PARAMS['action']+ENV_PARAMS['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, ENV_PARAMS['action'])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                # nn.init.normal_(m.weight, 0, 0.1)
        # self.fc3.weight.data.uniform_(-init_w, init_w)
        # self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDPG(object):
    def __init__(self):
        self.memory = np.zeros((MEMORY_CAPACITY, (ENV_PARAMS['obs'] + ENV_PARAMS['goal'])*2 + ENV_PARAMS['action'] + 2))
        self.Dmemory = np.zeros((DMEMORY_CAPACITY, (ENV_PARAMS['obs'] + ENV_PARAMS['goal'])*2 + ENV_PARAMS['action'] + 2))
        self.memory_counter = 0  # 记忆库计数
        self.Dmemory_counter = 0
        self.Actor_eval = ANet().cuda()
        self.Actor_target = ANet().cuda()
        self.Critic_eval = CNet().cuda()
        self.Critic_target = CNet().cuda()
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.L1Loss = nn.SmoothL1Loss()
        self.f = 0

    def choose_action(self, s):
        state = torch.FloatTensor(s).cuda()
        return self.Actor_eval(state).cpu().data.numpy()

    def learn(self):
        self.f += 1
        self.f %= 50
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        sample_Dindex = np.random.choice(DMEMORY_CAPACITY, DBATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        # b_dmemory = self.Dmemory[sample_Dindex, :]
        b_s_num = ENV_PARAMS['obs'] + ENV_PARAMS['goal']
        b_a_num = ENV_PARAMS['action']

        # b_s = torch.FloatTensor(np.concatenate((b_memory[:, :b_s_num], b_dmemory[:, :b_s_num])).reshape(-1, b_s_num)).cuda()
        # b_a = torch.FloatTensor(np.concatenate((b_memory[:, b_s_num:b_s_num + b_a_num], b_dmemory[:, b_s_num:b_s_num + b_a_num])).reshape(-1, b_a_num)).cuda()
        # b_r = torch.FloatTensor(np.concatenate((b_memory[:, b_s_num + b_a_num:b_s_num + b_a_num + 1], b_dmemory[:, b_s_num + b_a_num:b_s_num + b_a_num + 1])).reshape(-1, 1)).cuda()
        # b_s_ = torch.FloatTensor(np.concatenate((b_memory[:, b_s_num + b_a_num + 1:2*b_s_num+b_a_num + 1], b_dmemory[:, b_s_num + b_a_num + 1:2*b_s_num+b_a_num + 1])).reshape(-1, b_s_num)).cuda()
        b_s = torch.FloatTensor(b_memory[:, :b_s_num]).reshape(-1, b_s_num).cuda()
        b_a = torch.FloatTensor(b_memory[:, b_s_num:b_s_num + b_a_num]).reshape(-1, b_a_num).cuda()
        b_r = torch.FloatTensor(b_memory[:, b_s_num + b_a_num:b_s_num + b_a_num + 1]).reshape(-1, 1).cuda()
        b_s_ = torch.FloatTensor(b_memory[:, b_s_num + b_a_num + 1:2*b_s_num+b_a_num + 1]).reshape(-1, b_s_num).cuda()
        done = torch.FloatTensor(1-b_memory[:, -1:]).reshape(-1, 1).cuda()

        # Compute the target Q value
        target_Q = self.Critic_target(b_s_, self.Actor_target(b_s_))
        target_Q = b_r + (done*GAMMA * target_Q).detach()

        # Get current Q estimate
        current_Q = self.Critic_eval(b_s, b_a)

        # Compute critic loss
        critic_loss = self.loss_td(current_Q, target_Q)

        # Optimize the critic
        self.ctrain.zero_grad()
        critic_loss.backward()
        self.ctrain.step()

        # Compute actor loss
        actor_loss = -self.Critic_eval(b_s, self.Actor_eval(b_s)).mean().cuda()
        if self.f == 0:
            print("actor_loss:", actor_loss)
            # print("critic_loss:", critic_loss)
        # Optimize the actor
        self.atrain.zero_grad()
        actor_loss.backward()
        self.atrain.step()

        self.soft_update(self.Actor_target, self.Actor_eval, TAU)
        self.soft_update(self.Critic_target, self.Critic_eval, TAU)

    def store_transition(self, s, a, r, s_, d):

        s_num = ENV_PARAMS['obs'] + ENV_PARAMS['goal']
        a_num = ENV_PARAMS['action']
        s = np.array(s).reshape(-1, s_num)
        a = np.array(a).reshape(-1, a_num)
        r = np.array(r).reshape(-1, 1)
        s_ = np.array(s_).reshape(-1, s_num)
        d = np.array(d).reshape(-1, 1)
        transition = np.hstack((s, a, r, s_, d))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def store_Dtransition(self, s, a, r, s_):
        s_num = ENV_PARAMS['obs'] + ENV_PARAMS['goal']
        a_num = ENV_PARAMS['action']
        s = np.array(s).reshape(-1, s_num)
        a = np.array(a).reshape(-1, a_num)
        r = np.array(r).reshape(-1, 1)
        s_ = np.array(s_).reshape(-1, s_num)
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.Dmemory[index, :] = transition
        self.Dmemory_counter += 1

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data*(1.0 - tau) + param.data*tau
            )

    def save_mode(self):
        torch.save(self.Actor_eval, "model/Actor_eval1.pkl")
        torch.save(self.Actor_target, "model/Actor_target1.pkl")
        torch.save(self.Critic_eval, "model/Critic_eval1.pkl")
        torch.save(self.Critic_target, "model/Critic_target1.pkl")

    def load_mode(self):
        self.Actor_eval = torch.load("model/Actor_eval1.pkl")
        self.Actor_target = torch.load("model/Actor_target1.pkl")
        self.Critic_eval = torch.load("model/Critic_eval1.pkl")
        self.Critic_target = torch.load("model/Critic_target1.pkl")
