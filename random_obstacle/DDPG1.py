#!/usr/bin/env python
#-*- coding: utf-8 -*-
import gym
from openai_fetch.DDPG import DDPG
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
MAX_EPISODES = 5000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 100000
c1 = 100
c2 = 10
c3 = 6
c4 = 50
# ############This noise code is copied from openai baseline
# #########OrnsteinUhlenbeckActionNoise############# Openai Code#########
# 代表fetch的位置，桌子的位置和长宽高， 障碍物的位置和长宽高
ENV = {'obs': [0.1749, 0.48, 0, 1.3, 0.75, 0.2, 0.4, 0.6, 0.2]}
# ENV = {'obs': [0.1749, 0.48, 0, 1.3, 0.75, 0.2, 0.4, 0.6, 0.2]}
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.1, theta=.0, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def CheckCollision(env):
    for i in range(env.sim.data.ncon):  # 表示有多少个接触力
        contact = env.sim.data.contact[i]  # 接触力的数据
        # 输出当前状态下每一个接触力对应的是那两个geom  输出名字
        name = ['obstacle', 'table', 'robot0']
        can_colission = ['robot0:base_link', ]
        links_collission =  ['robot0:torso_lift_link', 'robot0:head_pan_link', 'robot0:head_tilt_link',
                     'robot0:head_camera_link', 'robot0:shoulder_pan_link', 'robot0:shoulder_lift_link',
                     'robot0:upperarm_roll_link', 'robot0:elbow_flex_link', 'robot0:forearm_roll_link',
                    'robot0:wrist_flex_link', 'robot0:wrist_roll_link', 'robot0:gripper_link', 'robot0:r_gripper_finger_link',
                    'robot0:l_gripper_finger_link', 'obstacle', 'table', 'robot0:base_link']
        vis = False
        # print(i)
        str1 = env.sim.model.geom_id2name(contact.geom1)
        str2 = env.sim.model.geom_id2name(contact.geom2)
        # print(str1)
        # print(str2)
        # print("\n")
        for j in can_colission:
            if str1.find(j) >=0 or str2.find(j) >=0 :
                vis = True
        if vis:
            continue
        vis = False
        for j in range(len(links_collission)):
            for k in range(len(links_collission)):
                if str1.find(links_collission[j]) >= 0 and str2.find(links_collission[k]) >= 0:
                    # vis = True
                    # print('geom1', contact.geom1, str1)
                    # print('geom2', contact.geom2, str2)
                    return True # 不允许自碰撞
        # if vis: # 允许自碰撞
        #     continue
        for j in name:
            if str1.find(j) >= 0 or str2.find(j) >= 0:
                    # print('geom1', contact.geom1, str1)
                    # print('geom2', contact.geom2, str2)
                    # print("\n")
                    return True
            # 如果你输出的name是None的话  请检查xml文件中的geom元素是否有名字着一个属性
            # print("wrong")
    return False

def check_space(p):
    if p[0] >= 0.8 and p[0] <= 1.8 and p[1]>=0.2 and p[1]<=1.2 and p[2] >= 0.4 and p[2] <= 1.3:
        return True
    return False

def plot(frame_idx, rewards, acc):
    plt.subplot(121)
    plt.plot(rewards)
    plt.ylabel("Total_reward")
    plt.xlabel("Episode")

    plt.subplot(122)
    plt.plot(acc)
    plt.ylabel("Accuracy(%)")
    plt.xlabel("Episode")
    plt.show()

def huber_loss(dis, delta=0.1):
    if dis < delta:
        return 0.5*dis*dis
    else:
        return delta*(dis - 0.5*delta)

if __name__ == "__main__":
    # OrderedDict([('achieved_goal', Box(3,)), ('desired_goal', Box(3,)), ('observation', Box(10,))])
    env = gym.make("FetchReach-v1")
    rl = DDPG()
    # rl.load_mode()
    var = 0.3
    # noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(7))
    total_rewards = []
    step_sums = []
    acc_epi = []
    # box = np.load('data/box' + str(0) + '.npy')
    root = [0, -1.0, .0, 1.5, .0, 1.0, .0]
    rot = root.copy()
    OBS = np.array(ENV['obs'])
    box = np.array([1.35305097, 0.63599385, 0.60611947])
    # 主循环
    acc = 0
    name = "obstacle02"
    # 获取障碍物的随机范围
    xyz_range = env.get_object_range(name)
    print(xyz_range)
    for i in range(1, MAX_EPISODES):
        # obs = env.reset()
        obs = env.fix_target(box)
        env.test_set_joint(root)
        env.random_set_object(name, xyz_range)
        while np.sqrt(np.sum(np.square(env.get_object(name) - box))) <= 0.1:
            env.random_set_object(name, xyz_range)
        state = obs.copy()
        Box_position = state['desired_goal'].reshape(1, -1)
        print("Box_Position:", Box_position)
        env_obs = np.append(np.array(ENV['obs']), (env.get_object(name)))
        print(env.get_object(name))
        # print(env_obs)
        env_obs = np.append(env_obs, np.array([0.04, 0.04, 0.04]))
        # print(env_obs)
        if i % 50 == 0:
            print("\n------------------Episode:{0}------------------".format(i))
            rl.save_mode()
        st = 0
        rw = 0
        success = False
        success_thd = 0
        while True:
            # ------------ choose action ------------
            s = np.array(env.test_get_joint())
            x = np.concatenate((s, env_obs, Box_position[0]), axis=0)
            noise = np.random.normal(loc=0, scale=var, size=7)
            action = rl.choose_action(x)
            action += noise
            action[2], action[-1] = 0, 0
            collision = False
            next_state = state
            st += 1
            for _ in range(30):
                next_state, _, done, info = env.test_step(action/30)
                env.render()
                collision = CheckCollision(env=env)
                if collision:
                    next_state, _, done, info = env.test_step(-action / 30)
                    break
                if not check_space(next_state['achieved_goal']):
                    # print(next_state['achieved_goal'])
                    next_state, _, done, info = env.test_step(-action / 30)
                    break
            s_next = np.array(env.test_get_joint())
            dis = np.sqrt(np.sum(np.square(Box_position - next_state['achieved_goal']))).copy() # L2 distance
            a = np.sqrt(np.sum(np.square(action))).copy()
            if dis <= 0.05 and not collision:
                success = True
                # success_thd = c4*(float(success) - dis)
                acc += 1
            r = -c1*huber_loss(dis) - c2*a - c3*int(collision)
            # r = -dis - int(collision) * 2
            rw += r
            x1 = np.array(np.concatenate((s_next, env_obs, Box_position[0]), axis=0).reshape(1, -1))
            done = 1 if success or st==MAX_EP_STEPS else 0
            rl.store_transition(x, action, r, x1, done)
            if collision:
                env.test_set_joint(root)
            if rl.memory_counter > MEMORY_CAPACITY:
                env.render()
                rl.learn()
                var *= .999995
            if done:
                print("Step:{0}, total reward:{1}, average reward:{2},{3}".format(st, rw, rw*1.0/st,
                                                                                  'success' if success else '----'))
                total_rewards.append(rw)
                step_sums.append(st)
                break
        if i % 100 == 0:
            print("episode {0}, the accuracy is: {1}%".format(i, acc))
            acc_epi.append(acc)
            acc = 0
        # if i%1000 == 0:
        #     plot(MAX_EPISODES, total_rewards, acc_epi)
    plot(MAX_EPISODES, total_rewards, acc_epi)


# import gym
# #
# if __name__ == "__main__":
#     env = gym.make("FetchReach-v1")
#     name = "obstacle02"
#     xyz_range = env.get_object_range(name)
#     # print(env.get_object(name))
#     print(env.get_object_range(name))
#     for _ in range(100):
#         env.random_set_object(name, xyz_range)
#         env.render()

