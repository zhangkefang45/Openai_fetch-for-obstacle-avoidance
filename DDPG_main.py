#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
from openai_fetch.DDPG import DDPG
import numpy as np
import math
import matplotlib.pyplot as plt
MAX_EPISODES = 20000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 100000
# ############This noise code is copied from openai baseline
# #########OrnsteinUhlenbeckActionNoise############# Openai Code#########
# 代表fetch的位置，桌子的位置和长宽高， 障碍物的位置和长宽高
# ENV = {'obs': [0.1749, 0.48, 0, 1.3, 0.75, 0.2, 0.4, 0.6, 0.2, 1.2, 0.85, 0.6, 0.04, 0.02, 0.25]}
ENV = {'obs': [0.1749, 0.48, 0, 1.3, 0.75, 0.2, 0.4, 0.6, 0.2]}
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
    if p[0] >= 0.8 and p[0] <= 1.8 and p[1]>=0.2 and p[1]<=1.2 and p[2] >= 0.4 and p[2] <= 1.5:
        return True
    return False

def plot(frame_idx, rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

if __name__ == "__main__":
    # OrderedDict([('achieved_goal', Box(3,)), ('desired_goal', Box(3,)), ('observation', Box(10,))])
    env = gym.make("FetchReach-v1")
    rl = DDPG()
    # rl.load_mode()
    var = 0.3
    # noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(7))
    total_rewards = []
    step_sums = []
    # box = np.load('data/box' + str(0) + '.npy')
    root = [0, -1.0, .0, 1.5, .0, 1.0, .0]
    rot = root.copy()
    OBS = np.array(ENV['obs'])
    box = np.array([1.28305097, 0.63599385, 0.60611947])
    '''
    载入专家数据
    '''
    # for f in range(200):
    #     try:
    #         path = np.load('data/path' + str(f) + '.npy')
    #         path = np.array(path)
    #         print(f)
    #         rot = np.array(root)
    #         env.test_set_joint(root)
    #         for ep in range(2):
    #             for pa in path:
    #                 for _ in range(10):
    #                     tmp = rot+(pa/10)
    #                     obs1, _, _, _ = env.test_step(pa / 10)
    #                     # env.render()
    #                     r = -np.sqrt(np.sum(np.square(Box_position -obs1['achieved_goal']))) + 1
    #                     rl.store_Dtransition(np.concatenate((rot, OBS, box), axis=0).reshape(1, -1), pa, r,
    #                                         np.concatenate((tmp, OBS, box), axis=0).reshape(1, -1))
    #                     rot = tmp
    #     except FileNotFoundError:
    #          continue
    # print(rl.Dmemory_counter)
    # for f in range(10000):
    #     rl.learn()
    # 主循环
    for i in range(1, MAX_EPISODES):
        obs = env.reset()
        # obs = env.fix_target(box)
        env.test_set_joint(root)
        state = obs.copy()
        Box_position = state['desired_goal'].reshape(1, -1)
        print("Box_Position:", Box_position)

        if i % 50 == 0:
            print("\n------------------Episode:{0}------------------".format(i))
            rl.save_mode()
        st = 0
        rw = 0
        success = False
        while True:
            # ------------ choose action ------------
            s = np.array(env.test_get_joint())
            # now_action = np.array([s[0], s[1], s[3]]).reshape(1, -1)
            x = np.concatenate((s, ENV['obs'], Box_position[0]), axis=0)
            noise = np.random.normal(loc=0, scale=var, size=7)
            action = rl.choose_action(x)
            action += noise
            action[2], action[-1] = 0, 0
            # ac = np.append(action, [0, 0, 0])
            # ac = np.insert(ac, 2, 0)
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
                # if np.sqrt(np.sum(np.square(Box_position - next_state['achieved_goal']))) <= 0.05:
                #     ac = env.test_get_joint() - s
                #     break
            # print(next_state['achieved_goal'])
            s_next = np.array(env.test_get_joint())
            # action_after = np.array([s_next[0], s_next[1], s_next[3]]).reshape(1, -1)
            dis = np.sqrt(np.sum(np.square(Box_position - next_state['achieved_goal']))).copy()
            if dis <= 0.05:
                success = True
            # r = r - int(collision) * 3
            # print(env.test_get_joint(), "jjjjj")
            r = -dis
            rw += r
            x1 = np.array(np.concatenate((s_next, ENV['obs'], Box_position[0]), axis=0).reshape(1, -1))
            done = 1 if success or st==MAX_EP_STEPS else 0
            rl.store_transition(x, action, r, x1, done)
            # if collision:
            #     next_state, _, done, info = env.test_step(-action)
            # ----------- ---------------- -----------
            if rl.memory_counter > MEMORY_CAPACITY:
                env.render()
                # print("action is :", action)
                rl.learn()
                # if st%50 == 0:
                    # rl.Actor_eval
                    # print("action:", action)
                    # print("noise", noise)
                var *= .9995
            if done:
                print("Step:{0}, total reward:{1}, average reward:{2},{3}".format(st, rw, rw*1.0/st,
                                                                                  'success' if success else '----'))
                total_rewards.append(rw)
                step_sums.append(st)
                break
    plot(MAX_EPISODES, total_rewards)