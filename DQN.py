import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
# 设置 Matplotlib 字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

import math
import time
GAMMA = 1
lr = 0.00012
# 0.00012 700左右
# 小于0.00012比较稳定，但收敛速度慢
# 目标网络更新时使用0.0001效果不好 0.001大概2500次
# 学习率设置较大时，收敛速度更快，但容易过拟合，有时会出现震荡
# 学习率设置较小时，收敛速度较慢，但不容易过拟合，稳定性更好

EPSION = 0.1
buffer_size = 50000  # replay池的大小
batch_size = 32
num_episode = 100000
target_update = 5  # 每过多少个episode将net的参数复制到target_net
# 10

# 在文件开头添加设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 定义神经网络
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear1.weight.data.normal_(0, 0.1)
        self.Linear1.bias.data.zero_()
        self.Linear2 = nn.Linear(hidden_size, hidden_size)
        self.Linear2.weight.data.normal_(0, 0.1)
        self.Linear2.bias.data.zero_()
        self.Linear3 = nn.Linear(hidden_size, output_size)
        self.Linear3.weight.data.normal_(0, 0.1)
        self.Linear3.bias.data.zero_()


    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x


# nametuple容器
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))


# 经验池，保存较好的经验，可以有效防止震荡的问题

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.good_memory = []  # 存储高奖励的经验
        self.position = 0

    def push(self, *args):
        transition = Transition(*args)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        
        # 保存高奖励的经验
        if args[2] > 400:  # 奖励阈值
            self.good_memory.append(transition)
            if len(self.good_memory) > self.capacity // 5:  # 限制好经验的数量 5比10效果好
                self.good_memory.pop(0)
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # 混合采样策略
        normal_samples = random.sample(self.memory, batch_size - batch_size//4)
        good_samples = random.sample(self.good_memory, batch_size//4) if self.good_memory else []
        return normal_samples + good_samples

    def __len__(self):
        return len(self.memory)


class DQN(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.net = Net(input_size, hidden_size, output_size).to(device)
        self.target_net = Net(input_size, hidden_size, output_size).to(device)
        self.optim = optim.Adam(self.net.parameters(), lr=lr)

        self.target_net.load_state_dict(self.net.state_dict())
        self.buffer = ReplayMemory(buffer_size)
        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.HuberLoss(delta=1.0)
        self.steps_done = 0
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200

    def put(self, s0, a0, r, t, s1):
        self.buffer.push(s0, a0, r, t, s1)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if sample > eps_threshold:
            # 利用
            with torch.no_grad():
                return torch.argmax(self.net(torch.FloatTensor(state).to(device))).item()
        else:
            # 探索
            return random.randrange(self.net.Linear3.out_features)

    def update_parameters(self):
        if self.buffer.__len__() < batch_size:
            return
        samples = self.buffer.sample(batch_size)
        batch = Transition(*zip(*samples))
        
        state_batch = torch.Tensor(batch.state).to(device)
        action_batch = torch.LongTensor(np.vstack(batch.action).astype(int)).to(device)
        reward_batch = torch.Tensor(batch.reward).to(device)
        done_batch = torch.Tensor(batch.done).to(device)
        next_state_batch = torch.Tensor(batch.next_state).to(device)

        # 使用Double DQN
        with torch.no_grad():
            # 用当前网络选择动作
            next_action = self.net(next_state_batch).max(1)[1].unsqueeze(1)
            # 用目标网络评估价值
            next_state_values = self.target_net(next_state_batch).gather(1, next_action)
            q_tar = reward_batch.unsqueeze(1) + (1-done_batch.unsqueeze(1)) * GAMMA * next_state_values

        q_eval = self.net(state_batch).gather(1, action_batch)
        loss = self.loss_func(q_eval, q_tar)
        
        self.optim.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        self.optim.step()

# 设置 Matplotlib 字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def test(Agent, num_episode=100):
    x = []
    y = []
    for i in range(num_episode):
        s0, _ = env.reset()
        tot_reward = 0
        while True:
            a0 = Agent.select_action(s0)
            s1, r, terminated, truncated, _ = env.step(a0)
            tot_reward += r
            if terminated or truncated:
                break
            s0 = s1
        x.append(i+1)
        y.append(tot_reward)
    plt.plot(x, y, label='DQN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN on CartPole-v1')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    # 状态空间：4维
    # 动作空间：1维，并且是离散的，只有0和1两个动作
    Agent = DQN(env.observation_space.shape[0], 256, env.action_space.n)
    average_reward = 0  # 目前所有的episode的reward的平均值
    num = 0
    x = []
    y = []

    start_time = time.time()
    for i_episode in range(num_episode):

        s0, _ = env.reset()
        tot_reward = 0  # 每个episode的总reward
        while True:
            if num >= 100:
                break
            a0 = Agent.select_action(s0)
            s1, r, terminated, truncated, _ = env.step(a0)
            tot_reward += r  # 计算当前episode的总reward
            if (terminated or truncated) and tot_reward < 500:
                r = -2
                t = 1
            else:
                t = 0
            Agent.put(s0, a0, r, t, s1)  # 放入replay池
            s0 = s1
            Agent.update_parameters()
            if terminated or truncated:
                if tot_reward >= 400:
                    num += 1
                else:
                    num = 0
                average_reward = average_reward + 1 / (i_episode + 1) * (
                        tot_reward - average_reward)
                print('Episode ', i_episode, ' tot_reward: ', tot_reward, ' average_reward: ', average_reward)
                x.append(i_episode)
                y.append(tot_reward)
                break
        if i_episode % target_update == 0:
            Agent.target_net.load_state_dict(Agent.net.state_dict())
    plt.plot(x, y, label='DQN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('简化PER训练过程')
    plt.legend()
    plt.show()
    end_time = time.time()
    print(f"训练时间: {end_time - start_time:.2f} 秒")
    test(Agent, 100)
    