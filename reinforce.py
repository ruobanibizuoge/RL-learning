import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np
from matplotlib import pyplot as plt, font_manager
from torch.distributions import Categorical

GAMMA = 1
lr = 0.01
num_episode = 5000

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()

        self.Linear1 = nn.Linear(input_size, hidden_size)
        # 初始化第一层权重和偏置
        self.Linear1.weight.data.normal_(0, 0.1)
        self.Linear1.bias.data.zero_()

        self.Linear2 = nn.Linear(hidden_size, output_size)
        # 初始化第二层权重和偏置
        self.Linear2.weight.data.normal_(0, 0.1)
        self.Linear2.bias.data.zero_()

    def forward(self, x):
        # 计算第一层的输出
        x = F.relu(self.Linear1(x))
        # 计算第二层的输出
        # x = F.softmax(self.Linear2(x), dim=1)
        x = F.sigmoid(self.Linear2(x))
        return x

class Reinforce:
    def __init__(self, input_size, hidden_size, output_size):
        self.net = Policy(input_size, hidden_size, output_size)
        self.optim = optim.Adam(params=self.net.parameters(), lr=lr)

    def select_action(self, s):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()
        s = s.unsqueeze(0)
        probs = self.net(s)
        tmp = Categorical(probs)
        a = tmp.sample()
        log_prob = tmp.log_prob(a)
        return a.item(), log_prob

    def update_parameters(self, rewards, log_probs):
        G = 0
        loss = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + GAMMA * G
            loss = loss - G * log_probs[i]
        # 梯度清零
        self.optim.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 根据计算的梯度更新参数
        self.optim.step()

if __name__ == '__main__':
    x = []
    y = []
    env = gym.make('CartPole-v1')
    average_reward = 0
    Agent = Reinforce(env.observation_space.shape[0], 16, env.action_space.n)
    # scores_deque = deque(maxlen=100)
    num = 0
    for i_episode in range(1, num_episode + 1):
        s, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        while not done:
            # 这一个回合的动作选取都是根据当前的策略网络来的
            a, prob = Agent.select_action(s)
            s1, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            log_probs.append(prob)
            rewards.append(r)
            s = s1
        if sum(rewards) == 500:
            num += 1
            # break
        else:
            num = 0
        if num >= 100:
            break
        average_reward = average_reward + (1 / (i_episode + 1)) * (np.sum(rewards) - average_reward)
        print('episode: ', i_episode, "tot_rewards: ", np.sum(rewards), 'average_rewards: ', average_reward)
        # x.append(i_episode + 1)
        # y.append(np.sum(rewards))
        # print('Episode {}\t Average Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        Agent.update_parameters(rewards, log_probs)
    # 设置字体，选择一个支持中文的字体
    for i in range(100):
        s, _ = env.reset()
        done = False
        reward = 0
        while not done:
            a, prob = Agent.select_action(s)
            s1, r, terminated, truncated, _ = env.step(a)
            reward += r
            done = terminated or truncated
            s = s1
        x.append(i+1)
        y.append(reward)
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 替换为你的字体路径
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.plot(x, y)
    plt.xlabel('迭代次数', fontproperties=font_prop)
    plt.ylabel('总奖励（总步数）', fontproperties=font_prop)
    plt.title('奖励与迭代次数关系图', fontproperties=font_prop)
    # plt.savefig('./reinforce.png')  # 保存为 PNG 格式
    plt.show()  # Add this line to display the plot