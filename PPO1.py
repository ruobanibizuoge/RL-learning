import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os

class Critic(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.Linear1 = nn.Linear(state_dim, hidden_dim)
        self.Linear1.weight.data.normal_(0, 0.1)
        self.Linear1.bias.data.zero_()

        self.Linear2 = nn.Linear(hidden_dim, 1)
        self.Linear2.weight.data.normal_(0, 0.1)
        self.Linear2.bias.data.zero_()


    def forward(self, states):
        out = F.relu(self.Linear1(states))
        out = self.Linear2(out)
        return out


class Actor(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.Linear1 = nn.Linear(state_dim, hidden_dim)
        self.Linear1.weight.data.normal_(0, 0.1)
        self.Linear1.bias.data.zero_()
        
        self.Linear2 = nn.Linear(hidden_dim, action_dim)
        self.Linear2.weight.data.normal_(0, 0.1)
        self.Linear2.bias.data.zero_()

    def forward(self, states):
        out = F.relu(self.Linear1(states))
        out = F.softmax(self.Linear2(out), dim=1)
        return out


class PPO:

    def __init__(self, state_dim, hidden_dim, action_dim, gamma, lamda, epochs, eps):
        self.actor = Actor(state_dim, hidden_dim, action_dim)
        self.critic = Critic(state_dim, hidden_dim)
        self.action_dim = action_dim
        self.gamma = gamma
        self.lamda = lamda
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=2e-3)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=2e-3)
        self.epochs = epochs
        self.eps = eps
        # KL散度的目标值
        self.target_kl = 0.01
        # KL惩罚系数
        self.beta = 1
        # 自适应KL系数的参数
        self.beta_up = 1.5
        self.beta_down = 0.5
    def update(self, data):
        # 获取数据
        states = torch.tensor(data['states'], dtype=torch.float)
        actions = torch.tensor(data['actions'], dtype=torch.long).view(-1, 1)
        next_states = torch.tensor(data['next_states'], dtype=torch.float)
        rewards = torch.tensor(data['rewards'], dtype=torch.float)
        dones = torch.tensor(data['done'], dtype=torch.long)

        # 获取旧策略的概率分布
        old_probs = self.actor(states).detach()
        old_log_probs = torch.log(old_probs.gather(1, actions)).squeeze().detach()
        
        td_target = rewards + (self.gamma * self.critic(next_states).squeeze()) * (1 - dones)
        td_delta = td_target - self.critic(states).squeeze()
        advantages = self.cal_advantage(self.gamma, self.lamda, td_delta)

        for i in range(self.epochs):
            # 获取新策略的概率分布
            new_probs = self.actor(states)
            log_probs = torch.log(new_probs.gather(1, actions)).squeeze()

            # 计算KL散度
            kl = torch.mean(torch.sum(old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10)), dim=1))
            # 计算策略比率
            ratio = torch.exp(log_probs - old_log_probs)
            # 计算策略损失
            policy_loss = -torch.mean(advantages * ratio)
            # 添加KL惩罚项
            actor_loss = policy_loss + self.beta * kl
            # 计算价值损失
            critic_loss = torch.mean(F.mse_loss(self.critic(states).squeeze(), td_target.detach()))
            # 更新网络参数
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # 自适应调整beta
            if kl > self.target_kl * 1.2:
                self.beta *= self.beta_up
            elif kl < self.target_kl * 0.8:
                self.beta *= self.beta_down

    def cal_advantage(self, gamma, lamda, td_delta):
        # 防止梯度计算错误
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lamda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)


    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        return action_dist.sample().item()


def test(agent):
    x = []
    y = []
    for i in range(100):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.take_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        x.append(i+1)
        y.append(total_reward)
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 替换为你的字体路径
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.plot(x, y)
    plt.xlabel('测试次数', fontproperties=font_prop)
    plt.ylabel('总奖励（总步数）', fontproperties=font_prop)
    plt.title('奖励与测试次数关系图', fontproperties=font_prop)
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建图片保存路径
    save_path = os.path.join(current_dir, 'ppo1.png')
    plt.savefig(save_path)  # 保存为 PNG 格式
    plt.show()  # Add this line to display the plot

lr = 2e-3
num_episodes = 5000
hidden_dim = 128
gamma = 1
lamda = 0.95
env_name = 'CartPole-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
epochs = 10
eps = 0.2

agent = PPO(state_dim, hidden_dim, action_dim, gamma, lamda, epochs, eps)

num = 0

for i in range(num_episodes):
    data = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'done': [],
    }
    done = 0
    state, _= env.reset()
    G = 0
    while not done:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        data['states'].append(state)
        data['actions'].append(action)
        data['next_states'].append(next_state)
        data['rewards'].append(reward)
        data['done'].append(done)
        state = next_state
        G += reward
    agent.update(data)
    if G >= 500:
        num += 1
    else:
        num = 0
    if num >= 100:
        print("Solved!")
        break
    print("Episode:", i, "Reward:", G)

test(agent)