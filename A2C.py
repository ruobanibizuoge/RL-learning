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

# 设置 Matplotlib 字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        # 修改softmax的维度：如果是单个状态，使用dim=0；如果是批量状态，使用dim=1
        if len(states.shape) == 1:  # 单个状态
            out = F.softmax(self.Linear2(out.unsqueeze(0)), dim=1).squeeze(0)
        else:  # 批量状态
            out = F.softmax(self.Linear2(out), dim=1)
        return out
    
class A2C:
    def __init__(self, state_dim, hidden_dim, action_dim, lr_c, lr_a, gamma, epochs):
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.actor = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.gamma = gamma
        self.epochs = epochs

    def select_action(self, state):
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float32).to(device)  # 不需要unsqueeze
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(device)
        
        # 计算TD目标
        td_target = rewards + self.gamma * self.critic(next_states).detach() * (1 - dones)
        td_delta = td_target - self.critic(states)
        td_delta = td_delta.detach().cpu().numpy()

        for i in range(self.epochs):
            # 计算critic的损失
            loss_critic = F.mse_loss(self.critic(states), td_target.detach())
            
            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # 添加梯度裁剪
            self.critic_optimizer.step()

            # 计算actor的损失
            td_delta_tensor = torch.tensor(td_delta).float().to(device)
            log_probs = torch.log(self.actor(states).gather(1, actions)).squeeze()
            loss_actor = -(log_probs * td_delta_tensor).mean()
            
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # 添加梯度裁剪
            self.actor_optimizer.step()

def test(agent, num_episodes = 100):
    env = gym.make("CartPole-v1")
    returns = []
    for i in range(num_episodes):
        s, _ = env.reset()
        total_reward = 0
        while True:
            a = agent.select_action(s)
            s1, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_reward += r
            s = s1
            if done:
                break
        print(f"Episode {i+1}: Return = {total_reward}")
        returns.append(total_reward)
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('A2C测试过程')
    plt.show()

if __name__ == "__main__":
    num_episode = 100000
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 获取状态维度
    action_dim = env.action_space.n  # 获取动作维度
    agent = A2C(state_dim=state_dim, 
           hidden_dim=512,
           action_dim=action_dim, 
           lr_c=2e-4,  # 降低学习率
           lr_a=2e-4,  # 降低学习率
           gamma=1,  # 修改折扣因子
           epochs=10)  # 减少更新次数以提高稳定性
    num = 0

    returns = []
    for i in range(num_episode):
        s0, _ = env.reset()
        total_reward = 0
        transition_dict = {
            'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        done = False
        while not done:
            a = agent.select_action(s0)
            s1, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_reward += r
            record_r = r
            if total_reward >= 400:
                record_r = 2
            # 将所有数据转换为numpy数组再存储
            transition_dict['states'].append(np.array(s0))
            transition_dict['actions'].append(a)
            transition_dict['next_states'].append(np.array(s1))
            transition_dict['rewards'].append(record_r)
            transition_dict['dones'].append(done)
            
            s0 = s1
            
        print(f"Episode {i+1}: Return = {total_reward}")
        returns.append(total_reward)
        if total_reward >= 500:
            num += 1
        else:
            num = 0
        if num >= 100:
            break
        # 在更新之前将列表转换为numpy数组
        for key in transition_dict.keys():
            transition_dict[key] = np.array(transition_dict[key])
            
        agent.update(transition_dict)
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('A2C训练过程')
    plt.show()
    test(agent)
    