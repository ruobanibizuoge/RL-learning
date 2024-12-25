import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import math

GAMMA = 1
lr = 0.00012
EPSION = 0.1
num_episode = 100000
target_update = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

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

class DQN(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.net = Net(input_size, hidden_size, output_size).to(device)
        self.target_net = Net(input_size, hidden_size, output_size).to(device)
        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        
        self.target_net.load_state_dict(self.net.state_dict())
        self.loss_func = nn.MSELoss()
        self.steps_done = 0
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax(self.net(torch.FloatTensor(state).to(device))).item()
        else:
            return random.randrange(self.net.Linear3.out_features)

    def update_parameters(self, state, action, reward, done, next_state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        action = torch.LongTensor([[action]]).to(device)
        reward = torch.FloatTensor([[reward]]).to(device)
        done = torch.FloatTensor([[done]]).to(device)

        # Double DQN
        with torch.no_grad():
            next_action = self.net(next_state).max(1)[1].unsqueeze(1)
            next_state_values = self.target_net(next_state).gather(1, next_action)
            q_tar = reward + (1-done) * GAMMA * next_state_values

        q_eval = self.net(state).gather(1, action)
        loss = self.loss_func(q_eval, q_tar)
        
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        self.optim.step()

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
    plt.plot(x, y, label='DQN without Replay Memory')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN without Replay Memory on CartPole-v1')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    Agent = DQN(env.observation_space.shape[0], 256, env.action_space.n)
    average_reward = 0
    num = 0
    
    for i_episode in range(num_episode):
        s0, _ = env.reset()
        tot_reward = 0
        while True:
            if num >= 100:
                break
            a0 = Agent.select_action(s0)
            s1, r, terminated, truncated, _ = env.step(a0)
            tot_reward += r

            if terminated or truncated:
                r = -2
                t = 1
            else:
                t = 0
                
            # 直接更新网络，不使用经验回放
            Agent.update_parameters(s0, a0, r, t, s1)
            s0 = s1
            
            if terminated or truncated:
                if tot_reward >= 400:
                    num += 1
                else:
                    num = 0
                average_reward = average_reward + 1 / (i_episode + 1) * (tot_reward - average_reward)
                print('Episode ', i_episode, ' tot_reward: ', tot_reward, ' average_reward: ', average_reward)
                break
                
        if i_episode % target_update == 0:
            Agent.target_net.load_state_dict(Agent.net.state_dict())
            
    test(Agent, 100)
