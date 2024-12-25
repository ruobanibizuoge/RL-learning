import pickle
import random

import gym
import numpy as np
import matplotlib.pyplot as plt
import time

from matplotlib import font_manager

start_time = time.time()  # 记录开始时间

# 创建 CartPole 环境
env = gym.make('CartPole-v1', render_mode='rgb-array')

# 定义离散状态空间
cart_position_bins = np.linspace(-4.8, 4.8, 5)
cart_velocity_bins = np.linspace(-0.5, 0.5, 9)
pole_angle_bins = np.linspace(-0.20944*2, 0.20944*2, 5)  # ±12 degrees in radians
pole_velocity_bins = np.linspace(-np.radians(50), np.radians(50), 9)


# 状态离散化
def discretize_state(state):
    cart_position, cart_velocity, pole_angle, pole_velocity = state
    return (
        np.digitize(cart_position, cart_position_bins),
        np.digitize(cart_velocity, cart_velocity_bins),
        np.digitize(pole_angle, pole_angle_bins),
        np.digitize(pole_velocity, pole_velocity_bins)
    )


# 初始化参数
state_space_size = (len(cart_position_bins) + 1, len(cart_velocity_bins) + 1,
                    len(pole_angle_bins) + 1, len(pole_velocity_bins) + 1)
action_space_size = env.action_space.n

# 初始化 Q 值和计数
Q = np.zeros(state_space_size + (action_space_size,))
returns = np.zeros(state_space_size + (action_space_size,))
state_action_count = np.zeros(state_space_size + (action_space_size,))

# 折扣因子
gamma = 1

def monte_carlo(num_episodes):
    for episode in range(num_episodes):
        state = discretize_state(env.reset()[0])
        done = False
        episode_data = []

        while not done:
            action = env.action_space.sample()
            # action = policy(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_observation)
            episode_data.append((state, action, reward))
            state = next_state
            done = terminated or truncated


        # 计算每个状态-动作对的回报
        G = 0
        for state, action, reward in reversed(episode_data):
            G = reward + (gamma * G)

            # 增量均值写法
            state_action_count[state][action] += 1
            Q[state][action] += (G - Q[state][action]) / state_action_count[state][action]
            # 经验均值写法

            # returns[state][action] += G
            # Q[state][action] = returns[state][action] / state_action_count[state][action]  # 更新 Q 值

        if episode % 100 == 0:
            print(episode)
    print(len(episode_data))


# 运行 Monte Carlo 方法
num_episodes = 20000
monte_carlo(num_episodes)


# # 测试学习到的策略
# with open('policy.pkl', 'rb') as f:
#     Q = pickle.load(f)
def run_policy():

    x = []
    y = []
    data = np.zeros(5)
    for episode in range(100):
        state = discretize_state(env.reset()[0])
        done = False
        total_reward = 0

        while not done:
            action = np.argmax(Q[state])
            next_observation, reward, terminated, truncated, _ = env.step(action)
            state = discretize_state(next_observation)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {episode + 1}: Total reward = {total_reward}")
        x.append(episode + 1)
        y.append(total_reward)
    # 设置字体，选择一个支持中文的字体
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 替换为你的字体路径
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.plot(x, y)
    plt.xlabel('测试次数', fontproperties=font_prop)
    plt.ylabel('总奖励（总步数）', fontproperties=font_prop)
    plt.title('奖励与测试次数关系图', fontproperties=font_prop)
    plt.savefig('monte.png')  # 保存为 PNG 格式
    plt.show()  # Add this line to display the plot



run_policy()
env.close()

# # 保存策略
# with open('policy.pkl', 'wb') as f:
#     pickle.dump(Q, f)

# 如果需要加载策略，可以调用
# with open('policy.pkl', 'rb') as f:
#     Q = pickle.load(f)

end_time = time.time()  # 记录结束时间
print(f"took {end_time - start_time:.2f} seconds.")