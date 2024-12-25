import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 创建 CartPole 环境
env = gym.make('CartPole-v1')

# 定义状态离散化的区间
cart_position_bins = np.linspace(-4.8, 4.8, 5)
cart_velocity_bins = np.linspace(-0.5, 0.5, 9)
pole_angle_bins = np.linspace(-0.20944 * 2, 0.20944 * 2, 5)  # 大约等于 ±12度 (弧度)
pole_velocity_bins = np.linspace(-np.radians(50), np.radians(50), 9)

def discretize_state(state):
    cart_position, cart_velocity, pole_angle, pole_velocity = state
    state_discrete = [
        np.digitize(cart_position, cart_position_bins),
        np.digitize(cart_velocity, cart_velocity_bins),
        np.digitize(pole_angle, pole_angle_bins),
        np.digitize(pole_velocity, pole_velocity_bins)
    ]
    return tuple(state_discrete)


# 初始化参数
state_space_size = (len(cart_position_bins) + 1, len(cart_velocity_bins) + 1,
                    len(pole_angle_bins) + 1, len(pole_velocity_bins) + 1)
action_space_size = env.action_space.n

# 初始化值函数表 V 和策略 π
V = np.zeros(state_space_size)
policy = np.zeros(state_space_size, dtype=int)  # 动作是 0 或 1

# 值迭代算法的参数
gamma = 1  # 折扣因子

R = np.zeros(state_space_size + (action_space_size,))
returns = np.zeros(state_space_size + (action_space_size,))
state_action_count = np.zeros(state_space_size + (action_space_size,))

state_action_state = np.zeros(state_space_size + (action_space_size,) + state_space_size)
p_total = np.zeros(state_space_size + (action_space_size,) + state_space_size)

# 初始化状态转移字典
transitions = {}


def update_transitions(state, action, next_state):
    if state not in transitions:
        transitions[state] = {}
    if action not in transitions[state]:
        transitions[state][action] = {}
    if next_state not in transitions[state][action]:
        transitions[state][action][next_state] = 0
    transitions[state][action][next_state] += 1

def monte_carlo(num_episodes):
    for episode in range(num_episodes):
        state = discretize_state(env.reset()[0])
        done = False
        episode_data = []

        while not done:
            # action = choose_policy(state)
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_observation)
            episode_data.append((state, action, reward))
            update_transitions(state, action, next_state)
            state = next_state
            done = terminated or truncated
        for state, action, reward in reversed(episode_data):
            returns[state][action] += reward
            state_action_count[state][action] += 1
            # 计算每个状态-动作对的即时回报
            R[state][action] = returns[state][action] / state_action_count[state][action]  # 更新 R 值
        if episode % 100 == 0:
            print(episode)

    for state in transitions.keys():
        for action in transitions[state].keys():
            for next_state in transitions[state][action].keys():
                p_total[state][action][next_state] = transitions[state][action][next_state] / state_action_count[state][action]


# 值迭代算法
def value_iteration(max_iterations):
    for iteration in range(max_iterations + 1):
        for state in transitions.keys():
            action_values = []
            for action in sorted(transitions[state].keys()):
                value = R[state][action]
                for next_state in transitions[state][action].keys():
                    value += p_total[state][action][next_state] * (gamma * V[next_state])
                action_values.append((value, action))
            m = 0
            a = 0
            for q, action in action_values:
                if q > m:
                    m = q
                    a = action
            # 更新值函数
            V[state] = m
            # 最后更新策略
            if iteration == max_iterations:
                policy[state] = a
        if iteration % 100 == 0:
            print(iteration)


# 使用学习到的策略进行模拟测试
def run_policy(env, episodes=100):
    x = []
    y = []
    for episode in range(episodes):

        observation, _ = env.reset()  # 获取观测状态
        state = discretize_state(observation)
        done = False
        total_reward = 0
        while not done:
            env.render()
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = discretize_state(next_state)
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
    plt.savefig('total_reward_plot.png')  # 保存为 PNG 格式
    plt.show()  # Add this line to display the plot


def main():
    num = 100000
    monte_carlo(num)
    print("monte finish")
    max_iterations = 1000  # 最大迭代次数
    value_iteration(max_iterations)
    # 运行测试，使用学习到的策略
    run_policy(env)


main()
