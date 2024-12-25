## 目前最优：行为策略：随机策略，目标策略：贪心策略，学习率 = 1/(访问次数+0.5)，gamma=1
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 创建 CartPole 环境
env = gym.make('CartPole-v1')

# 定义状态离散化的区间
cart_position_bins = np.linspace(-4.8, 4.8, 5)
cart_velocity_bins = np.linspace(-0.5, 0.5, 9)
pole_angle_bins = np.linspace(-0.20944 * 2, 0.20944 * 2, 5)
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

# 初始化 Q 值和计数
Q = np.zeros(state_space_size + (action_space_size,))
returns = np.zeros(state_space_size + (action_space_size,))
state_action_count = np.zeros(state_space_size + (action_space_size,))

# 参数
gamma = 1  # 折扣因子
epsilon = 0.8
min_epsilon = 0.001
decay_rate = 0.9999

def qlearning_train(num_episodes):
    global epsilon
    for episode in range(num_episodes):
        state = discretize_state(env.reset()[0])
        done = False

        while not done:
            action = np.argmax(Q[state]) if np.random.rand() > epsilon else env.action_space.sample()
            # action = env.action_space.sample()
            # action = np.argmax(Q[state])
            next_observation, reward, terminated, truncated, _ = env.step(action)

            next_state = discretize_state(next_observation)
            # next_action = np.argmax(Q[next_state]) if np.random.rand() > epsilon else env.action_space.sample()
            # next_action = env.action_space.sample()
            next_action = np.argmax(Q[next_state])

            state_action_count[state][action] += 1
            # Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            alpha = 1.0 / (state_action_count[state][action] + 0.5)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            state = next_state
            done = terminated or truncated

        # epsilon = max(min_epsilon, epsilon * decay_rate)
        if episode % 100 == 0:
            print(episode)


def run_policy(episodes=100):
    m = 500
    x = []
    y = []
    for episode in range(episodes):
        state = discretize_state(env.reset()[0])
        done = False
        total_reward = 0
        while not done:
            env.render()
            action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = discretize_state(next_state)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {episode + 1}: Total reward = {total_reward}")
        m = min(m, total_reward)
        x.append(episode + 1)
        y.append(total_reward)
    print(m)
    # 设置字体，选择一个支持中文的字体
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 替换为你的字体路径
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.plot(x, y)
    # plt.xlabel('测试次数', fontproperties=font_prop)
    # plt.ylabel('总奖励（总步数）', fontproperties=font_prop)
    # plt.title('奖励与测试次数关系图', fontproperties=font_prop)
    # plt.savefig('Sarsa.png')  # 保存为 PNG 格式
    plt.show()  # Add this line to display the plot


def main():
    episodes_Q = 100000
    qlearning_train(episodes_Q)
    run_policy()
    env.close()


main()
