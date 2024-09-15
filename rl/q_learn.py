import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle 

# env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")

episodes = 150000


# === Print Environment Information ===
print(f"Observation Space: {env.observation_space.n}")
print(f"Action Space: {env.action_space.n}")



# Q-Learning
q_table = np.zeros([env.observation_space.n, env.action_space.n])
print(f"Q-Table Shape: {q_table.shape}")

# PipeLine
# 1. Initialize Q-Table
# 2. For given state choose random action
# 3. Update Q-Table using update rule => Q(s,a) = Q(s,a) + a(reward + gamma*max(Q(s',a')) - Q(s,a))
# 4. Set epsilon greedy policy
# 5. Repeat step 2-4 until convergence

def run(episodes):
    # Hyperparameters
    alpha = 0.9      # Learning Rate
    gamma = 0.98    # Discount Factor
    epsilon = 1.0   # Exploration Rate - Choose greedy action with (1 - epsilion) probability
    decay_rate = 0.0001
    max_steps = 7
    all_rewards = []
    rand_gen = np.random.default_rng()

    for i in range(episodes):
        state, info = env.reset()
        terminated = False
        Reward = 0
        for _ in range(max_steps):
            if rand_gen.random() < (1 - epsilon):
                a = np.argmax(q_table[state, :])
                print(f"Greedy action: {a}")
            else:
                a = env.action_space.sample()
            new_state, reward, terminated, truncated, info = env.step(a)
            Reward = Reward + reward
            env.render()

            q_table[state, a] = q_table[state, a] + alpha*(reward + gamma*np.max(q_table[new_state, :]) - q_table[state, a])
            state = new_state

        epsilon = max(epsilon - decay_rate, 0)

        if epsilon ==0:
            a = 0.0001

            
        # Record total reward untill that episode
        all_rewards.append(Reward)

        # Print Log
        print(f"Episode {i+1} terminated, Reward: {Reward}, Num of Rewards: {sum(all_rewards)}")

    plt_episode = np.arange(0, episodes)
    plt_rewards = []
    for i in plt_episode:
        plt_rewards.append(sum(all_rewards[:i]))
    plt.plot(plt_episode, plt_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Q-Learning")
    plt.savefig("q_learning.png")

    # Save Q-Table
    f = open("q_table.pkl", "wb")
    pickle.dump(q_table, f)
    f.close()

def test():
    f = open("q_table.pkl", "rb")
    q_table = pickle.load(f)
    f.close()

    state, info = env.reset()
    terminated = False
    for i in q_table:
        print(i)
    while not terminated:
        a = np.argmax(q_table[state, :])
        new_state, reward, terminated, truncated, info = env.step(a)
        env.render()
        state = new_state

# run(episodes)
test()