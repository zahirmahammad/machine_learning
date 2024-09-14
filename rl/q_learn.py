import gymnasium as gym

env = gym.make('FrozenLake-v1', desc=None, map_name=None, is_slippery=True, render_mode='human')

episodes = 10

for i in range(episodes):
    obs, info = env.reset()
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        env.render()