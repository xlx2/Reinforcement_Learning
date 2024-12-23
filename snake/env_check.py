from stable_baselines3.common.env_checker import check_env
from snake.snake_env import SnakeGameEnv

env = SnakeGameEnv(render_mode="human")
check_env(env)  # 检测环境

episodes = 2
for episode in range(episodes):
    done = False
    env.reset()
    while not done:
        action = env.action_space.sample()
        print("action", action)
        obs, reward, done, info, _ = env.step(action)
        print('reward', reward)
        env.render()
env.close()
