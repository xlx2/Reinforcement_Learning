from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from snake_env import SnakeGameEnv

# 创建和包装环境
def create_env(render_mode=None):
    env = SnakeGameEnv(render_mode=render_mode)
    return Monitor(env)

# 创建矢量化环境
def create_vec_env(n_envs=4):
    return make_vec_env(create_env, n_envs=n_envs)

# 创建训练和评估环境
train_env = create_vec_env()
eval_env = create_vec_env(n_envs=1)

# 初始化 A2C 模型
model = A2C(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    learning_rate=7e-4,
    gamma=0.99,
    n_steps=5,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./snake_tensorboard/",
    device="mps",
)

# 设置评估回调
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./snake_best_model/",
    log_path="./snake_eval_logs/",
    eval_freq=10000,
    deterministic=True,
    render=False,
)

# 开始训练
model.learn(total_timesteps=100000, callback=eval_callback)

# 保存模型
model.save("snake_a2c_mlp_model")

# 加载并测试模型
del model
model = A2C.load("snake_a2c_mlp_model")

# 渲染测试
env = SnakeGameEnv(render_mode="human")
obs, info = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs, info = env.reset()
env.close()
